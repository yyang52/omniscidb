/*
 * Copyright 2021 MapD Technologies, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#ifdef HAVE_SYSTEM_TFS
#ifndef __CUDACC__

#include <vector>
#include "QueryEngine/OmniSciTypes.h"

template <typename T, typename Z>
struct GeoRaster {
  const T bin_dim_meters_;
  const bool geographic_coords_;
  const Z null_sentinel_;
  std::vector<Z> z_;
  T x_min_;
  T x_max_;
  T y_min_;
  T y_max_;
  T x_range_;
  T y_range_;
  T x_meters_per_degree_;
  T y_meters_per_degree_;
  int64_t num_x_bins_;
  int64_t num_y_bins_;
  int64_t num_bins_;
  T x_scale_input_to_bin_;
  T y_scale_input_to_bin_;
  T x_scale_bin_to_input_;
  T y_scale_bin_to_input_;

  template <typename T2, typename Z2>
  GeoRaster(const Column<T2>& input_x,
            const Column<T2>& input_y,
            const Column<Z2>& input_z,
            const double bin_dim_meters,
            const bool geographic_coords,
            const bool align_bins_to_zero_based_grid);

  template <typename T2, typename Z2>
  GeoRaster(const Column<T2>& input_x,
            const Column<T2>& input_y,
            const Column<Z2>& input_z,
            const double bin_dim_meters,
            const bool geographic_coords,
            const bool align_bins_to_zero_based_grid,
            const T x_min,
            const T x_max,
            const T y_min,
            const T y_max);

  inline T get_x_bin(const T input) const {
    return (input - x_min_) * x_scale_input_to_bin_;
  }

  inline T get_y_bin(const T input) const {
    return (input - y_min_) * y_scale_input_to_bin_;
  }

  inline bool is_null(const Z value) const { return value == null_sentinel_; }

  inline bool is_bin_out_of_bounds(const int64_t source_x_bin,
                                   const int64_t source_y_bin) const {
    return (source_x_bin < 0 || source_x_bin >= num_x_bins_ || source_y_bin < 0 ||
            source_y_bin >= num_y_bins_);
  }

  inline Z offset_source_z_from_raster_z(const int64_t source_x_bin,
                                         const int64_t source_y_bin,
                                         const Z source_z_offset) const;

  inline Z fill_bin_from_avg_neighbors(const int64_t x_centroid_bin,
                                       const int64_t y_centroid_bin,
                                       const int64_t bins_radius) const;

  void align_bins_max_inclusive();

  void align_bins_max_exclusive();

  void calculate_bins_and_scales();

  template <typename T2, typename Z2>
  void compute(const Column<T2>& input_x,
               const Column<T2>& input_y,
               const Column<Z2>& input_z);

  int64_t outputDenseColumns(TableFunctionManager& mgr,
                             Column<T>& output_x,
                             Column<T>& output_y,
                             Column<Z>& output_z,
                             const int64_t neighborhood_null_fill_radius) const;
};

// clang-format off
/*
  UDTF: tf_geo_rasterize__cpu_template(TableFunctionManager, Cursor<Column<T> x, Column<T> y, Column<Z> z>, T, bool, int64_t) | filter_table_function_transpose=on -> Column<T> x, Column<T> y, Column<Z> z, T=[float, double], Z=[float, double]
 */
// clang-format on

template <typename T, typename Z>
TEMPLATE_NOINLINE int32_t
tf_geo_rasterize__cpu_template(TableFunctionManager& mgr,
                               const Column<T>& input_x,
                               const Column<T>& input_y,
                               const Column<Z>& input_z,
                               const T bin_dim_meters,
                               const bool geographic_coords,
                               const int64_t null_neighborhood_fill_radius,
                               Column<T>& output_x,
                               Column<T>& output_y,
                               Column<Z>& output_z) {
  if (bin_dim_meters <= 0) {
    return mgr.error_message(
        "tf_geo_rasterize: bin_dim_meters argument must be greater than 0");
  }

  if (null_neighborhood_fill_radius < 0) {
    return mgr.error_message(
        "tf_geo_rasterize: null_neighborhood_fill_radius argument must be greater than "
        "or equal to 0");
  }

  GeoRaster<T, Z> geo_raster(
      input_x, input_y, input_z, bin_dim_meters, geographic_coords, true);
  return geo_raster.outputDenseColumns(
      mgr, output_x, output_y, output_z, null_neighborhood_fill_radius);
}

// clang-format off
/*
  UDTF: tf_geo_rasterize__cpu_template(TableFunctionManager, Cursor<Column<T>, Column<T>, Column<Z>>, T, bool, int64_t, T, T, T, T) -> Column<T> x, Column<T> y, Column<Z> z, T=[float, double], Z=[float, double]
 */
// clang-format on

template <typename T, typename Z>
TEMPLATE_NOINLINE int32_t
tf_geo_rasterize__cpu_template(TableFunctionManager& mgr,
                               const Column<T>& input_x,
                               const Column<T>& input_y,
                               const Column<Z>& input_z,
                               const T bin_dim_meters,
                               const bool geographic_coords,
                               const int64_t null_neighborhood_fill_radius,
                               const T x_min,
                               const T x_max,
                               const T y_min,
                               const T y_max,
                               Column<T>& output_x,
                               Column<T>& output_y,
                               Column<Z>& output_z) {
  if (bin_dim_meters <= 0) {
    return mgr.error_message(
        "tf_geo_rasterize: bin_dim_meters argument must be greater than 0");
  }

  if (null_neighborhood_fill_radius < 0) {
    return mgr.error_message(
        "tf_geo_rasterize: null_neighborhood_fill_radius argument must be greater than "
        "or equal to 0");
  }

  if (x_min >= x_max) {
    return mgr.error_message("tf_geo_rasterize: x_min must be less than x_max");
  }

  if (y_min >= y_max) {
    return mgr.error_message("tf_geo_rasterize: y_min must be less than y_max");
  }

  GeoRaster<T, Z> geo_raster(input_x,
                             input_y,
                             input_z,
                             bin_dim_meters,
                             geographic_coords,
                             true,
                             x_min,
                             x_max,
                             y_min,
                             y_max);
  return geo_raster.outputDenseColumns(
      mgr, output_x, output_y, output_z, null_neighborhood_fill_radius);
}

#include "GeoRaster.cpp"

#endif  // __CUDACC__
#endif  // HAVE_SYSTEM_TFS
