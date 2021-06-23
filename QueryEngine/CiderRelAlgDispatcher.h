/*
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

#include <rapidjson/error/en.h>
#include <rapidjson/error/error.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include "CalciteDeserializerUtils.h"
#include "JsonAccessors.h"
#include "RelAlgDagBuilder.h"
#include "catalogApi.h"

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

class CiderRelAlgDispatcher {
 public:
  CiderRelAlgDispatcher() {};
  // todo: we need data type.
  std::vector<std::shared_ptr<RelAlgNode>> run(const rapidjson::Value& rels, MetaDesc meta);

 private:
  // TODO: placeholder replace Scan
  std::shared_ptr<RelScan> dispatchTableScan(const rapidjson::Value& scan_ra, MetaDesc meta);

  std::shared_ptr<RelProject> dispatchProject(const rapidjson::Value& proj_ra);

  std::shared_ptr<RelFilter> dispatchFilter(const rapidjson::Value& filter_ra);
  std::shared_ptr<RelJoin> dispatchJoin(const rapidjson::Value& join_ra);
  std::shared_ptr<RelAggregate> dispatchAggregate(const rapidjson::Value& agg_ra);
  std::shared_ptr<RelSort> dispatchSort(const rapidjson::Value& sort_ra);
  std::shared_ptr<RelLogicalValues> dispatchLogicalValues(
      const rapidjson::Value& logical_values_ra);
  std::shared_ptr<RelLogicalUnion> dispatchUnion(
      const rapidjson::Value& logical_union_ra);
  RelAlgInputs getRelAlgInputs(const rapidjson::Value& node);
  std::pair<std::string, std::string> getKVOptionPair(std::string& str, size_t& pos);
  ExplainedQueryHint parseHintString(std::string& hint_string);
  void getRelAlgHints(const rapidjson::Value& json_node,
                      std::shared_ptr<RelAlgNode> node);
  std::shared_ptr<const RelAlgNode> prev(const rapidjson::Value& crt_node);

  std::vector<std::shared_ptr<RelAlgNode>> nodes_;
};
void check_empty_inputs_field(const rapidjson::Value& node);
const TableDescriptor* getTableFromScanNode(const rapidjson::Value& scan_ra, const MetaDesc meta);
std::vector<std::string> getFieldNamesFromScanNode(const rapidjson::Value& scan_ra);
