/*
 * Copyright 2017 MapD Technologies, Inc.
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

#include <gtest/gtest.h>

#include "TestHelpers.h"
#include "Logger/Logger.h"

// #include "QueryEngine/OModuler.h"
#include "QueryEngine/CiderRelAlgDispatcher.h"

TEST(TestOModuler, SimpleProject) {
  // Catalog_Namespace::Catalog cat = Catalog_Namespace::Catalog();
  // details::RelAlgDispatcher dp(cat);
  // OModuler oModuler(dp);
  // std::shared_ptr<RelAlgNode> raPtr = oModuler.ast_convert(nullptr);

  // auto workUnit =
  //     OMModules::kernel.createWorkUnit(node, context);  // will do code gen internally

  // while (hasMoreData()) {
  //   inputData = OMModules::Data.convert(bufferFromDS, context) outputData =
  //       workUnit.executeWithData(inputData, context)
  // }

  CiderRelAlgDispatcher patcher();
  

  
  EXPECT_EQ(1, 1);
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  return err;
}
