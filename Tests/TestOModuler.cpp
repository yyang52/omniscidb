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
#include "QueryEngine/catalogApi.h"

// #include "QueryEngine/OModuler.h"
#include "QueryEngine/CiderRelAlgDispatcher.h"
#include "QueryEngine/CiderWorkUnit.h"
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
  TableDescriptor table1;
  TableDescriptor table2;
  TableDescriptor table3;
  table1.tableName = "CATALOG";
  table2.tableName = "mapd";
  table3.tableName = "test";
  auto tbPtr1 = &table1;
  auto tbPtr2 = &table2;
  auto tbPtr3 = &table3;
  MetaDesc meta;
  meta.buildData("CATALOG",tbPtr1);
  meta.buildData("mapd",tbPtr2);
  //meta.buildData("test",tbPtr3);











  CiderRelAlgDispatcher patcher;

  auto res = CiderUnitModuler::createCiderUnitModuler(nullptr);

  // std::unique_ptr<QueryMemoryDescriptor> qmd_ptr = res.compile();

  //auto res = CiderWorkUnit::createCiderWorkUnit(nullptr);
  const char* json = "{\"rels\":[{\"id\":\"0\",\"relOp\":\"LogicalTableScan\",\"fieldNames\":[\"b\",\"dec\",\"d\",\"f\",\"m\",\"n\",\"o\",\"real_str\",\"str\",\"fx\",\"t\",\"x\",\"y\",\"z\"],\"table\":[\"CATALOG\",\"mapd\"],\"inputs\":[]},{\"id\":\"1\",\"relOp\":\"LogicalProject\",\"fields\":[\"y\"],\"exprs\":[{\"input\":12}]},{\"id\":\"2\",\"relOp\":\"LogicalAggregate\",\"group\":[0],\"aggs\":[{\"agg\":\"COUNT\",\"type\":{\"type\":\"BIGINT\",\"nullable\":false},\"distinct\":false,\"operands\":[]}]}]}";
  rapidjson::Document q;
  q.Parse(json);  
  std::cout<<"1"<<std::endl;
  const auto& rels_ = field(q, "rels"); 
  auto result= patcher.run(rels_,meta);

  std::cout<<"test,begin"<<std::endl;
  std::shared_ptr<RelProject> logical_project = std::dynamic_pointer_cast<RelProject>(result[1]);
  //RelAlgNode func test
  EXPECT_EQ(logical_project->getId(),1);
  EXPECT_EQ(logical_project->hasContextData(),0);
  EXPECT_EQ(logical_project->isNop(),0);
  EXPECT_EQ(logical_project->inputCount(),1);
  EXPECT_EQ(logical_project->getOutputMetainfo().size(),0);
  RelAlgNode* vanish = nullptr;
  EXPECT_EQ(logical_project->hasInput(vanish),1);

  //RelProject func test
  EXPECT_EQ(logical_project->isSimple(),0);
  EXPECT_EQ(logical_project->isIdentity(),0);
  EXPECT_EQ(logical_project->isRenaming(),0);
  EXPECT_EQ(logical_project->hasWindowFunctionExpr(),0);
  EXPECT_EQ(logical_project->hasDeliveredHint(),0);
  EXPECT_EQ(logical_project->size(),1);
  EXPECT_EQ(logical_project->toString(),"RelProject([&RexAbstractInput(12)], [\"y\"])");
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
