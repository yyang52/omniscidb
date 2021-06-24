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
  table1.tableName = "omnisci";
  table2.tableName = "flights_2008_10k";
  table3.tableName = "test";
  auto tbPtr1 = &table1;
  auto tbPtr2 = &table2;
  auto tbPtr3 = &table3;
  MetaDesc meta;
  meta.buildData("omnisci",tbPtr1);
  meta.buildData("flights_2008_10k",tbPtr2);
  //meta.buildData("test",tbPtr3);











  CiderRelAlgDispatcher patcher;

//  auto res = CiderUnitModuler::createCiderUnitModuler(nullptr);
  // std::unique_ptr<QueryMemoryDescriptor> qmd_ptr = res.compile();
//  auto res = CiderWorkUnit::createCiderWorkUnit(nullptr);


  //auto res = CiderWorkUnit::createCiderWorkUnit(nullptr);
  const char* json = "{\"rels\":[{\"id\":\"0\",\"relOp\":\"LogicalTableScan\",\"fieldNames\":[\"flight_year\",\"flight_month\",\"flight_dayofmonth\",\"flight_dayofweek\",\"deptime\",\"crsdeptime\",\"arrtime\",\"crsarrtime\",\"uniquecarrier\",\"flightnum\",\"tailnum\",\"actualelapsedtime\",\"crselapsedtime\",\"airtime\",\"arrdelay\",\"depdelay\",\"origin\",\"dest\",\"distance\",\"taxiin\",\"taxiout\",\"cancelled\",\"cancellationcode\",\"diverted\",\"carrierdelay\",\"weatherdelay\",\"nasdelay\",\"securitydelay\",\"lateaircraftdelay\",\"dep_timestamp\",\"arr_timestamp\",\"carrier_name\",\"plane_type\",\"plane_manufacturer\",\"plane_issue_date\",\"plane_model\",\"plane_status\",\"plane_aircraft_type\",\"plane_engine_type\",\"plane_year\",\"origin_name\",\"origin_city\",\"origin_state\",\"origin_country\",\"origin_lat\",\"origin_lon\",\"dest_name\",\"dest_city\",\"dest_state\",\"dest_country\",\"dest_lat\",\"dest_lon\",\"origin_merc_x\",\"origin_merc_y\",\"dest_merc_x\",\"dest_merc_y\",\"rowid\"],\"table\":[\"omnisci\",\"flights_2008_10k\"],\"inputs\":[]},{\"id\":\"1\",\"relOp\":\"LogicalFilter\",\"condition\":{\"op\":\"<\",\"operands\":[{\"input\":18},{\"literal\":175,\"type\":\"DECIMAL\",\"target_type\":\"INTEGER\",\"scale\":0,\"precision\":3,\"type_scale\":0,\"type_precision\":10}],\"type\":{\"type\":\"BOOLEAN\",\"nullable\":true}}},{\"id\":\"2\",\"relOp\":\"LogicalProject\",\"fields\":[\"Origin\",\"Destination\",\"airtime\"],\"exprs\":[{\"input\":41},{\"input\":47},{\"input\":13}]},{\"id\":\"3\",\"relOp\":\"LogicalAggregate\",\"fields\":[\"Origin\",\"Destination\",\"AverageAirtime\"],\"group\":[0,1],\"aggs\":[{\"agg\":\"AVG\",\"type\":{\"type\":\"DOUBLE\",\"nullable\":false},\"distinct\":false,\"operands\":[2]}]}]}";
  rapidjson::Document q;
  q.Parse(json);  
  const auto& rels_ = field(q, "rels"); 
  auto result= patcher.run(rels_,meta);
  RelAlgNode* vanish = nullptr;

  std::cout<<"test,begin"<<std::endl;

  std::shared_ptr<RelScan> logical_scan = std::dynamic_pointer_cast<RelScan>(result[0]);
  std::cout<<"_____________________RelScan________________________"<<std::endl;  
  std::cout<<logical_scan->getId()<<std::endl;
  std::cout<<logical_scan->hasContextData()<<std::endl;
  std::cout<<logical_scan->isNop()<<std::endl;
  std::cout<<logical_scan->inputCount()<<std::endl;
  std::cout<<logical_scan->getOutputMetainfo().size()<<std::endl;
  std::cout<<logical_scan->hasInput(vanish)<<std::endl;

  std::cout<<logical_scan->size()<<std::endl;
  std::cout<<logical_scan->toString()<<std::endl;
  EXPECT_EQ(logical_scan->getId(),1);
  EXPECT_EQ(logical_scan->hasContextData(),0);
  EXPECT_EQ(logical_scan->isNop(),0);
  EXPECT_EQ(logical_scan->inputCount(),0);
  EXPECT_EQ(logical_scan->getOutputMetainfo().size(),0);
  EXPECT_EQ(logical_scan->hasInput(vanish),0);
  EXPECT_EQ(logical_scan->size(),57);
  EXPECT_EQ(logical_scan->toString(),"RelScan(flights_2008_10k, [\"flight_year\", \"flight_month\", \"flight_dayofmonth\", \"flight_dayofweek\", \"deptime\", \"crsdeptime\", \"arrtime\", \"crsarrtime\", \"uniquecarrier\", \"flightnum\", \"tailnum\", \"actualelapsedtime\", \"crselapsedtime\", \"airtime\", \"arrdelay\", \"depdelay\", \"origin\", \"dest\", \"distance\", \"taxiin\", \"taxiout\", \"cancelled\", \"cancellationcode\", \"diverted\", \"carrierdelay\", \"weatherdelay\", \"nasdelay\", \"securitydelay\", \"lateaircraftdelay\", \"dep_timestamp\", \"arr_timestamp\", \"carrier_name\", \"plane_type\", \"plane_manufacturer\", \"plane_issue_date\", \"plane_model\", \"plane_status\", \"plane_aircraft_type\", \"plane_engine_type\", \"plane_year\", \"origin_name\", \"origin_city\", \"origin_state\", \"origin_country\", \"origin_lat\", \"origin_lon\", \"dest_name\", \"dest_city\", \"dest_state\", \"dest_country\", \"dest_lat\", \"dest_lon\", \"origin_merc_x\", \"origin_merc_y\", \"dest_merc_x\", \"dest_merc_y\", \"rowid\"])");   
  // std::cout<<logical_scan->hasDeliveredHint()<<std::endl;
  
  //RelFilter
  std::shared_ptr<RelFilter> logical_RelFilter = std::dynamic_pointer_cast<RelFilter>(result[1]);
  std::cout<<"_____________________RelFilter________________________"<<std::endl;  
  std::cout<<logical_RelFilter->getId()<<std::endl;
  std::cout<<logical_RelFilter->hasContextData()<<std::endl;
  std::cout<<logical_RelFilter->isNop()<<std::endl;
  std::cout<<logical_RelFilter->inputCount()<<std::endl;
  std::cout<<logical_RelFilter->getOutputMetainfo().size()<<std::endl;
  std::cout<<logical_RelFilter->hasInput(vanish)<<std::endl;

  std::cout<<logical_RelFilter->size()<<std::endl;
  std::cout<<logical_RelFilter->toString()<<std::endl;



  //relproject
  std::shared_ptr<RelProject> logical_project = std::dynamic_pointer_cast<RelProject>(result[2]);
  std::cout<<"_____________________RelProject________________________"<<std::endl;

  std::cout<<logical_project->getId()<<std::endl;
  std::cout<<logical_project->hasContextData()<<std::endl;
  std::cout<<logical_project->isNop()<<std::endl;
  std::cout<<logical_project->inputCount()<<std::endl;
  std::cout<<logical_project->getOutputMetainfo().size()<<std::endl;
  std::cout<<logical_project->hasInput(vanish)<<std::endl;
  std::cout<<logical_project->isSimple()<<std::endl;
  std::cout<<logical_project->isIdentity()<<std::endl;
  std::cout<<logical_project->isRenaming()<<std::endl;
  std::cout<<logical_project->hasWindowFunctionExpr()<<std::endl;
  std::cout<<logical_project->hasDeliveredHint()<<std::endl;
  std::cout<<logical_project->size()<<std::endl;
  std::cout<<logical_project->toString()<<std::endl;
  //RelAlgNode func test
  EXPECT_EQ(logical_project->getId(),3);
  EXPECT_EQ(logical_project->hasContextData(),0);
  EXPECT_EQ(logical_project->isNop(),0);
  EXPECT_EQ(logical_project->inputCount(),1);
  EXPECT_EQ(logical_project->getOutputMetainfo().size(),0);
  //RelAlgNode* vanish = nullptr;
  EXPECT_EQ(logical_project->hasInput(vanish),0);

  //RelProject func test
  EXPECT_EQ(logical_project->isSimple(),0);
  EXPECT_EQ(logical_project->isIdentity(),0);
  EXPECT_EQ(logical_project->isRenaming(),0);
  EXPECT_EQ(logical_project->hasWindowFunctionExpr(),0);
  EXPECT_EQ(logical_project->hasDeliveredHint(),0);
  EXPECT_EQ(logical_project->size(),3);
  EXPECT_EQ(logical_project->toString(),"RelProject([&RexAbstractInput(41), &RexAbstractInput(47), &RexAbstractInput(13)], [\"Origin\", \"Destination\", \"airtime\"])");
  
  //RelAggregate
  std::shared_ptr<RelAggregate> logical_RelAggregate = std::dynamic_pointer_cast<RelAggregate>(result[3]);
  std::cout<<"_____________________RelAggregate________________________"<<std::endl;
  std::cout<<logical_RelAggregate->getId()<<std::endl;
  std::cout<<logical_RelAggregate->hasContextData()<<std::endl;
  std::cout<<logical_RelAggregate->isNop()<<std::endl;
  std::cout<<logical_RelAggregate->inputCount()<<std::endl;
  std::cout<<logical_RelAggregate->getOutputMetainfo().size()<<std::endl;
  std::cout<<logical_RelAggregate->hasInput(vanish)<<std::endl;

  std::cout<<logical_RelAggregate->size()<<std::endl;
  std::cout<<logical_RelAggregate->toString()<<std::endl;


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
