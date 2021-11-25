#include "TestHelpers.h"

#include "../ImportExport/Importer.h"
#include "../Parser/parser.h"
#include "../QueryEngine/ArrowResultSet.h"
#include "../QueryEngine/CgenState.h"
#include "../QueryEngine/Descriptors/RelAlgExecutionDescriptor.h"
#include "../QueryEngine/Execute.h"
#include "../QueryEngine/ExpressionRange.h"
#include "../QueryEngine/ResultSetReductionJIT.h"
#include "../QueryRunner/QueryRunner.h"
#include "../Shared/DateConverters.h"
#include "../Shared/StringTransform.h"
#include "../Shared/scope.h"
#include "../SqliteConnector/SqliteConnector.h"
#include "ClusterTester.h"
#include "DistributedLoader.h"

#include <gtest/gtest.h>
#include <boost/algorithm/string.hpp>
#include <boost/any.hpp>
#include <boost/program_options.hpp>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using QR = QueryRunner::QueryRunner;

std::shared_ptr<ResultSet> run_multiple_agg_CPU(const std::string& query_str) {
  return QR::get()->runSQL(query_str, ExecutorDeviceType::CPU, false, true);
}

std::shared_ptr<ResultSet> run_multiple_agg(const std::string& query_str,
                                            const ExecutorDeviceType device_type) {
  return QR::get()->runSQL(query_str, device_type, false, true);
}

inline void run_ddl_statement(const std::string& create_table_stmt) {
  QR::get()->runDDLStatement(create_table_stmt);
}

void create_and_populate_data() {
  std::string drop_old_test{"DROP TABLE IF EXISTS tpch_lineitem;"};
  run_ddl_statement(drop_old_test);
  std::string lineitem_cols =
      "l_orderkey BIGINT, l_partkey BIGINT,l_suppkey BIGINT,l_linenumber INT,"
      "l_quantity DOUBLE,l_extendedprice DOUBLE,l_discount DOUBLE,"
      "l_tax DOUBLE,l_returnflag VARCHAR,l_linestatus VARCHAR,"
      "l_shipdate DATE,l_commitdate DATE,l_receiptdate DATE,"
      "l_shipinstruct VARCHAR,l_shipmode VARCHAR,l_comment VARCHAR";

  std::string create_query = "CREATE TABLE tpch_lineitem(" + lineitem_cols + ");";
  run_ddl_statement(create_query);

  std::vector<std::string> lineitem_rows;
  // 20 rows
  lineitem_rows.push_back(
      "114000001,3500852,887,4,31.00,57433.08,0.08,0.05,'N','O','1995-08-26','1995-08-26'"
      ",'1995-08-26','TAKEBACKRETURN','TRUCK','inalaccounts.blit'");
  lineitem_rows.push_back(
      "114000006,1716871,66896,3,9.00,16990.11,0.04,0.05,'N','F','1995-08-26','1995-08-"
      "26','1995-08-26','TAKEBACKRETURN','TRUCK','tornis.furiouslydogged'");
  lineitem_rows.push_back(
      "114000035,2930870,30899,1,11.00,20908.03,0.03,0.08,'R','F','1995-08-26','1995-08-"
      "27','1995-08-26','NONE','RAIL','xpressaccountsnagentici'");
  lineitem_rows.push_back(
      "114000064,3510290,160308,6,43.00,55905.16,0.01,0.08,'N','O','1996-04-20','1996-04-"
      "08','1996-04-30','COLLECTCOD','RAIL','xfluffilyfuriouslyboldsent'");
  lineitem_rows.push_back(
      "114000068,1414246,114261,6,33.00,38285.61,0.00,0.02,'N','O','1996-12-09','1996-11-"
      "04','1996-12-29','COLLECTCOD','SHIP','regularfoxessleep'");
  lineitem_rows.push_back(
      "114000096,1538148,38163,3,22.00,26093.54,0.10,0.04,'A','F','1994-06-21','1994-08-"
      "28','1994-07-11','NONE','FOB','.regularpackagesh'");
  lineitem_rows.push_back(
      "114000101,2760490,160491,3,49.00,75967.64,0.05,0.00,'A','F','1994-11-25','1994-12-"
      "27','1994-12-19','DELIVERINPERSON','MAIL','lybravedepositsdet'");
  lineitem_rows.push_back(
      "114000129,1446883,196905,5,45.00,82341.45,0.07,0.04,'A','F','1992-11-23','1992-11-"
      "29','1992-12-11','TAKEBACKRETURN','FOB','theodolites'");
  lineitem_rows.push_back(
      "114000133,1299603,99604,2,12.00,19230.48,0.03,0.05,'A','F','1993-05-22','1993-05-"
      "27','1993-05-31','COLLECTCOD','FOB','packages--reg'");
  lineitem_rows.push_back(
      "114000161,2939697,139698,3,43.00,74671.65,0.02,0.03,'N','O','1998-04-01','1998-02-"
      "12','1998-04-04','NONE','AIR','insttheslylysp'");
  lineitem_rows.push_back(
      "114000165,1845403,195431,1,2.00,2696.62,0.04,0.02,'N','O','1997-08-09','1997-10-"
      "04','1997-08-19','DELIVERINPERSON','SHIP','fluffilydespitethequi'");
  lineitem_rows.push_back(
      "114000195,1832953,82963,4,7.00,13201.02,0.07,0.07,'N','O','1995-12-02','1995-12-"
      "14','1995-12-13','NONE','AIR','osstheslylyfi'");
  lineitem_rows.push_back(
      "114000225,469,150470,1,32.00,43822.72,0.00,0.00,'N','O','1997-03-26','1997-02-19',"
      "'1997-04-14','COLLECTCOD','SHIP','tructionsaccordingtotheunusualpac'");
  lineitem_rows.push_back(
      "114000230,1513026,113027,4,26.00,27012.70,0.01,0.08,'N','O','1997-04-04','1997-06-"
      "10','1997-04-24','DELIVERINPERSON','SHIP','xessleepslylyacros'");
  lineitem_rows.push_back(
      "114000258,982847,82856,2,8.00,15438.40,0.09,0.07,'N','O','1998-06-14','1998-07-11'"
      ",'1998-06-21','COLLECTCOD','MAIL','gularrequestsuseacrossthecareful'");
  lineitem_rows.push_back(
      "114000262,2947255,97298,1,50.00,65105.50,0.05,0.08,'N','O','1997-11-23','1997-12-"
      "17','1997-12-09','TAKEBACKRETURN','SHIP','asymptotessleepsly'");
  lineitem_rows.push_back(
      "114000293,3433882,33883,1,31.00,56287.01,0.10,0.02,'N','O','1998-10-02','1998-09-"
      "17','1998-10-12','COLLECTCOD','RAIL','ffy,ironicdeposit'");
  lineitem_rows.push_back(
      "114000325,1520409,70431,3,15.00,21439.95,0.06,0.03,'A','F','1993-08-14','1993-09-"
      "06','1993-08-28','DELIVERINPERSON','SHIP','altheodolites.quickly'");
  lineitem_rows.push_back(
      "114000357,469717,119720,3,5.00,8433.45,0.08,0.01,'A','F','1994-02-25','1994-02-25'"
      ",'1994-03-17','DELIVERINPERSON','REGAIR','icdepositsnagafterthefin'");
  lineitem_rows.push_back(
      "114000386,809601,109610,4,40.00,60422.40,0.04,0.02,'A','F','1993-05-06','1993-06-"
      "04','1993-05-18','TAKEBACKRETURN','MAIL','nicdepositskindleacrossthequickly'");

  std::string insert_prefix = "INSERT INTO tpch_lineitem VALUES( ";
  for (int i = 0; i < lineitem_rows.size(); i++) {
    std::string insert_query = insert_prefix + lineitem_rows[i] + ");";
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
  }
}

void create_and_populate_ORDERS() {
  std::string drop_old_test{"DROP TABLE IF EXISTS tpch_orders;"};
  run_ddl_statement(drop_old_test);
  std::string orders_cols =
      "o_orderkey BIGINT, o_custkey BIGINT, o_orderstatus VARCHAR, o_totalprice DOUBLE, "
      "o_orderdate DATE, o_orderpriority VARCHAR, o_clerk VARCHAR, o_shippriority INT, "
      "o_comment VARCHAR";
  std::string create_query = "CREATE TABLE tpch_orders(" + orders_cols + ");";
  run_ddl_statement(create_query);

  std::vector<std::string> orders_rows;

  orders_rows.push_back(
      "5,889696,'F',140220.31,'1994-07-30','5-LOW','Clerk#000018496',0,'quickly."
      "bolddepositssleepslyly.packagesuseslyly'");
  orders_rows.push_back(
      "34,1220002,'O',59254.50,'1998-07-21','3-MEDIUM','Clerk#000004456',0,'"
      "lyfinalpackages.fluffilyfinaldepositswakeblithelyideas.spe'");
  orders_rows.push_back(
      "39,1635250,'O',340193.08,'1996-09-20','3-MEDIUM','Clerk#000013172',0,'oleexpress,"
      "ironicrequests:ir'");
  orders_rows.push_back(
      "68,570935,'O',328503.78,'1998-04-18','3-MEDIUM','Clerk#000008788',0,'"
      "pintobeanssleepcarefully.blithelyironicdepositshagglefuriouslyacro'");
  orders_rows.push_back(
      "97,421199,'F',116100.34,'1993-01-29','3-MEDIUM','Clerk#000010937',0,'"
      "hangblithelyalongtheregularaccounts.furiouslyevenideasafterthe'");
  orders_rows.push_back(
      "102,14315,'O',162720.74,'1997-05-09','2-HIGH','Clerk#000011904',0,'"
      "slylyaccordingtotheasymptotes.carefullyfinalpackagesintegratefurious'");
  orders_rows.push_back(
      "131,1854973,'F',107846.43,'1994-06-08','3-MEDIUM','Clerk#000012497',0,'"
      "afterthefluffilyspecialfoxesintegrates'");
  orders_rows.push_back(
      "160,1649869,'O',147995.85,'1996-12-19','4-NOTSPECIFIED','Clerk#000006827',0,'"
      "thelyspecialsauterneswakeslylyoft'");
  orders_rows.push_back(
      "165,544723,'F',225315.28,'1993-01-30','4-NOTSPECIFIED','Clerk#000005837',0,'"
      "acrosstheblithelyregularaccounts.bold'");
  orders_rows.push_back(
      "194,1234469,'F',200948.75,'1992-04-05','3-MEDIUM','Clerk#000007025',0,'"
      "egularrequestshaggleslylyregular,regularpintobeans.asymptote'");
  orders_rows.push_back(
      "199,1059383,'O',123552.11,'1996-03-07','2-HIGH','Clerk#000009776',0,'gtheodolites."
      "specialpackag'");
  orders_rows.push_back(
      "228,881552,'F',5099.38,'1993-02-25','1-URGENT','Clerk#000011232',0,'"
      "eswasslylyamongtheregularfoxes.blithelyregulardependenci'");
  orders_rows.push_back(
      "257,2453849,'O',10753.90,'1998-03-28','3-MEDIUM','Clerk#000013583',0,'"
      "tsagainsttheslywarhorsescajoleslylyaccounts'");
  orders_rows.push_back(
      "262,607030,'O',179962.03,'1995-11-25','4-NOTSPECIFIED','Clerk#000011007',0,'"
      "lpackages.blithelyfinalpintobeansusecarefu'");
  orders_rows.push_back(
      "291,2820994,'F',88721.73,'1994-03-13','1-URGENT','Clerk#000018450',0,'dolites."
      "carefullyregularpintobeanscajol'");
  orders_rows.push_back(
      "320,6037,'O',50406.31,'1997-11-21','2-HIGH','Clerk#000011441',0,'"
      "arfoxesnagblithely'");
  orders_rows.push_back(
      "325,800473,'F',132957.49,'1993-10-17','5-LOW','Clerk#000016869',0,'"
      "lysometimespendingpa'");
  orders_rows.push_back(
      "354,2765360,'O',237529.38,'1996-03-14','2-HIGH','Clerk#000010203',0,'"
      "lyregularideaswakeacrosstheslylysilentideas.finaldepositseatb'");
  orders_rows.push_back(
      "359,1551988,'F',206553.02,'1994-12-19','3-MEDIUM','Clerk#000018675',0,'ndolphins."
      "specialcourtsabovethecarefullyironicrequestsuse'");
  orders_rows.push_back(
      "388,893351,'F',190558.10,'1992-12-16','4-NOTSPECIFIED','Clerk#000007115',0,'"
      "arfoxesabovethefuriouslyironicdepositsnagslylyfinalreque'");

  std::string insert_prefix = "INSERT INTO tpch_orders VALUES( ";
  for (int i = 0; i < orders_rows.size(); i++) {
    std::string insert_query = insert_prefix + orders_rows[i] + ");";
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
  }
}

void create_and_populate_PART() {
  std::string drop_old_test{"DROP TABLE IF EXISTS tpch_part;"};
  run_ddl_statement(drop_old_test);
  std::string part_cols =
      "p_partkey BIGINT, p_name VARCHAR, p_mfgr VARCHAR, p_brand VARCHAR, p_type "
      "VARCHAR, "
      "p_size INT, p_container VARCHAR, p_retailprice DOUBLE, p_comment VARCHAR";

  std::string create_query = "CREATE TABLE tpch_part(" + part_cols + ");";
  run_ddl_statement(create_query);

  std::vector<std::string> part_rows;

  part_rows.push_back(
      "1 ,'goldenrod lavender spring chocolate lace','Manufacturer#1','Brand#13','PROMO "
      "BURNISHED COPPER ',7 ,'JUMBO PKG ',901.00 ,'ly. slyly ironi '");
  part_rows.push_back(
      "2 ,'blush thistle blue yellow saddle ','Manufacturer#1','Brand#13','LARGE BRUSHED "
      "BRASS ',1 ,'LG CASE ',902.00 ,'lar accounts amo '");
  part_rows.push_back(
      "3 ,'spring green yellow purple cornsilk ','Manufacturer#4','Brand#42','STANDARD "
      "POLISHED BRASS ',21 ,'WRAP CASE ',903.00 ,'egular deposits hag '");
  part_rows.push_back(
      "4 ,'cornflower chocolate smoke green pink ','Manufacturer#3','Brand#34','SMALL "
      "PLATED BRASS ',14 ,'MED DRUM ',904.00 ,'p furiously r '");
  part_rows.push_back(
      "5 ,'forest brown coral puff cream ','Manufacturer#3','Brand#32','STANDARD "
      "POLISHED TIN ',15 ,'SM PKG ',905.00 ,' wake carefully '");
  part_rows.push_back(
      "6 ,'bisque cornflower lawn forest magenta ','Manufacturer#2','Brand#24','PROMO "
      "PLATED STEEL ',4 ,'MED BAG ',906.00 ,'sual a '");
  part_rows.push_back(
      "7 ,'moccasin green thistle khaki floral ','Manufacturer#1','Brand#11','SMALL "
      "PLATED COPPER ',45 ,'SM BAG ',907.00 ,'lyly. ex '");
  part_rows.push_back(
      "8 ,'misty lace thistle snow royal ','Manufacturer#4','Brand#44','PROMO BURNISHED "
      "TIN ',41 ,'LG DRUM ',908.00 ,'eposi '");
  part_rows.push_back(
      "9 ,'thistle dim navajo dark gainsboro ','Manufacturer#4','Brand#43','SMALL "
      "BURNISHED STEEL ',12 ,'WRAP CASE ',909.00 ,'ironic foxe '");
  part_rows.push_back(
      "10 ,'linen pink saddle puff powder ','Manufacturer#5','Brand#54','LARGE BURNISHED "
      "STEEL ',44 ,'LG CAN ',910.01 ,'ithely final deposit '");
  part_rows.push_back(
      "11 ,'spring maroon seashell almond orchid ','Manufacturer#2','Brand#25','STANDARD "
      "BURNISHED NICKEL',43 ,'WRAP BOX ',911.01 ,'ng gr '");
  part_rows.push_back(
      "12 ,'cornflower wheat orange maroon ghost ','Manufacturer#3','Brand#33','MEDIUM "
      "ANODIZED STEEL ',25 ,'JUMBO CASE ',912.01 ,' quickly '");
  part_rows.push_back(
      "13 ,'ghost olive orange rosy thistle ','Manufacturer#5','Brand#55','MEDIUM "
      "BURNISHED NICKEL ',1 ,'JUMBO PACK ',913.01 ,'osits. '");
  part_rows.push_back(
      "14 ,'khaki seashell rose cornsilk navajo ','Manufacturer#1','Brand#13','SMALL "
      "POLISHED STEEL ',28 ,'JUMBO BOX ',914.01 ,'kages c '");
  part_rows.push_back(
      "15 ,'blanched honeydew sky turquoise medium ','Manufacturer#1','Brand#15','LARGE "
      "ANODIZED BRASS ',45 ,'LG CASE ',915.01 ,'usual ac '");
  part_rows.push_back(
      "16 ,'deep sky turquoise drab peach ','Manufacturer#3','Brand#32','PROMO PLATED "
      "TIN ',2 ,'MED PACK ',916.01 ,'unts a '");
  part_rows.push_back(
      "17 ,'indian navy coral pink deep ','Manufacturer#4','Brand#43','ECONOMY BRUSHED "
      "STEEL ',16 ,'LG BOX ',917.01 ,' regular accounts '");
  part_rows.push_back(
      "18 ,'turquoise indian lemon lavender misty ','Manufacturer#1','Brand#11','SMALL "
      "BURNISHED STEEL ',42 ,'JUMBO PACK ',918.01 ,'s cajole slyly a '");
  part_rows.push_back(
      "19 ,'chocolate navy tan deep brown ','Manufacturer#2','Brand#23','SMALL ANODIZED "
      "NICKEL ',33 ,'WRAP BOX ',919.01 ,' pending acc '");
  part_rows.push_back(
      "20 ,'ivory navy honeydew sandy midnight ','Manufacturer#1','Brand#12','LARGE "
      "POLISHED NICKEL ',48 ,'MED BAG ',920.02 ,'are across the asympt'");

  std::string insert_prefix = "INSERT INTO tpch_part VALUES( ";
  for (int i = 0; i < part_rows.size(); i++) {
    std::string insert_query = insert_prefix + part_rows[i] + ");";
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
  }
}

// TEST(TPCH, select_lineitem) {
//   std::string query = "select * from tpch_lineitem;";
//   auto res = run_multiple_agg_CPU(query);
//   std::cout << "result have " << res->rowCount() << " rows." << std::endl;
// }

// TEST(TPCH, select_orders) {
//   std::string query = "select * from tpch_orders;";
//   auto res = run_multiple_agg_CPU(query);
//   std::cout << "result have " << res->rowCount() << " rows." << std::endl;
// }

// TEST(TPCH, select_part) {
//   std::string query = "select * from tpch_part;";
//   auto res = run_multiple_agg_CPU(query);
//   std::cout << "result have " << res->rowCount() << " rows." << std::endl;
// }

// TEST(TPCH, Q6) {
//   std::string query =
//       "select sum(l_extendedprice * l_discount) as revenue from tpch_lineitem where "
//       "l_shipdate >= date '1994-01-01' and l_shipdate < date '1994-01-01' + interval '1' "
//       "year and l_discount between .06 - 0.01 and .06 + 0.01 and l_quantity <24;";
//   auto res = run_multiple_agg_CPU(query);
//   std::cout << "result have " << res->rowCount() << " rows." << std::endl;
// }

 TEST(TPCH, Q6M) {
   std::string query =
       "select l_orderkey  from tpch_lineitem where "
       "l_shipdate >= date '1994-01-01' ;";
   auto res = run_multiple_agg_CPU(query);
//   std::cout << "result have " << res->rowCount() << " rows." << std::endl;
 }

// TEST(TPCH, Q1) {
//   std::string query =
//       "select l_returnflag, l_linestatus, sum(l_quantity) as sum_qty, "
//       "sum(l_extendedprice) as sum_base_price, sum(l_extendedprice * (1 - l_discount))
//       " "as sum_disc_price, sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as "
//       "sum_charge, avg(l_quantity) as avg_qty, avg(l_extendedprice) as avg_price, "
//       "avg(l_discount) as avg_disc, count(*) as count_order from  tpch_lineitem where "
//       "l_shipdate <= date '1998-12-01' - interval '90' day group by l_returnflag, "
//       "l_linestatus order by l_returnflag, l_linestatus; ";
//   auto res = run_multiple_agg_CPU(query);
//   std::cout << "result have " << res->rowCount() << " rows." << std::endl;
// }

//TEST(TPCH, Q12) {
//  std::string query =
//      "select l_shipmode, sum(case when o_orderpriority = '1-URGENT' or o_orderpriority "
//      "= '2-HIGH' then 1 else 0 end) as high_line_count, sum(case when o_orderpriority "
//      "<> '1-URGENT' and o_orderpriority <> '2-HIGH' then 1 else 0 end) as "
//      "low_line_count from 	tpch_orders, tpch_lineitem where o_orderkey = l_orderkey "
//      "and "
//      "l_shipmode in ('MAIL', 'SHIP') and l_commitdate < l_receiptdate and l_shipdate < "
//      "l_commitdate and l_receiptdate >= date '1994-01-01' and l_receiptdate < date "
//      "'1994-01-01' + interval '1' year group by l_shipmode order by l_shipmode;";
//  auto res = run_multiple_agg_CPU(query);
//  std::cout << "result have " << res->rowCount() << " rows." << std::endl;
//}

// TEST(TPCH, Q14) {
//   std::string query =
//       "select 100.00 * sum(case when p_type like 'PROMO%' then l_extendedprice * (1 - "
//       "l_discount) else 0 end) / sum(l_extendedprice * (1 - l_discount)) as "
//       "promo_revenue from tpch_lineitem, tpch_part where l_partkey = p_partkey and l_shipdate >= "
//       "date '1995-09-01' and l_shipdate < date '1995-09-01' + interval '1' month ;";
//   auto res = run_multiple_agg_CPU(query);
//   std::cout << "result have " << res->rowCount() << " rows." << std::endl;
// }

// TEST(TPCH, Q_window) {
//   std::string query = "select l_returnflag, AVG(l_quantity) OVER (PARTITION BY l_returnflag) as avg_qty from tpch_lineitem;";
//   auto res = run_multiple_agg_CPU(query);
//   std::cout << "result have " << res->rowCount() << " rows." << std::endl;
//
// }

 TEST(TPCH, Q_limit) {
   std::string query = "select * from tpch_lineitem limit 11;";
   auto res = run_multiple_agg_CPU(query);
   std::cout << "result have " << res->rowCount() << " rows." << std::endl;
 }

int main(int argc, char** argv) {
  std::cout << "Starting CodeGenIRTest" << std::endl;
  int err;

  testing::InitGoogleTest(&argc, argv);
  namespace po = boost::program_options;

  po::options_description desc("Options");

  // these two are here to allow passing correctly google testing parameters
  desc.add_options()("gtest_list_tests", "list all test");
  desc.add_options()("gtest_filter", "filters tests, use --help for details");

  desc.add_options()("disable-literal-hoisting", "Disable literal hoisting");
  desc.add_options()("with-sharding", "Create sharded tables");
  desc.add_options()("from-table-reordering",
                     po::value<bool>(&g_from_table_reordering)
                         ->default_value(g_from_table_reordering)
                         ->implicit_value(true),
                     "Enable automatic table reordering in FROM clause");
  desc.add_options()("bigint-count",
                     po::value<bool>(&g_bigint_count)
                         ->default_value(g_bigint_count)
                         ->implicit_value(false),
                     "Use 64-bit count");
  desc.add_options()("disable-shared-mem-group-by",
                     po::value<bool>(&g_enable_smem_group_by)
                         ->default_value(g_enable_smem_group_by)
                         ->implicit_value(false),
                     "Enable/disable using GPU shared memory for GROUP BY.");
  desc.add_options()("enable-columnar-output",
                     po::value<bool>(&g_enable_columnar_output)
                         ->default_value(g_enable_columnar_output)
                         ->implicit_value(true),
                     "Enable/disable using columnar output format.");
  desc.add_options()("enable-bump-allocator",
                     po::value<bool>(&g_enable_bump_allocator)
                         ->default_value(g_enable_bump_allocator)
                         ->implicit_value(true),
                     "Enable the bump allocator for projection queries on GPU.");
  desc.add_options()("keep-data", "Don't drop tables at the end of the tests");
  desc.add_options()("use-existing-data",
                     "Don't create and drop tables and only run select tests (it "
                     "implies --keep-data).");
  desc.add_options()("dump-ir",
                     po::value<bool>()->default_value(false)->implicit_value(true),
                     "Dump IR and PTX for all executed queries to file."
                     " Currently only supports single node tests.");
  desc.add_options()("use-temporary-tables",
                     "Use temporary tables instead of physical storage.");
//  desc.add_options()("use-tbb",
//                     po::value<bool>(&g_use_tbb_pool)
//                         ->default_value(g_use_tbb_pool)
//                         ->implicit_value(true),
//                     "Use TBB thread pool implementation for query dispatch.");
  desc.add_options()("use-disk-cache",
                     "Use the disk cache for all tables with minimum size settings.");

  desc.add_options()(
      "test-help",
      "Print all ExecuteTest specific options (for gtest options use `--help`).");

  logger::LogOptions log_options(argv[0]);
  log_options.severity_ = logger::Severity::DEBUG1;
  log_options.severity_clog_ = logger::Severity::DEBUG1;
  log_options.set_options();  // update default values
  // desc.add(log_options.get_options());

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("test-help")) {
    std::cout << "Usage: ExecuteTest" << std::endl << std::endl;
    std::cout << desc << std::endl;
    return 0;
  }

  if (vm["dump-ir"].as<bool>()) {
    // Only log IR, PTX channels to file with no rotation size.
    log_options.channels_ = {logger::Channel::IR, logger::Channel::PTX};
    log_options.rotation_size_ = std::numeric_limits<size_t>::max();
  }

  logger::init(log_options);

  File_Namespace::DiskCacheConfig disk_cache_config{};
  if (vm.count("use-disk-cache")) {
    disk_cache_config = File_Namespace::DiskCacheConfig{
        File_Namespace::DiskCacheConfig::getDefaultPath(std::string(BASE_PATH)),
        File_Namespace::DiskCacheLevel::all};
  }

  QR::init(&disk_cache_config, BASE_PATH);

  create_and_populate_data();
  create_and_populate_ORDERS();
  create_and_populate_PART();

  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  Executor::nukeCacheOfExecutors();

  ResultSetReductionJIT::clearCache();
  QR::reset();
  return err;
}
