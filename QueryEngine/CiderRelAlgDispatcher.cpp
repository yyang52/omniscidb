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
#include "CiderRelAlgDispatcher.h"

unsigned node_id(const rapidjson::Value& ra_node) noexcept {
  const auto& id = field(ra_node, "id");
  return std::stoi(json_str(id));
}

std::string json_node_to_string(const rapidjson::Value& node) noexcept {
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  node.Accept(writer);
  return buffer.GetString();
}

// The parse_* functions below de-serialize expressions as they come from Calcite.
// RelAlgDagBuilder will take care of making the representation easy to
// navigate for lower layers, for example by replacing RexAbstractInput with RexInput.

std::unique_ptr<RexAbstractInput> parse_abstract_input(
    const rapidjson::Value& expr) noexcept {
  const auto& input = field(expr, "input");
  return std::unique_ptr<RexAbstractInput>(new RexAbstractInput(json_i64(input)));
}

std::unique_ptr<RexLiteral> parse_literal(const rapidjson::Value& expr) {
  CHECK(expr.IsObject());
  const auto& literal = field(expr, "literal");
  const auto type = to_sql_type(json_str(field(expr, "type")));
  const auto target_type = to_sql_type(json_str(field(expr, "target_type")));
  const auto scale = json_i64(field(expr, "scale"));
  const auto precision = json_i64(field(expr, "precision"));
  const auto type_scale = json_i64(field(expr, "type_scale"));
  const auto type_precision = json_i64(field(expr, "type_precision"));
  if (literal.IsNull()) {
    return std::unique_ptr<RexLiteral>(new RexLiteral(target_type));
  }
  switch (type) {
    case kINT:
    case kBIGINT:
    case kDECIMAL:
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH:
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
      return std::unique_ptr<RexLiteral>(new RexLiteral(json_i64(literal),
                                                        type,
                                                        target_type,
                                                        scale,
                                                        precision,
                                                        type_scale,
                                                        type_precision));
    case kDOUBLE: {
      if (literal.IsDouble()) {
        return std::unique_ptr<RexLiteral>(new RexLiteral(json_double(literal),
                                                          type,
                                                          target_type,
                                                          scale,
                                                          precision,
                                                          type_scale,
                                                          type_precision));
      } else if (literal.IsInt64()) {
        return std::make_unique<RexLiteral>(static_cast<double>(literal.GetInt64()),
                                            type,
                                            target_type,
                                            scale,
                                            precision,
                                            type_scale,
                                            type_precision);

      } else if (literal.IsUint64()) {
        return std::make_unique<RexLiteral>(static_cast<double>(literal.GetUint64()),
                                            type,
                                            target_type,
                                            scale,
                                            precision,
                                            type_scale,
                                            type_precision);
      }
      UNREACHABLE() << "Unhandled type: " << literal.GetType();
    }
    case kTEXT:
      return std::unique_ptr<RexLiteral>(new RexLiteral(json_str(literal),
                                                        type,
                                                        target_type,
                                                        scale,
                                                        precision,
                                                        type_scale,
                                                        type_precision));
    case kBOOLEAN:
      return std::unique_ptr<RexLiteral>(new RexLiteral(json_bool(literal),
                                                        type,
                                                        target_type,
                                                        scale,
                                                        precision,
                                                        type_scale,
                                                        type_precision));
    case kNULLT:
      return std::unique_ptr<RexLiteral>(new RexLiteral(target_type));
    default:
      CHECK(false);
  }
  CHECK(false);
  return nullptr;
}

std::unique_ptr<const RexScalar> parse_scalar_expr(const rapidjson::Value& expr);

SQLTypeInfo parse_type(const rapidjson::Value& type_obj) {
  if (type_obj.IsArray()) {
    throw QueryNotSupported("Composite types are not currently supported.");
  }
  CHECK(type_obj.IsObject() && type_obj.MemberCount() >= 2)
      << json_node_to_string(type_obj);
  const auto type = to_sql_type(json_str(field(type_obj, "type")));
  const auto nullable = json_bool(field(type_obj, "nullable"));
  const auto precision_it = type_obj.FindMember("precision");
  const int precision =
      precision_it != type_obj.MemberEnd() ? json_i64(precision_it->value) : 0;
  const auto scale_it = type_obj.FindMember("scale");
  const int scale = scale_it != type_obj.MemberEnd() ? json_i64(scale_it->value) : 0;
  SQLTypeInfo ti(type, !nullable);
  ti.set_precision(precision);
  ti.set_scale(scale);
  return ti;
}

std::vector<std::unique_ptr<const RexScalar>> parse_expr_array(
    const rapidjson::Value& arr) {
  std::vector<std::unique_ptr<const RexScalar>> exprs;
  for (auto it = arr.Begin(); it != arr.End(); ++it) {
    exprs.emplace_back(parse_scalar_expr(*it));
  }
  return exprs;
}

SqlWindowFunctionKind parse_window_function_kind(const std::string& name) {
  if (name == "ROW_NUMBER") {
    return SqlWindowFunctionKind::ROW_NUMBER;
  }
  if (name == "RANK") {
    return SqlWindowFunctionKind::RANK;
  }
  if (name == "DENSE_RANK") {
    return SqlWindowFunctionKind::DENSE_RANK;
  }
  if (name == "PERCENT_RANK") {
    return SqlWindowFunctionKind::PERCENT_RANK;
  }
  if (name == "CUME_DIST") {
    return SqlWindowFunctionKind::CUME_DIST;
  }
  if (name == "NTILE") {
    return SqlWindowFunctionKind::NTILE;
  }
  if (name == "LAG") {
    return SqlWindowFunctionKind::LAG;
  }
  if (name == "LEAD") {
    return SqlWindowFunctionKind::LEAD;
  }
  if (name == "FIRST_VALUE") {
    return SqlWindowFunctionKind::FIRST_VALUE;
  }
  if (name == "LAST_VALUE") {
    return SqlWindowFunctionKind::LAST_VALUE;
  }
  if (name == "AVG") {
    return SqlWindowFunctionKind::AVG;
  }
  if (name == "MIN") {
    return SqlWindowFunctionKind::MIN;
  }
  if (name == "MAX") {
    return SqlWindowFunctionKind::MAX;
  }
  if (name == "SUM") {
    return SqlWindowFunctionKind::SUM;
  }
  if (name == "COUNT") {
    return SqlWindowFunctionKind::COUNT;
  }
  if (name == "$SUM0") {
    return SqlWindowFunctionKind::SUM_INTERNAL;
  }
  throw std::runtime_error("Unsupported window function: " + name);
}

std::vector<std::unique_ptr<const RexScalar>> parse_window_order_exprs(
    const rapidjson::Value& arr) {
  std::vector<std::unique_ptr<const RexScalar>> exprs;
  for (auto it = arr.Begin(); it != arr.End(); ++it) {
    exprs.emplace_back(parse_scalar_expr(field(*it, "field")));
  }
  return exprs;
}

SortDirection parse_sort_direction(const rapidjson::Value& collation) {
  return json_str(field(collation, "direction")) == std::string("DESCENDING")
             ? SortDirection::Descending
             : SortDirection::Ascending;
}

NullSortedPosition parse_nulls_position(const rapidjson::Value& collation) {
  return json_str(field(collation, "nulls")) == std::string("FIRST")
             ? NullSortedPosition::First
             : NullSortedPosition::Last;
}

std::vector<SortField> parse_window_order_collation(const rapidjson::Value& arr) {
  std::vector<SortField> collation;
  size_t field_idx = 0;
  for (auto it = arr.Begin(); it != arr.End(); ++it, ++field_idx) {
    const auto sort_dir = parse_sort_direction(*it);
    const auto null_pos = parse_nulls_position(*it);
    collation.emplace_back(field_idx, sort_dir, null_pos);
  }
  return collation;
}

RexWindowFunctionOperator::RexWindowBound parse_window_bound(
    const rapidjson::Value& window_bound_obj) {
  CHECK(window_bound_obj.IsObject());
  RexWindowFunctionOperator::RexWindowBound window_bound;
  window_bound.unbounded = json_bool(field(window_bound_obj, "unbounded"));
  window_bound.preceding = json_bool(field(window_bound_obj, "preceding"));
  window_bound.following = json_bool(field(window_bound_obj, "following"));
  window_bound.is_current_row = json_bool(field(window_bound_obj, "is_current_row"));
  const auto& offset_field = field(window_bound_obj, "offset");
  if (offset_field.IsObject()) {
    window_bound.offset = parse_scalar_expr(offset_field);
  } else {
    CHECK(offset_field.IsNull());
  }
  window_bound.order_key = json_i64(field(window_bound_obj, "order_key"));
  return window_bound;
}

// std::unique_ptr<const RexSubQuery> parse_subquery(const rapidjson::Value& expr) {
//   const auto& operands = field(expr, "operands");
//   CHECK(operands.IsArray());
//   CHECK_GE(operands.Size(), unsigned(0));
//   const auto& subquery_ast = field(expr, "subquery");

//   RelAlgDagBuilder subquery_dag(root_dag_builder, subquery_ast, cat, nullptr);
//   auto subquery = std::make_shared<RexSubQuery>(subquery_dag.getRootNodeShPtr());
//   root_dag_builder.registerSubquery(subquery);
//   return subquery->deepCopy();
// }

std::unique_ptr<RexOperator> parse_operator(const rapidjson::Value& expr) {
  const auto op_name = json_str(field(expr, "op"));
  const bool is_quantifier =
      op_name == std::string("PG_ANY") || op_name == std::string("PG_ALL");
  const auto op = is_quantifier ? kFUNCTION : to_sql_op(op_name);
  const auto& operators_json_arr = field(expr, "operands");
  CHECK(operators_json_arr.IsArray());
  auto operands = parse_expr_array(operators_json_arr);
  const auto type_it = expr.FindMember("type");
  CHECK(type_it != expr.MemberEnd());
  auto ti = parse_type(type_it->value);
  // if (op == kIN && expr.HasMember("subquery")) {
  //   auto subquery = parse_subquery(expr);
  //   operands.emplace_back(std::move(subquery));
  // }
  if (expr.FindMember("partition_keys") != expr.MemberEnd()) {
    const auto& partition_keys_arr = field(expr, "partition_keys");
    auto partition_keys = parse_expr_array(partition_keys_arr);
    const auto& order_keys_arr = field(expr, "order_keys");
    auto order_keys = parse_window_order_exprs(order_keys_arr);
    const auto collation = parse_window_order_collation(order_keys_arr);
    const auto kind = parse_window_function_kind(op_name);
    const auto lower_bound = parse_window_bound(field(expr, "lower_bound"));
    const auto upper_bound = parse_window_bound(field(expr, "upper_bound"));
    bool is_rows = json_bool(field(expr, "is_rows"));
    ti.set_notnull(false);
    return std::make_unique<RexWindowFunctionOperator>(kind,
                                                       operands,
                                                       partition_keys,
                                                       order_keys,
                                                       collation,
                                                       lower_bound,
                                                       upper_bound,
                                                       is_rows,
                                                       ti);
  }
  return std::unique_ptr<RexOperator>(op == kFUNCTION
                                          ? new RexFunctionOperator(op_name, operands, ti)
                                          : new RexOperator(op, operands, ti));
}

std::unique_ptr<RexCase> parse_case(const rapidjson::Value& expr) {
  const auto& operands = field(expr, "operands");
  CHECK(operands.IsArray());
  CHECK_GE(operands.Size(), unsigned(2));
  std::unique_ptr<const RexScalar> else_expr;
  std::vector<
      std::pair<std::unique_ptr<const RexScalar>, std::unique_ptr<const RexScalar>>>
      expr_pair_list;
  for (auto operands_it = operands.Begin(); operands_it != operands.End();) {
    auto when_expr = parse_scalar_expr(*operands_it++);
    if (operands_it == operands.End()) {
      else_expr = std::move(when_expr);
      break;
    }
    auto then_expr = parse_scalar_expr(*operands_it++);
    expr_pair_list.emplace_back(std::move(when_expr), std::move(then_expr));
  }
  return std::unique_ptr<RexCase>(new RexCase(expr_pair_list, else_expr));
}

std::vector<std::string> strings_from_json_array(
    const rapidjson::Value& json_str_arr) noexcept {
  CHECK(json_str_arr.IsArray());
  std::vector<std::string> fields;
  for (auto json_str_arr_it = json_str_arr.Begin(); json_str_arr_it != json_str_arr.End();
       ++json_str_arr_it) {
    CHECK(json_str_arr_it->IsString());
    fields.emplace_back(json_str_arr_it->GetString());
  }
  return fields;
}

std::vector<size_t> indices_from_json_array(
    const rapidjson::Value& json_idx_arr) noexcept {
  CHECK(json_idx_arr.IsArray());
  std::vector<size_t> indices;
  for (auto json_idx_arr_it = json_idx_arr.Begin(); json_idx_arr_it != json_idx_arr.End();
       ++json_idx_arr_it) {
    CHECK(json_idx_arr_it->IsInt());
    CHECK_GE(json_idx_arr_it->GetInt(), 0);
    indices.emplace_back(json_idx_arr_it->GetInt());
  }
  return indices;
}

std::unique_ptr<const RexAgg> parse_aggregate_expr(const rapidjson::Value& expr) {
  const auto agg = to_agg_kind(json_str(field(expr, "agg")));
  const auto distinct = json_bool(field(expr, "distinct"));
  const auto agg_ti = parse_type(field(expr, "type"));
  const auto operands = indices_from_json_array(field(expr, "operands"));
  if (operands.size() > 1 && (operands.size() != 2 || agg != kAPPROX_COUNT_DISTINCT)) {
    throw QueryNotSupported("Multiple arguments for aggregates aren't supported");
  }
  return std::unique_ptr<const RexAgg>(new RexAgg(agg, distinct, agg_ti, operands));
}

std::unique_ptr<const RexScalar> parse_scalar_expr(const rapidjson::Value& expr) {
  CHECK(expr.IsObject());
  if (expr.IsObject() && expr.HasMember("input")) {
    return std::unique_ptr<const RexScalar>(parse_abstract_input(expr));
  }
  if (expr.IsObject() && expr.HasMember("literal")) {
    return std::unique_ptr<const RexScalar>(parse_literal(expr));
  }
  if (expr.IsObject() && expr.HasMember("op")) {
    const auto op_str = json_str(field(expr, "op"));
    if (op_str == std::string("CASE")) {
      return std::unique_ptr<const RexScalar>(parse_case(expr));
    }
    if (op_str == std::string("$SCALAR_QUERY")) {
      throw QueryNotSupported("Expression node " + json_node_to_string(expr) +
                              " not supported");  // todo
    }
    return std::unique_ptr<const RexScalar>(parse_operator(expr));
  }
  throw QueryNotSupported("Expression node " + json_node_to_string(expr) +
                          " not supported");
}

JoinType to_join_type(const std::string& join_type_name) {
  if (join_type_name == "inner") {
    return JoinType::INNER;
  }
  if (join_type_name == "left") {
    return JoinType::LEFT;
  }
  throw QueryNotSupported("Join type (" + join_type_name + ") not supported");
}

std::vector<std::shared_ptr<RelAlgNode>> CiderRelAlgDispatcher::run(
    const rapidjson::Value& rels, MetaDesc meta) {
  for (auto rels_it = rels.Begin(); rels_it != rels.End(); ++rels_it) {
    const auto& crt_node = *rels_it;
    const auto id = node_id(crt_node);
    CHECK_EQ(static_cast<size_t>(id), nodes_.size());
    CHECK(crt_node.IsObject());
    std::shared_ptr<RelAlgNode> ra_node = nullptr;
    const auto rel_op = json_str(field(crt_node, "relOp"));
    if (rel_op == std::string("LogicalProject")) {
      ra_node = dispatchProject(crt_node);
    } else if (rel_op == std::string("LogicalFilter")) {
      ra_node = dispatchFilter(crt_node);
    } else if (rel_op == std::string("LogicalAggregate")) {
      ra_node = dispatchAggregate(crt_node);
    } else if (rel_op == std::string("LogicalJoin")) {
      ra_node = dispatchJoin(crt_node);
    } else if (rel_op == std::string("LogicalSort")) {
      ra_node = dispatchSort(crt_node);
    } else if (rel_op == std::string("LogicalValues")) {
      ra_node = dispatchLogicalValues(crt_node);
    } else if (rel_op == std::string("LogicalUnion")) {
      ra_node = dispatchUnion(crt_node);
    } else if (rel_op == std::string("EnumerableTableScan") ||
        rel_op == std::string("LogicalTableScan")) {
        std::cout<<"LogicalTableScan,begin"<<std::endl;
        ra_node = dispatchTableScan(crt_node,meta);
    } else {
      throw QueryNotSupported(std::string("Node ") + rel_op + " not supported yet");
    }
    nodes_.push_back(ra_node);
  }

  return std::move(nodes_);
}

std::shared_ptr<RelScan> CiderRelAlgDispatcher::dispatchTableScan(const rapidjson::Value& scan_ra, MetaDesc meta) {
  check_empty_inputs_field(scan_ra);
  CHECK(scan_ra.IsObject());
  const auto td = getTableFromScanNode(scan_ra,meta);
  const auto field_names = getFieldNamesFromScanNode(scan_ra);
  if (scan_ra.HasMember("hints")) {
    auto scan_node = std::make_shared<RelScan>(td, field_names);
    getRelAlgHints(scan_ra, scan_node);
    return scan_node;
  }
  return std::make_shared<RelScan>(td, field_names);
}

std::shared_ptr<RelProject> CiderRelAlgDispatcher::dispatchProject(
    const rapidjson::Value& proj_ra) {
  const auto inputs = getRelAlgInputs(proj_ra);
  CHECK_EQ(size_t(1), inputs.size());
  const auto& exprs_json = field(proj_ra, "exprs");
  CHECK(exprs_json.IsArray());
  std::vector<std::unique_ptr<const RexScalar>> exprs;
  for (auto exprs_json_it = exprs_json.Begin(); exprs_json_it != exprs_json.End();
       ++exprs_json_it) {
    exprs.emplace_back(parse_scalar_expr(*exprs_json_it));
  }
  const auto& fields = field(proj_ra, "fields");
  if (proj_ra.HasMember("hints")) {
    auto project_node = std::make_shared<RelProject>(
        exprs, strings_from_json_array(fields), inputs.front());
    getRelAlgHints(proj_ra, project_node);
    return project_node;
  }
  return std::make_shared<RelProject>(
      exprs, strings_from_json_array(fields), inputs.front());
}

std::shared_ptr<RelFilter> CiderRelAlgDispatcher::dispatchFilter(
    const rapidjson::Value& filter_ra) {
  const auto inputs = getRelAlgInputs(filter_ra);
  CHECK_EQ(size_t(1), inputs.size());
  const auto id = node_id(filter_ra);
  CHECK(id);
  auto condition = parse_scalar_expr(field(filter_ra, "condition"));
  return std::make_shared<RelFilter>(condition, inputs.front());
}

std::shared_ptr<RelAggregate> CiderRelAlgDispatcher::dispatchAggregate(
    const rapidjson::Value& agg_ra) {
  const auto inputs = getRelAlgInputs(agg_ra);
  CHECK_EQ(size_t(1), inputs.size());
  const auto fields = strings_from_json_array(field(agg_ra, "fields"));
  const auto group = indices_from_json_array(field(agg_ra, "group"));
  for (size_t i = 0; i < group.size(); ++i) {
    CHECK_EQ(i, group[i]);
  }
  if (agg_ra.HasMember("groups") || agg_ra.HasMember("indicator")) {
    throw QueryNotSupported("GROUP BY extensions not supported");
  }
  const auto& aggs_json_arr = field(agg_ra, "aggs");
  CHECK(aggs_json_arr.IsArray());
  std::vector<std::unique_ptr<const RexAgg>> aggs;
  for (auto aggs_json_arr_it = aggs_json_arr.Begin();
       aggs_json_arr_it != aggs_json_arr.End();
       ++aggs_json_arr_it) {
    aggs.emplace_back(parse_aggregate_expr(*aggs_json_arr_it));
  }
  if (agg_ra.HasMember("hints")) {
    auto agg_node =
        std::make_shared<RelAggregate>(group.size(), aggs, fields, inputs.front());
    getRelAlgHints(agg_ra, agg_node);
    return agg_node;
  }
  return std::make_shared<RelAggregate>(group.size(), aggs, fields, inputs.front());
}

std::shared_ptr<RelJoin> CiderRelAlgDispatcher::dispatchJoin(
    const rapidjson::Value& join_ra) {
  const auto inputs = getRelAlgInputs(join_ra);
  CHECK_EQ(size_t(2), inputs.size());
  const auto join_type = to_join_type(json_str(field(join_ra, "joinType")));
  auto filter_rex = parse_scalar_expr(field(join_ra, "condition"));
  if (join_ra.HasMember("hints")) {
    auto join_node =
        std::make_shared<RelJoin>(inputs[0], inputs[1], filter_rex, join_type);
    getRelAlgHints(join_ra, join_node);
    return join_node;
  }
  return std::make_shared<RelJoin>(inputs[0], inputs[1], filter_rex, join_type);
}

int64_t get_int_literal_field(const rapidjson::Value& obj,
                              const char field[],
                              const int64_t default_val) noexcept {
  const auto it = obj.FindMember(field);
  if (it == obj.MemberEnd()) {
    return default_val;
  }
  std::unique_ptr<RexLiteral> lit(parse_literal(it->value));
  CHECK_EQ(kDECIMAL, lit->getType());
  CHECK_EQ(unsigned(0), lit->getScale());
  CHECK_EQ(unsigned(0), lit->getTypeScale());
  return lit->getVal<int64_t>();
}

std::shared_ptr<RelSort> CiderRelAlgDispatcher::dispatchSort(
    const rapidjson::Value& sort_ra) {
  const auto inputs = getRelAlgInputs(sort_ra);
  CHECK_EQ(size_t(1), inputs.size());
  std::vector<SortField> collation;
  const auto& collation_arr = field(sort_ra, "collation");
  CHECK(collation_arr.IsArray());
  for (auto collation_arr_it = collation_arr.Begin();
       collation_arr_it != collation_arr.End();
       ++collation_arr_it) {
    const size_t field_idx = json_i64(field(*collation_arr_it, "field"));
    const auto sort_dir = parse_sort_direction(*collation_arr_it);
    const auto null_pos = parse_nulls_position(*collation_arr_it);
    collation.emplace_back(field_idx, sort_dir, null_pos);
  }
  auto limit = get_int_literal_field(sort_ra, "fetch", -1);
  const auto offset = get_int_literal_field(sort_ra, "offset", 0);
  auto ret =
      std::make_shared<RelSort>(collation, limit > 0 ? limit : 0, offset, inputs.front());
  ret->setEmptyResult(limit == 0);
  return ret;
}

std::shared_ptr<RelLogicalValues> CiderRelAlgDispatcher::dispatchLogicalValues(
    const rapidjson::Value& logical_values_ra) {
  const auto& tuple_type_arr = field(logical_values_ra, "type");
  CHECK(tuple_type_arr.IsArray());
  std::vector<TargetMetaInfo> tuple_type;
  for (auto tuple_type_arr_it = tuple_type_arr.Begin();
       tuple_type_arr_it != tuple_type_arr.End();
       ++tuple_type_arr_it) {
    const auto component_type = parse_type(*tuple_type_arr_it);
    const auto component_name = json_str(field(*tuple_type_arr_it, "name"));
    tuple_type.emplace_back(component_name, component_type);
  }
  const auto& inputs_arr = field(logical_values_ra, "inputs");
  CHECK(inputs_arr.IsArray());
  const auto& tuples_arr = field(logical_values_ra, "tuples");
  CHECK(tuples_arr.IsArray());

  if (inputs_arr.Size()) {
    throw QueryNotSupported("Inputs not supported in logical values yet.");
  }

  std::vector<RelLogicalValues::RowValues> values;
  if (tuples_arr.Size()) {
    for (const auto& row : tuples_arr.GetArray()) {
      CHECK(row.IsArray());
      const auto values_json = row.GetArray();
      if (!values.empty()) {
        CHECK_EQ(values[0].size(), values_json.Size());
      }
      values.emplace_back(RelLogicalValues::RowValues{});
      for (const auto& value : values_json) {
        CHECK(value.IsObject());
        CHECK(value.HasMember("literal"));
        values.back().emplace_back(parse_literal(value));
      }
    }
  }

  return std::make_shared<RelLogicalValues>(tuple_type, values);
}

std::shared_ptr<RelLogicalUnion> CiderRelAlgDispatcher::dispatchUnion(
    const rapidjson::Value& logical_union_ra) {
  auto inputs = getRelAlgInputs(logical_union_ra);
  auto const& all_type_bool = field(logical_union_ra, "all");
  CHECK(all_type_bool.IsBool());
  return std::make_shared<RelLogicalUnion>(std::move(inputs), all_type_bool.GetBool());
}

RelAlgInputs CiderRelAlgDispatcher::getRelAlgInputs(const rapidjson::Value& node) {
  if (node.HasMember("inputs")) {
    const auto str_input_ids = strings_from_json_array(field(node, "inputs"));
    RelAlgInputs ra_inputs;
    for (const auto& str_id : str_input_ids) {
      ra_inputs.push_back(nodes_[std::stoi(str_id)]);
    }
    return ra_inputs;
  }
  return {prev(node)};
}

std::pair<std::string, std::string> CiderRelAlgDispatcher::getKVOptionPair(
    std::string& str,
    size_t& pos) {
  auto option = str.substr(0, pos);
  std::string delim = "=";
  size_t delim_pos = option.find(delim);
  auto key = option.substr(0, delim_pos);
  auto val = option.substr(delim_pos + 1, option.length());
  str.erase(0, pos + delim.length() + 1);
  return {key, val};
}

ExplainedQueryHint CiderRelAlgDispatcher::parseHintString(std::string& hint_string) {
  std::string white_space_delim = " ";
  int l = hint_string.length();
  hint_string = hint_string.erase(0, 1).substr(0, l - 2);
  size_t pos = 0;
  if ((pos = hint_string.find("options:")) != std::string::npos) {
    // need to parse hint options
    std::vector<std::string> tokens;
    std::string hint_name = hint_string.substr(0, hint_string.find(white_space_delim));
    auto hint_type = RegisteredQueryHint::translateQueryHint(hint_name);
    bool kv_list_op = false;
    std::string raw_options = hint_string.substr(pos + 8, hint_string.length() - 2);
    if (raw_options.find('{') != std::string::npos) {
      kv_list_op = true;
    } else {
      CHECK(raw_options.find('[') != std::string::npos);
    }
    auto t1 = raw_options.erase(0, 1);
    raw_options = t1.substr(0, t1.length() - 1);
    std::string op_delim = ", ";
    if (kv_list_op) {
      // kv options
      std::unordered_map<std::string, std::string> kv_options;
      while ((pos = raw_options.find(op_delim)) != std::string::npos) {
        auto kv_pair = getKVOptionPair(raw_options, pos);
        kv_options.emplace(kv_pair.first, kv_pair.second);
      }
      // handle the last kv pair
      auto kv_pair = getKVOptionPair(raw_options, pos);
      kv_options.emplace(kv_pair.first, kv_pair.second);
      return {hint_type, true, false, true, kv_options};
    } else {
      std::vector<std::string> list_options;
      while ((pos = raw_options.find(op_delim)) != std::string::npos) {
        list_options.emplace_back(raw_options.substr(0, pos));
        raw_options.erase(0, pos + white_space_delim.length() + 1);
      }
      // handle the last option
      list_options.emplace_back(raw_options.substr(0, pos));
      return {hint_type, true, false, false, list_options};
    }
  } else {
    // marker hint: no extra option for this hint
    std::string hint_name = hint_string.substr(0, hint_string.find(white_space_delim));
    auto hint_type = RegisteredQueryHint::translateQueryHint(hint_name);
    return {hint_type, true, true, false};
  }
}

void CiderRelAlgDispatcher::getRelAlgHints(const rapidjson::Value& json_node,
                                           std::shared_ptr<RelAlgNode> node) {
  std::string hint_explained = json_str(field(json_node, "hints"));
  size_t pos = 0;
  std::string delim = "|";
  std::vector<std::string> hint_list;
  while ((pos = hint_explained.find(delim)) != std::string::npos) {
    hint_list.emplace_back(hint_explained.substr(0, pos));
    hint_explained.erase(0, pos + delim.length());
  }
  // handling the last one
  hint_list.emplace_back(hint_explained.substr(0, pos));

  const auto agg_node = std::dynamic_pointer_cast<RelAggregate>(node);
  if (agg_node) {
    for (std::string& hint : hint_list) {
      auto parsed_hint = parseHintString(hint);
      agg_node->addHint(parsed_hint);
    }
  }
  const auto project_node = std::dynamic_pointer_cast<RelProject>(node);
  if (project_node) {
    for (std::string& hint : hint_list) {
      auto parsed_hint = parseHintString(hint);
      project_node->addHint(parsed_hint);
    }
  }
  const auto scan_node = std::dynamic_pointer_cast<RelScan>(node);
  if (scan_node) {
    for (std::string& hint : hint_list) {
      auto parsed_hint = parseHintString(hint);
      scan_node->addHint(parsed_hint);
    }
  }
  const auto join_node = std::dynamic_pointer_cast<RelJoin>(node);
  if (join_node) {
    for (std::string& hint : hint_list) {
      auto parsed_hint = parseHintString(hint);
      join_node->addHint(parsed_hint);
    }
  }

  const auto compound_node = std::dynamic_pointer_cast<RelCompound>(node);
  if (compound_node) {
    for (std::string& hint : hint_list) {
      auto parsed_hint = parseHintString(hint);
      compound_node->addHint(parsed_hint);
    }
  }
}

std::shared_ptr<const RelAlgNode> CiderRelAlgDispatcher::prev(
    const rapidjson::Value& crt_node) {
  const auto id = node_id(crt_node);
  CHECK(id);
  CHECK_EQ(static_cast<size_t>(id), nodes_.size());
  return nodes_.back();
}

const TableDescriptor* getTableFromScanNode(const rapidjson::Value& scan_ra, MetaDesc meta) {
  std::cout<<"2"<<std::endl;
  const auto& table_json = field(scan_ra, "table");
  CHECK(table_json.IsArray());
  CHECK_EQ(unsigned(2), table_json.Size());
  const auto td = meta.getData(table_json[1].GetString());
  CHECK(td);
  return td;
}

void check_empty_inputs_field(const rapidjson::Value& node) {
  std::cout<<"3"<<std::endl;
  const auto& inputs_json = field(node, "inputs");
  CHECK(inputs_json.IsArray() && !inputs_json.Size());
}
std::vector<std::string> getFieldNamesFromScanNode(const rapidjson::Value& scan_ra) {
  std::cout<<"4"<<std::endl;
  const auto& fields_json = field(scan_ra, "fieldNames");
  std::cout<<"5"<<std::endl;
  return strings_from_json_array(fields_json);
}