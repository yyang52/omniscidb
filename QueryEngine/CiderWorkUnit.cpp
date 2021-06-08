#include "CiderWorkUnit.h"
#include "CiderRelAlgTranslator.h"
#include "ExpressionRewrite.h"

#include "Shared/sqltypes.h"

const RelAlgNode* get_data_sink(const RelAlgNode* ra_node) {
  if (auto table_func = dynamic_cast<const RelTableFunction*>(ra_node)) {
    return table_func;
  }
  if (auto join = dynamic_cast<const RelJoin*>(ra_node)) {
    CHECK_EQ(size_t(2), join->inputCount());
    return join;
  }
  if (!dynamic_cast<const RelLogicalUnion*>(ra_node)) {
    CHECK_EQ(size_t(1), ra_node->inputCount());
  }
  auto only_src = ra_node->getInput(0);
  const bool is_join = dynamic_cast<const RelJoin*>(only_src) ||
                       dynamic_cast<const RelLeftDeepInnerJoin*>(only_src);
  return is_join ? only_src : ra_node;
}

int table_id_from_ra(const RelAlgNode* ra_node) {
  const auto scan_ra = dynamic_cast<const RelScan*>(ra_node);
  if (scan_ra) {
    const auto td = scan_ra->getTableDescriptor();
    CHECK(td);
    return td->tableId;
  }
  return -ra_node->getId();
}

std::unordered_map<const RelAlgNode*, int> get_input_nest_levels(
    const RelAlgNode* ra_node,
    const std::vector<size_t>& input_permutation) {
  const auto data_sink_node = get_data_sink(ra_node);
  std::unordered_map<const RelAlgNode*, int> input_to_nest_level;
  for (size_t input_idx = 0; input_idx < data_sink_node->inputCount(); ++input_idx) {
    const auto input_node_idx =
        input_permutation.empty() ? input_idx : input_permutation[input_idx];
    const auto input_ra = data_sink_node->getInput(input_node_idx);
    // Having a non-zero mapped value (input_idx) results in the query being interpretted
    // as a JOIN within CodeGenerator::codegenColVar() due to rte_idx being set to the
    // mapped value (input_idx) which originates here. This would be incorrect for UNION.
    size_t const idx = dynamic_cast<const RelLogicalUnion*>(ra_node) ? 0 : input_idx;
    const auto it_ok = input_to_nest_level.emplace(input_ra, idx);
    CHECK(it_ok.second);
    LOG_IF(INFO, !input_permutation.empty())
        << "Assigned input " << input_ra->toString() << " to nest level " << input_idx;
  }
  return input_to_nest_level;
}

std::vector<std::shared_ptr<Analyzer::Expr>> synthesize_inputs(
    const RelAlgNode* ra_node,
    const size_t nest_level,
    const std::vector<TargetMetaInfo>& in_metainfo,
    const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level) {
  CHECK_LE(size_t(1), ra_node->inputCount());
  CHECK_GE(size_t(2), ra_node->inputCount());
  const auto input = ra_node->getInput(nest_level);
  const auto it_rte_idx = input_to_nest_level.find(input);
  CHECK(it_rte_idx != input_to_nest_level.end());
  const int rte_idx = it_rte_idx->second;
  const int table_id = table_id_from_ra(input);
  std::vector<std::shared_ptr<Analyzer::Expr>> inputs;
  const auto scan_ra = dynamic_cast<const RelScan*>(input);
  int input_idx = 0;
  for (const auto& input_meta : in_metainfo) {
    inputs.push_back(
        std::make_shared<Analyzer::ColumnVar>(input_meta.get_type_info(),
                                              table_id,
                                              scan_ra ? input_idx + 1 : input_idx,
                                              rte_idx));
    ++input_idx;
  }
  return inputs;
}

JoinType get_join_type(const RelAlgNode* ra) {
  auto sink = get_data_sink(ra);
  if (auto join = dynamic_cast<const RelJoin*>(sink)) {
    return join->getJoinType();
  }
  if (dynamic_cast<const RelLeftDeepInnerJoin*>(sink)) {
    return JoinType::INNER;
  }

  return JoinType::INVALID;
}

std::shared_ptr<Analyzer::Expr> set_transient_dict(
    const std::shared_ptr<Analyzer::Expr> expr) {
  const auto& ti = expr->get_type_info();
  if (!ti.is_string() || ti.get_compression() != kENCODING_NONE) {
    return expr;
  }
  auto transient_dict_ti = ti;
  transient_dict_ti.set_compression(kENCODING_DICT);
  transient_dict_ti.set_comp_param(TRANSIENT_DICT_ID);
  transient_dict_ti.set_fixed_size();
  return expr->add_cast(transient_dict_ti);
}

std::list<std::shared_ptr<Analyzer::Expr>> translate_groupby_exprs(
    const RelCompound* compound,
    const std::vector<std::shared_ptr<Analyzer::Expr>>& scalar_sources) {
  if (!compound->isAggregate()) {
    return {nullptr};
  }
  std::list<std::shared_ptr<Analyzer::Expr>> groupby_exprs;
  for (size_t group_idx = 0; group_idx < compound->getGroupByCount(); ++group_idx) {
    groupby_exprs.push_back(set_transient_dict(scalar_sources[group_idx]));
  }
  return groupby_exprs;
}
std::shared_ptr<Analyzer::Expr> cast_dict_to_none(
    const std::shared_ptr<Analyzer::Expr>& input) {
  const auto& input_ti = input->get_type_info();
  if (input_ti.is_string() && input_ti.get_compression() == kENCODING_DICT) {
    return input->add_cast(SQLTypeInfo(kTEXT, input_ti.get_notnull()));
  }
  return input;
}

std::list<std::shared_ptr<Analyzer::Expr>> translate_groupby_exprs(
    const RelAggregate* aggregate,
    const std::vector<std::shared_ptr<Analyzer::Expr>>& scalar_sources) {
  std::list<std::shared_ptr<Analyzer::Expr>> groupby_exprs;
  for (size_t group_idx = 0; group_idx < aggregate->getGroupByCount(); ++group_idx) {
    groupby_exprs.push_back(set_transient_dict(scalar_sources[group_idx]));
  }
  return groupby_exprs;
}
std::vector<Analyzer::Expr*> translate_targets(
    std::vector<std::shared_ptr<Analyzer::Expr>>& target_exprs_owned,
    const std::vector<std::shared_ptr<Analyzer::Expr>>& scalar_sources,
    const std::list<std::shared_ptr<Analyzer::Expr>>& groupby_exprs,
    const RelCompound* compound,
    const CiderRelAlgTranslator& translator,
    const ExecutorType executor_type) {
  std::vector<Analyzer::Expr*> target_exprs;
  for (size_t i = 0; i < compound->size(); ++i) {
    const auto target_rex = compound->getTargetExpr(i);
    const auto target_rex_agg = dynamic_cast<const RexAgg*>(target_rex);
    std::shared_ptr<Analyzer::Expr> target_expr;
    if (target_rex_agg) {
      target_expr =
          CiderRelAlgTranslator::translateAggregateRex(target_rex_agg, scalar_sources);
    } else {
      const auto target_rex_scalar = dynamic_cast<const RexScalar*>(target_rex);
      const auto target_rex_ref = dynamic_cast<const RexRef*>(target_rex_scalar);
      if (target_rex_ref) {
        const auto ref_idx = target_rex_ref->getIndex();
        CHECK_GE(ref_idx, size_t(1));
        CHECK_LE(ref_idx, groupby_exprs.size());
        const auto groupby_expr = *std::next(groupby_exprs.begin(), ref_idx - 1);
        target_expr = var_ref(groupby_expr.get(), Analyzer::Var::kGROUPBY, ref_idx);
      } else {
        target_expr = translator.translateScalarRex(target_rex_scalar);
        auto rewritten_expr = rewrite_expr(target_expr.get());
        target_expr = fold_expr(rewritten_expr.get());
        if (executor_type == ExecutorType::Native) {
          try {
            target_expr = set_transient_dict(target_expr);
          } catch (...) {
            // noop
          }
        } else {
          target_expr = cast_dict_to_none(target_expr);
        }
      }
    }
    CHECK(target_expr);
    target_exprs_owned.push_back(target_expr);
    target_exprs.push_back(target_expr.get());
  }
  return target_exprs;
}

std::vector<Analyzer::Expr*> translate_targets(
    std::vector<std::shared_ptr<Analyzer::Expr>>& target_exprs_owned,
    const std::vector<std::shared_ptr<Analyzer::Expr>>& scalar_sources,
    const std::list<std::shared_ptr<Analyzer::Expr>>& groupby_exprs,
    const RelAggregate* aggregate,
    const CiderRelAlgTranslator& translator) {
  std::vector<Analyzer::Expr*> target_exprs;
  size_t group_key_idx = 1;
  for (const auto& groupby_expr : groupby_exprs) {
    auto target_expr =
        var_ref(groupby_expr.get(), Analyzer::Var::kGROUPBY, group_key_idx++);
    target_exprs_owned.push_back(target_expr);
    target_exprs.push_back(target_expr.get());
  }

  for (const auto& target_rex_agg : aggregate->getAggExprs()) {
    auto target_expr = CiderRelAlgTranslator::translateAggregateRex(target_rex_agg.get(),
                                                                    scalar_sources);
    CHECK(target_expr);
    target_expr = fold_expr(target_expr.get());
    target_exprs_owned.push_back(target_expr);
    target_exprs.push_back(target_expr.get());
  }
  return target_exprs;
}
bool is_count_distinct(const Analyzer::Expr* expr) {
  const auto agg_expr = dynamic_cast<const Analyzer::AggExpr*>(expr);
  return agg_expr && agg_expr->get_is_distinct();
}

bool is_agg(const Analyzer::Expr* expr) {
  const auto agg_expr = dynamic_cast<const Analyzer::AggExpr*>(expr);
  if (agg_expr && agg_expr->get_contains_agg()) {
    auto agg_type = agg_expr->get_aggtype();
    if (agg_type == SQLAgg::kMIN || agg_type == SQLAgg::kMAX ||
        agg_type == SQLAgg::kSUM || agg_type == SQLAgg::kAVG) {
      return true;
    }
  }
  return false;
}

inline SQLTypeInfo get_logical_type_for_expr(const Analyzer::Expr& expr) {
  if (is_count_distinct(&expr)) {
    return SQLTypeInfo(kBIGINT, false);
  } else if (is_agg(&expr)) {
    return get_nullable_logical_type_info(expr.get_type_info());
  }
  return get_logical_type_info(expr.get_type_info());
}

template <class RA>
std::vector<TargetMetaInfo> get_targets_meta(
    const RA* ra_node,
    const std::vector<Analyzer::Expr*>& target_exprs) {
  std::vector<TargetMetaInfo> targets_meta;
  CHECK_EQ(ra_node->size(), target_exprs.size());
  for (size_t i = 0; i < ra_node->size(); ++i) {
    CHECK(target_exprs[i]);
    // TODO(alex): remove the count distinct type fixup.
    targets_meta.emplace_back(ra_node->getFieldName(i),
                              get_logical_type_for_expr(*target_exprs[i]),
                              target_exprs[i]->get_type_info());
  }
  return targets_meta;
}

template <>
std::vector<TargetMetaInfo> get_targets_meta(
    const RelFilter* filter,
    const std::vector<Analyzer::Expr*>& target_exprs) {
  RelAlgNode const* input0 = filter->getInput(0);
  if (auto const* input = dynamic_cast<RelCompound const*>(input0)) {
    return get_targets_meta(input, target_exprs);
  } else if (auto const* input = dynamic_cast<RelProject const*>(input0)) {
    return get_targets_meta(input, target_exprs);
  } else if (auto const* input = dynamic_cast<RelLogicalUnion const*>(input0)) {
    return get_targets_meta(input, target_exprs);
  } else if (auto const* input = dynamic_cast<RelAggregate const*>(input0)) {
    return get_targets_meta(input, target_exprs);
  } else if (auto const* input = dynamic_cast<RelScan const*>(input0)) {
    return get_targets_meta(input, target_exprs);
  }
  UNREACHABLE() << "Unhandled node type: " << input0->toString();
  return {};
}

CiderWorkUnit createAggregateCiderWorkUnit(const RelAggregate* aggregate,
                                           const SortInfo& sort_info,
                                           const bool just_explain = false) {
  const auto input_to_nest_level = get_input_nest_levels(aggregate, {});  //

  const auto join_type = get_join_type(aggregate);
  time_t now_(0);
  CiderRelAlgTranslator translator({join_type}, now_, just_explain);
  CHECK_EQ(size_t(1), aggregate->inputCount());
  const auto source = aggregate->getInput(0);
  const auto& in_metainfo = source->getOutputMetainfo();
  const auto scalar_sources =
      synthesize_inputs(aggregate, size_t(0), in_metainfo, input_to_nest_level);
  const auto groupby_exprs = translate_groupby_exprs(aggregate, scalar_sources);
  std::vector<std::shared_ptr<Analyzer::Expr>> target_exprs_owned;
  const auto target_exprs = translate_targets(
      target_exprs_owned, scalar_sources, groupby_exprs, aggregate, translator);
  const auto targets_meta = get_targets_meta(aggregate, target_exprs);
  aggregate->setOutputMetainfo(targets_meta);
  int max_groups_buffer_entry_default_guess = 16384;
  // std::shared_ptr<const query_state::QueryState> query_state=
  // std::make_shared<query_state::QueryState>();
  return {CiderRelAlgExecutionUnit{{},
                                   {},
                                   {},
                                   groupby_exprs,
                                   target_exprs,
                                   nullptr,
                                   sort_info,
                                   0,
                                   RegisteredQueryHint::defaults(),
                                   false,
                                   std::nullopt},
          aggregate,
          max_groups_buffer_entry_default_guess,
          nullptr};
}

CiderUnitModuler CiderUnitModuler::createCiderUnitModuler(
    std::shared_ptr<RelAlgNode> plan) {
  const auto aggregate = dynamic_cast<const RelAggregate*>(plan.get());
  bool just_explain = false;
  std::shared_ptr<CiderWorkUnit> work_unit;
  if (aggregate) {
    work_unit = std::make_shared<CiderWorkUnit>(createAggregateCiderWorkUnit(
        aggregate, {{}, SortAlgorithm::Default, 0, 0}, just_explain));
  }
  return std::move(CiderUnitModuler(work_unit));
}

// void CiderUnitModuler::createCiderUnitModuler(CiderWorkUnit worker) {
//   query_comp_desc_owned
// }
