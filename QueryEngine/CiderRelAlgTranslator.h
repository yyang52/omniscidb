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

#ifndef QUERYENGINE_CIDERRELALGTRANSLATOR_H
#define QUERYENGINE_CIDERRELALGTRANSLATOR_H

#include "Execute.h"
#include "RelAlgDagBuilder.h"

#include <ctime>
#include <memory>
#include <unordered_map>
#include <vector>

namespace Analyzer {

class Expr;

}  // namespace Analyzer

namespace query_state {

class QueryState;

}  // namespace query_state

class CiderRelAlgTranslator {
 public:
  CiderRelAlgTranslator(const std::vector<JoinType>& join_types,
                        const time_t now,
                        const bool just_explain)
      : join_types_(join_types)
      , now_(now)
      , generated_geos_ops_(false)
      , just_explain_(just_explain) {}

  std::shared_ptr<Analyzer::Expr> translateScalarRex(const RexScalar* rex) const;

  static std::shared_ptr<Analyzer::Expr> translateAggregateRex(
      const RexAgg* rex,
      const std::vector<std::shared_ptr<Analyzer::Expr>>& scalar_sources);

  static std::shared_ptr<Analyzer::Expr> translateLiteral(const RexLiteral*);

  bool generated_geos_ops() { return generated_geos_ops_; }

 private:
  std::shared_ptr<Analyzer::Expr> translateScalarSubquery(const RexSubQuery*) const;

  std::shared_ptr<Analyzer::Expr> translateInput(const RexInput*) const;

  std::shared_ptr<Analyzer::Expr> translateUoper(const RexOperator*) const;

  std::shared_ptr<Analyzer::Expr> translateInOper(const RexOperator*) const;

  std::shared_ptr<Analyzer::Expr> getInIntegerSetExpr(std::shared_ptr<Analyzer::Expr> arg,
                                                      const ResultSet& val_set) const;

  std::shared_ptr<Analyzer::Expr> translateOper(const RexOperator*) const;

  std::shared_ptr<Analyzer::Expr> translateOverlapsOper(const RexOperator*) const;

  std::shared_ptr<Analyzer::Expr> translateCase(const RexCase*) const;

  std::shared_ptr<Analyzer::Expr> translateLike(const RexFunctionOperator*) const;

  std::shared_ptr<Analyzer::Expr> translateRegexp(const RexFunctionOperator*) const;

  std::shared_ptr<Analyzer::Expr> translateLikely(const RexFunctionOperator*) const;

  std::shared_ptr<Analyzer::Expr> translateUnlikely(const RexFunctionOperator*) const;

  std::shared_ptr<Analyzer::Expr> translateExtract(const RexFunctionOperator*) const;

  std::shared_ptr<Analyzer::Expr> translateDateadd(const RexFunctionOperator*) const;

  std::shared_ptr<Analyzer::Expr> translateDatePlusMinus(const RexOperator*) const;

  std::shared_ptr<Analyzer::Expr> translateDatediff(const RexFunctionOperator*) const;

  std::shared_ptr<Analyzer::Expr> translateDatepart(const RexFunctionOperator*) const;

  std::shared_ptr<Analyzer::Expr> translateLength(const RexFunctionOperator*) const;

  std::shared_ptr<Analyzer::Expr> translateKeyForString(const RexFunctionOperator*) const;

  std::shared_ptr<Analyzer::Expr> translateSampleRatio(const RexFunctionOperator*) const;

  std::shared_ptr<Analyzer::Expr> translateCurrentUser(const RexFunctionOperator*) const;

  std::shared_ptr<Analyzer::Expr> translateLower(const RexFunctionOperator*) const;

  std::shared_ptr<Analyzer::Expr> translateCardinality(const RexFunctionOperator*) const;

  std::shared_ptr<Analyzer::Expr> translateItem(const RexFunctionOperator*) const;

  std::shared_ptr<Analyzer::Expr> translateCurrentDate() const;

  std::shared_ptr<Analyzer::Expr> translateCurrentTime() const;

  std::shared_ptr<Analyzer::Expr> translateCurrentTimestamp() const;

  std::shared_ptr<Analyzer::Expr> translateDatetime(const RexFunctionOperator*) const;

  std::shared_ptr<Analyzer::Expr> translateHPTLiteral(const RexFunctionOperator*) const;

  std::shared_ptr<Analyzer::Expr> translateAbs(const RexFunctionOperator*) const;

  std::shared_ptr<Analyzer::Expr> translateSign(const RexFunctionOperator*) const;

  std::shared_ptr<Analyzer::Expr> translateOffsetInFragment() const;

  std::shared_ptr<Analyzer::Expr> translateArrayFunction(
      const RexFunctionOperator*) const;

  std::shared_ptr<Analyzer::Expr> translateFunction(const RexFunctionOperator*) const;

  std::shared_ptr<Analyzer::Expr> translateWindowFunction(
      const RexWindowFunctionOperator*) const;

  Analyzer::ExpressionPtrVector translateFunctionArgs(const RexFunctionOperator*) const;

  std::shared_ptr<Analyzer::Expr> translateUnaryGeoFunction(
      const RexFunctionOperator*) const;

  std::shared_ptr<Analyzer::Expr> translateBinaryGeoFunction(
      const RexFunctionOperator*) const;

  std::shared_ptr<Analyzer::Expr> translateTernaryGeoFunction(
      const RexFunctionOperator*) const;

  std::shared_ptr<Analyzer::Expr> translateFunctionWithGeoArg(
      const RexFunctionOperator*) const;

  std::shared_ptr<Analyzer::Expr> translateGeoComparison(const RexOperator*) const;

  std::shared_ptr<Analyzer::Expr> translateGeoProjection(const RexFunctionOperator*,
                                                         SQLTypeInfo&,
                                                         const bool with_bounds) const;

  std::shared_ptr<Analyzer::Expr> translateGeoPredicate(const RexFunctionOperator*,
                                                        SQLTypeInfo&,
                                                        const bool with_bounds) const;

  std::shared_ptr<Analyzer::Expr> translateGeoBinaryConstructor(
      const RexFunctionOperator*,
      SQLTypeInfo&,
      const bool with_bounds) const;

  std::shared_ptr<Analyzer::Expr> translateGeoOverlapsOper(const RexOperator*) const;

  std::vector<std::shared_ptr<Analyzer::Expr>> translateGeoFunctionArg(
      const RexScalar* rex_scalar,
      SQLTypeInfo& arg_ti,
      int32_t& lindex,
      const bool with_bounds,
      const bool with_render_group,
      const bool expand_geo_col,
      const bool is_projection = false) const;

  std::vector<std::shared_ptr<Analyzer::Expr>> translateGeoColumn(
      const RexInput*,
      SQLTypeInfo&,
      const bool with_bounds,
      const bool with_render_group,
      const bool expand_geo_col) const;

  std::vector<std::shared_ptr<Analyzer::Expr>> translateGeoLiteral(const RexLiteral*,
                                                                   SQLTypeInfo&,
                                                                   bool) const;

  std::shared_ptr<const query_state::QueryState> query_state_;
  const std::vector<JoinType> join_types_;
  time_t now_;
  mutable bool generated_geos_ops_;
  const bool just_explain_;
};


using namespace std::literals;

#endif  // QUERYENGINE_CIDERRELALGTRANSLATOR_H
