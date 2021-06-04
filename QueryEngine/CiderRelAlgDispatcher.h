#include <rapidjson/error/en.h>
#include <rapidjson/error/error.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include "CalciteDeserializerUtils.h"
#include "JsonAccessors.h"
#include "RelAlgDagBuilder.h"

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

class CiderRelAlgDispatcher {
 public:
  CiderRelAlgDispatcher() {}
  // todo: we need data type.
  std::vector<std::shared_ptr<RelAlgNode>> run(const rapidjson::Value& rels);

 private:
  // TODO: placeholder replace Scan

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
