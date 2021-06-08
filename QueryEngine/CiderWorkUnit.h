
#include <memory>

#include "CiderRelAlgExecution.h"
#include "QueryRewrite.h"
#include "RelAlgDagBuilder.h"

struct CiderWorkUnit {
  CiderRelAlgExecutionUnit exe_unit;
  const RelAlgNode* body;
  const size_t max_groups_buffer_entry_guess;
  std::unique_ptr<QueryRewriter> query_rewriter;
  // const std::vector<size_t> input_permutation;
  // const std::vector<size_t> left_deep_join_input_sizes;
};

class CiderUnitModuler {
 public:
  static CiderUnitModuler createCiderUnitModuler(std::shared_ptr<RelAlgNode> plan);
  // std::unique_ptr<QueryMemoryDescriptor> compile();
  int executeWithData(char* inputData) { return -1; };

 private:
  CiderUnitModuler(std::shared_ptr<CiderWorkUnit> worker_unit) : worker_unit_(worker_unit){};
  std::shared_ptr<CiderWorkUnit> worker_unit_;
};