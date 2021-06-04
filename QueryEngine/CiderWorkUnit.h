
#include <memory>

#include "RelAlgDagBuilder.h"

class CiderWorkUnit {
 public:
  CiderWorkUnit(){};
  static CiderWorkUnit createCiderWorkUnit(std::shared_ptr<RelAlgNode> plan);

  int executeWithData(char* inputData) { return -1; };
};