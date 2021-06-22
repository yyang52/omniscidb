#pragma once

#include "Catalog/TableDescriptor.h"

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>


 class MetaDesc{
  public:
    const TableDescriptor* getData(std::string tableName)const {
    // we give option not to populate fragmenter (default true/yes) as it can be heavy for
    // pure metadata calls
    auto tableDescIt = tableInfos.find(to_upper(tableName));
    if (tableDescIt == tableInfos.end()) {  // check to make sure table exists
      return nullptr;
    }
    TableDescriptor* td = tableDescIt->second;
    std::unique_lock<std::mutex> td_lock(*td->mutex_.get());
    return td;  // returns pointer to table descriptor
  };

    void buildData(std::string tableName, TableDescriptor* tbPtr) {
      tableInfos.insert(std::pair<std::string, TableDescriptor*>(to_upper(tableName), tbPtr));
    };
  private:
    std::map<std::string, TableDescriptor*> tableInfos;
  };

