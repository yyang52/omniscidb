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

#ifndef OMNISCI_CIDERRESULTPROVIDER_H
#define OMNISCI_CIDERRESULTPROVIDER_H

#include "QueryRunner/QueryRunner.h"

class ExecutionResult;

class CiderResultIterator {
 public:
  std::shared_ptr<ExecutionResult> next();
};

class CiderResultProvider {
 public:
  std::shared_ptr<CiderResultIterator> getIterator();
};

#endif  // OMNISCI_CIDERRESULTPROVIDER_H
