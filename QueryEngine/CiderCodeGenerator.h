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

#pragma once

#include "CgenState.h"
#include "CiderWorkUnit.h"
#include "CodeGenerator.h"
#include "CompilationContext.h"
#include "Descriptors/QueryCompilationDescriptor.h"
#include "Execute.h"
#include "GpuSharedMemoryUtils.h"
#include "LLVMFunctionAttributesUtil.h"
#include "OutputBufferInitialization.h"
#include "QueryTemplateGenerator.h"

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/FormattedStream.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Instrumentation.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/InstSimplifyPass.h>
#include <llvm/Transforms/Utils.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include <memory>

struct CiderMetrics {
  int64_t kernel_queue_time_ms_ = 0;
  int64_t compilation_queue_time_ms_ = 0;
};

class CiderCodeGenerator {
 public:
  std::tuple<CompilationResult, std::unique_ptr<QueryMemoryDescriptor>> compileWorkUnit(
      const std::vector<InputTableInfo>& query_infos,
      const PlanState::DeletedColumnsMap& deleted_cols_map,
      const RelAlgExecutionUnit& ra_exe_unit,
      const CompilationOptions& co,
      const ExecutionOptions& eo,
      const CudaMgr_Namespace::CudaMgr* cuda_mgr,
      const bool allow_lazy_fetch,
      std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
      const size_t max_groups_buffer_entry_guess,
      const int8_t crt_min_byte_width,
      const bool has_cardinality_estimation,
      ColumnCacheMap& column_cache,
      RenderInfo* render_info);

  std::shared_ptr<CompilationContext> getCodeFromCache(
      const CodeCacheKey& key,
      const CodeCache& cache,
      std::shared_ptr<CgenState> cgen_state);

  void addCodeToCache(const CodeCacheKey& key,
                      std::shared_ptr<CompilationContext> compilation_context,
                      llvm::Module* module,
                      CodeCache& cache);

  std::shared_ptr<CompilationContext> optimizeAndCodegenCPU(
      llvm::Function* query_func,
      llvm::Function* multifrag_query_func,
      const std::unordered_set<llvm::Function*>& live_funcs,
      const CompilationOptions& co,
      std::shared_ptr<CgenState> cgen_state);

  std::shared_ptr<CompilationContext> optimizeAndCodegenGPU(
      llvm::Function* query_func,
      llvm::Function* multifrag_query_func,
      std::unordered_set<llvm::Function*>& live_funcs,
      const bool no_inline,
      const CudaMgr_Namespace::CudaMgr* cuda_mgr,
      const CompilationOptions& co,
      std::shared_ptr<CgenState> cgen_state);

 private:
  CodeCache cpu_code_cache_;
  CodeCache gpu_code_cache_;
  Catalog_Namespace::Catalog* catalog_;
  Executor* executor_;
  std::shared_ptr<CgenState> cgen_state_;
  std::shared_ptr<PlanState> plan_state_;
  const unsigned block_size_x_;
  std::shared_ptr<CiderMetrics> metrics_;
};
