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

#include "CiderCodeGenerator.h"

extern std::unique_ptr<llvm::Module> g_rt_module;
extern std::unique_ptr<llvm::Module> udf_cpu_module;
extern std::unique_ptr<llvm::Module> udf_gpu_module;
extern std::unique_ptr<llvm::Module> rt_udf_cpu_module;
extern std::unique_ptr<llvm::Module> rt_udf_gpu_module;

std::shared_ptr<CompilationContext> CiderCodeGenerator::optimizeAndCodegenCPU(
    llvm::Function* query_func,
    llvm::Function* multifrag_query_func,
    const std::unordered_set<llvm::Function*>& live_funcs,
    const CompilationOptions& co,
    std::shared_ptr<CgenState> cgen_state) {
  auto module = multifrag_query_func->getParent();
  CodeCacheKey key{serialize_llvm_object(query_func),
                   serialize_llvm_object(cgen_state->row_func_)};
  if (cgen_state->filter_func_) {
    key.push_back(serialize_llvm_object(cgen_state->filter_func_));
  }
  for (const auto helper : cgen_state->helper_functions_) {
    key.push_back(serialize_llvm_object(helper));
  }
  auto cached_code = getCodeFromCache(key, cpu_code_cache_, cgen_state);
  if (cached_code) {
    return cached_code;
  }

  if (cgen_state->needs_geos_) {
    throw std::runtime_error("GEOS is disabled in this build");
  }

  auto execution_engine = CodeGenerator::generateNativeCPUCode(
      query_func, live_funcs, co);  //  this is what we want.
  auto cpu_compilation_context =
      std::make_shared<CpuCompilationContext>(std::move(execution_engine));
  cpu_compilation_context->setFunctionPointer(multifrag_query_func);
  addCodeToCache(key, cpu_compilation_context, module, cpu_code_cache_);
  return cpu_compilation_context;
}

std::shared_ptr<CompilationContext> CiderCodeGenerator::optimizeAndCodegenGPU(
    llvm::Function* query_func,
    llvm::Function* multifrag_query_func,
    std::unordered_set<llvm::Function*>& live_funcs,
    const bool no_inline,
    const CudaMgr_Namespace::CudaMgr* cuda_mgr,
    const CompilationOptions& co,
    std::shared_ptr<CgenState> cgen_state) {
#ifdef HAVE_CUDA  // we have some todos when enable CUDA, remember to double check!!!
  auto module = multifrag_query_func->getParent();

  CHECK(cuda_mgr);
  CodeCacheKey key{serialize_llvm_object(query_func),
                   serialize_llvm_object(cgen_state->row_func_)};
  if (cgen_state->filter_func_) {
    key.push_back(serialize_llvm_object(cgen_state->filter_func_));
  }
  for (const auto helper : cgen_state->helper_functions_) {
    key.push_back(serialize_llvm_object(helper));
  }
  auto cached_code = getCodeFromCache(key, gpu_code_cache_, cgen_state);
  if (cached_code) {
    return cached_code;
  }

  bool row_func_not_inlined = false;
  if (no_inline) {
    for (auto it = llvm::inst_begin(cgen_state->row_func_),
              e = llvm::inst_end(cgen_state->row_func_);
         it != e;
         ++it) {
      if (llvm::isa<llvm::CallInst>(*it)) {
        auto& get_gv_call = llvm::cast<llvm::CallInst>(*it);
        if (get_gv_call.getCalledFunction()->getName() == "array_size" ||
            get_gv_call.getCalledFunction()->getName() == "linear_probabilistic_count") {
          mark_function_never_inline(cgen_state->row_func_);
          row_func_not_inlined = true;
          break;
        }
      }
    }
  }

  initializeNVPTXBackend();                                         // todo
  CodeGenerator::GPUTarget gpu_target{nvptx_target_machine_.get(),  // todo
                                      cuda_mgr,
                                      cider_executor::blockSize(catalog_),  // todo
                                      cgen_state.get(),
                                      row_func_not_inlined};
  std::shared_ptr<GpuCompilationContext> compilation_context;

  if (cider::check_module_requires_libdevice(module)) {
    if (g_rt_libdevice_module == nullptr) {  // todo
      // raise error
      throw std::runtime_error(
          "libdevice library is not available but required by the UDF module");
    }

    // Bind libdevice it to the current module
    CodeGenerator::link_udf_module(g_rt_libdevice_module,  // todo
                                   *module,
                                   cgen_state.get(),
                                   llvm::Linker::Flags::OverrideFromSrc);

    // activate nvvm-reflect-ftz flag on the module
    module->addModuleFlag(llvm::Module::Override, "nvvm-reflect-ftz", (int)1);
    for (llvm::Function& fn : *module) {
      fn.addFnAttr("nvptx-f32ftz", "true");
    }
  }

  try {
    compilation_context = CodeGenerator::generateNativeGPUCode(
        query_func, multifrag_query_func, live_funcs, co, gpu_target);
    addCodeToCache(key, compilation_context, module, gpu_code_cache_);
  } catch (CudaMgr_Namespace::CudaErrorException& cuda_error) {
    if (cuda_error.getStatus() == CUDA_ERROR_OUT_OF_MEMORY) {
      // Thrown if memory not able to be allocated on gpu
      // Retry once after evicting portion of code cache
      LOG(WARNING) << "Failed to allocate GPU memory for generated code. Evicting "
                   << g_fraction_code_cache_to_evict * 100.  // todo
                   << "% of GPU code cache and re-trying.";
      gpu_code_cache_.evictFractionEntries(g_fraction_code_cache_to_evict);  // todo
      compilation_context = CodeGenerator::generateNativeGPUCode(
          query_func, multifrag_query_func, live_funcs, co, gpu_target);
      addCodeToCache(key, compilation_context, module, gpu_code_cache_);
    } else {
      throw;
    }
  }
  CHECK(compilation_context);
  return compilation_context;
#else
  return nullptr;
#endif
}

std::shared_ptr<CompilationContext> CiderCodeGenerator::getCodeFromCache(
    const CodeCacheKey& key,
    const CodeCache& cache,
    std::shared_ptr<CgenState> cgen_state) {
  auto it = cache.find(key);
  if (it != cache.cend()) {
    delete cgen_state->module_;
    cgen_state->module_ = it->second.second;
    return it->second.first;
  }
  return {};
}

void CiderCodeGenerator::addCodeToCache(
    const CodeCacheKey& key,
    std::shared_ptr<CompilationContext> compilation_context,
    llvm::Module* module,
    CodeCache& cache) {
  cache.put(key,
            std::make_pair<std::shared_ptr<CompilationContext>, decltype(module)>(
                std::move(compilation_context), std::move(module)));
}

// {
//   auto query_comp_desc_owned = std::make_unique<QueryCompilationDescriptor>();
//   std::unique_ptr<QueryMemoryDescriptor> query_mem_desc_owned;
//   query_mem_desc_owned = query_comp_desc_owned->compile(max_groups_buffer_entry_guess,
//                                                         crt_min_byte_width,
//                                                         has_cardinality_estimation,
//                                                         ra_exe_unit,
//                                                         query_infos,
//                                                         deleted_cols_map,
//                                                         column_fetcher,
//                                                         {device_type,
//                                                          co.hoist_literals,
//                                                          co.opt_level,
//                                                          co.with_dynamic_watchdog,
//                                                          co.allow_lazy_fetch,
//                                                          co.filter_on_deleted_column,
//                                                          co.explain_type,
//                                                          co.register_intel_jit_listener},
//                                                         eo,
//                                                         render_info,
//                                                         this)
// }

namespace cider {

#ifdef HAVE_CUDA

// check if linking with libdevice is required
// libdevice functions have a __nv_* prefix
bool check_module_requires_libdevice(llvm::Module* module) {
  for (llvm::Function& F : *module) {
    if (F.hasName() && F.getName().startswith("__nv_")) {
      LOG(INFO) << "Module requires linking with libdevice: " << std::string(F.getName());
      return true;
    }
  }
  LOG(DEBUG1) << "module does not require linking against libdevice";
  return false;
}

// Adds the missing intrinsics declarations to the given module
void add_intrinsics_to_module(llvm::Module* module) {
  for (llvm::Function& F : *module) {
    for (llvm::Instruction& I : instructions(F)) {
      if (llvm::IntrinsicInst* ii = llvm::dyn_cast<llvm::IntrinsicInst>(&I)) {
        if (llvm::Intrinsic::isOverloaded(ii->getIntrinsicID())) {
          llvm::Type* Tys[] = {ii->getFunctionType()->getReturnType()};
          llvm::Function& decl_fn =
              *llvm::Intrinsic::getDeclaration(module, ii->getIntrinsicID(), Tys);
          ii->setCalledFunction(&decl_fn);
        } else {
          // inserts the declaration into the module if not present
          llvm::Intrinsic::getDeclaration(module, ii->getIntrinsicID());
        }
      }
    }
  }
}

#endif

// These are some stateless util methods, copy from NativeCodegen.cpp.
// We use these methods with cider:: prefix
// todo: move to another file?
void eliminate_dead_self_recursive_funcs(
    llvm::Module& M,
    const std::unordered_set<llvm::Function*>& live_funcs) {
  std::vector<llvm::Function*> dead_funcs;
  for (auto& F : M) {
    bool bAlive = false;
    if (live_funcs.count(&F)) {
      continue;
    }
    for (auto U : F.users()) {
      auto* C = llvm::dyn_cast<const llvm::CallInst>(U);
      if (!C || C->getParent()->getParent() != &F) {
        bAlive = true;
        break;
      }
    }
    if (!bAlive) {
      dead_funcs.push_back(&F);
    }
  }
  for (auto pFn : dead_funcs) {
    pFn->eraseFromParent();
  }
}

void optimize_ir(llvm::Function* query_func,
                 llvm::Module* module,
                 llvm::legacy::PassManager& pass_manager,
                 const std::unordered_set<llvm::Function*>& live_funcs,
                 const CompilationOptions& co) {
  pass_manager.add(llvm::createAlwaysInlinerLegacyPass());
  pass_manager.add(llvm::createPromoteMemoryToRegisterPass());
  pass_manager.add(llvm::createInstSimplifyLegacyPass());
  pass_manager.add(llvm::createInstructionCombiningPass());
  pass_manager.add(llvm::createGlobalOptimizerPass());

  pass_manager.add(llvm::createLICMPass());
  if (co.opt_level == ExecutorOptLevel::LoopStrengthReduction) {
    pass_manager.add(llvm::createLoopStrengthReducePass());
  }
  pass_manager.run(*module);

  eliminate_dead_self_recursive_funcs(*module, live_funcs);
}

size_t get_shared_memory_size(const bool shared_mem_used,
                              const QueryMemoryDescriptor* query_mem_desc_ptr) {
  return shared_mem_used
             ? (query_mem_desc_ptr->getRowSize() * query_mem_desc_ptr->getEntryCount())
             : 0;
}

bool is_gpu_shared_mem_supported(const QueryMemoryDescriptor* query_mem_desc_ptr,
                                 const RelAlgExecutionUnit& ra_exe_unit,
                                 const CudaMgr_Namespace::CudaMgr* cuda_mgr,
                                 const ExecutorDeviceType device_type,
                                 const unsigned gpu_blocksize,
                                 const unsigned num_blocks_per_mp) {
  if (device_type == ExecutorDeviceType::CPU) {
    return false;
  }
  if (query_mem_desc_ptr->didOutputColumnar()) {
    return false;
  }
  CHECK(query_mem_desc_ptr);
  CHECK(cuda_mgr);
  /*
   * We only use shared memory strategy if GPU hardware provides native shared
   * memory atomics support. From CUDA Toolkit documentation:
   * https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html#atomic-ops "Like
   * Maxwell, Pascal [and Volta] provides native shared memory atomic operations
   * for 32-bit integer arithmetic, along with native 32 or 64-bit compare-and-swap
   * (CAS)."
   *
   **/
  if (!cuda_mgr->isArchMaxwellOrLaterForAll()) {
    return false;
  }

  if (query_mem_desc_ptr->getQueryDescriptionType() ==
          QueryDescriptionType::NonGroupedAggregate &&
      g_enable_smem_non_grouped_agg &&
      query_mem_desc_ptr->countDistinctDescriptorsLogicallyEmpty()) {
    // TODO: relax this, if necessary
    if (gpu_blocksize < query_mem_desc_ptr->getEntryCount()) {
      return false;
    }
    // skip shared memory usage when dealing with 1) variable length targets, 2)
    // not a COUNT aggregate
    const auto target_infos =
        target_exprs_to_infos(ra_exe_unit.target_exprs, *query_mem_desc_ptr);
    std::unordered_set<SQLAgg> supported_aggs{kCOUNT};
    if (std::find_if(target_infos.begin(),
                     target_infos.end(),
                     [&supported_aggs](const TargetInfo& ti) {
                       if (ti.sql_type.is_varlen() ||
                           !supported_aggs.count(ti.agg_kind)) {
                         return true;
                       } else {
                         return false;
                       }
                     }) == target_infos.end()) {
      return true;
    }
  }
  if (query_mem_desc_ptr->getQueryDescriptionType() ==
          QueryDescriptionType::GroupByPerfectHash &&
      g_enable_smem_group_by) {
    /**
     * To simplify the implementation for practical purposes, we
     * initially provide shared memory support for cases where there are at most as many
     * entries in the output buffer as there are threads within each GPU device. In
     * order to relax this assumption later, we need to add a for loop in generated
     * codes such that each thread loops over multiple entries.
     * TODO: relax this if necessary
     */
    if (gpu_blocksize < query_mem_desc_ptr->getEntryCount()) {
      return false;
    }

    // Fundamentally, we should use shared memory whenever the output buffer
    // is small enough so that we can fit it in the shared memory and yet expect
    // good occupancy.
    // For now, we allow keyless, row-wise layout, and only for perfect hash
    // group by operations.
    if (query_mem_desc_ptr->hasKeylessHash() &&
        query_mem_desc_ptr->countDistinctDescriptorsLogicallyEmpty() &&
        !query_mem_desc_ptr->useStreamingTopN()) {
      const size_t shared_memory_threshold_bytes = std::min(
          g_gpu_smem_threshold == 0 ? SIZE_MAX : g_gpu_smem_threshold,
          cuda_mgr->getMinSharedMemoryPerBlockForAllDevices() / num_blocks_per_mp);
      const auto output_buffer_size =
          query_mem_desc_ptr->getRowSize() * query_mem_desc_ptr->getEntryCount();
      if (output_buffer_size > shared_memory_threshold_bytes) {
        return false;
      }

      // skip shared memory usage when dealing with 1) variable length targets, 2)
      // non-basic aggregates (COUNT, SUM, MIN, MAX, AVG)
      // TODO: relax this if necessary
      const auto target_infos =
          target_exprs_to_infos(ra_exe_unit.target_exprs, *query_mem_desc_ptr);
      std::unordered_set<SQLAgg> supported_aggs{kCOUNT};
      if (g_enable_smem_grouped_non_count_agg) {
        supported_aggs = {kCOUNT, kMIN, kMAX, kSUM, kAVG};
      }
      if (std::find_if(target_infos.begin(),
                       target_infos.end(),
                       [&supported_aggs](const TargetInfo& ti) {
                         if (ti.sql_type.is_varlen() ||
                             !supported_aggs.count(ti.agg_kind)) {
                           return true;
                         } else {
                           return false;
                         }
                       }) == target_infos.end()) {
        return true;
      }
    }
  }
  return false;
}

#ifndef NDEBUG
std::string serialize_llvm_metadata_footnotes(llvm::Function* query_func,
                                              CgenState* cgen_state) {
  std::string llvm_ir;
  std::unordered_set<llvm::MDNode*> md;

  // Loop over all instructions in the query function.
  for (auto bb_it = query_func->begin(); bb_it != query_func->end(); ++bb_it) {
    for (auto instr_it = bb_it->begin(); instr_it != bb_it->end(); ++instr_it) {
      llvm::SmallVector<std::pair<unsigned, llvm::MDNode*>, 100> imd;
      instr_it->getAllMetadata(imd);
      for (auto [kind, node] : imd) {
        md.insert(node);
      }
    }
  }

  // Loop over all instructions in the row function.
  for (auto bb_it = cgen_state->row_func_->begin(); bb_it != cgen_state->row_func_->end();
       ++bb_it) {
    for (auto instr_it = bb_it->begin(); instr_it != bb_it->end(); ++instr_it) {
      llvm::SmallVector<std::pair<unsigned, llvm::MDNode*>, 100> imd;
      instr_it->getAllMetadata(imd);
      for (auto [kind, node] : imd) {
        md.insert(node);
      }
    }
  }

  // Loop over all instructions in the filter function.
  if (cgen_state->filter_func_) {
    for (auto bb_it = cgen_state->filter_func_->begin();
         bb_it != cgen_state->filter_func_->end();
         ++bb_it) {
      for (auto instr_it = bb_it->begin(); instr_it != bb_it->end(); ++instr_it) {
        llvm::SmallVector<std::pair<unsigned, llvm::MDNode*>, 100> imd;
        instr_it->getAllMetadata(imd);
        for (auto [kind, node] : imd) {
          md.insert(node);
        }
      }
    }
  }

  // Sort the metadata by canonical number and convert to text.
  if (!md.empty()) {
    std::map<size_t, std::string> sorted_strings;
    for (auto p : md) {
      std::string str;
      llvm::raw_string_ostream os(str);
      p->print(os, cgen_state->module_, true);
      os.flush();
      auto fields = split(str, {}, 1);
      if (fields.empty() || fields[0].empty()) {
        continue;
      }
      sorted_strings.emplace(std::stoul(fields[0].substr(1)), str);
    }
    llvm_ir += "\n";
    for (auto [id, text] : sorted_strings) {
      llvm_ir += text;
      llvm_ir += "\n";
    }
  }

  return llvm_ir;
}
#endif  // NDEBUG

llvm::StringRef get_gpu_target_triple_string() {
  return llvm::StringRef("nvptx64-nvidia-cuda");
}

llvm::StringRef get_gpu_data_layout() {
  return llvm::StringRef(
      "e-p:64:64:64-i1:8:8-i8:8:8-"
      "i16:16:16-i32:32:32-i64:64:64-"
      "f32:32:32-f64:64:64-v16:16:16-"
      "v32:32:32-v64:64:64-v128:128:128-n16:32:64");
}

void bind_pos_placeholders(const std::string& pos_fn_name,
                           const bool use_resume_param,
                           llvm::Function* query_func,
                           llvm::Module* module) {
  for (auto it = llvm::inst_begin(query_func), e = llvm::inst_end(query_func); it != e;
       ++it) {
    if (!llvm::isa<llvm::CallInst>(*it)) {
      continue;
    }
    auto& pos_call = llvm::cast<llvm::CallInst>(*it);
    if (std::string(pos_call.getCalledFunction()->getName()) == pos_fn_name) {
      if (use_resume_param) {
        const auto error_code_arg = get_arg_by_name(query_func, "error_code");
        llvm::ReplaceInstWithInst(
            &pos_call,
            llvm::CallInst::Create(module->getFunction(pos_fn_name + "_impl"),
                                   error_code_arg));
      } else {
        llvm::ReplaceInstWithInst(
            &pos_call,
            llvm::CallInst::Create(module->getFunction(pos_fn_name + "_impl")));
      }
      break;
    }
  }
}

void set_row_func_argnames(llvm::Function* row_func,
                           const size_t in_col_count,
                           const size_t agg_col_count,
                           const bool hoist_literals) {
  auto arg_it = row_func->arg_begin();

  if (agg_col_count) {
    for (size_t i = 0; i < agg_col_count; ++i) {
      arg_it->setName("out");
      ++arg_it;
    }
  } else {
    arg_it->setName("group_by_buff");
    ++arg_it;
    arg_it->setName("crt_matched");
    ++arg_it;
    arg_it->setName("total_matched");
    ++arg_it;
    arg_it->setName("old_total_matched");
    ++arg_it;
    arg_it->setName("max_matched");
    ++arg_it;
  }

  arg_it->setName("agg_init_val");
  ++arg_it;

  arg_it->setName("pos");
  ++arg_it;

  arg_it->setName("frag_row_off");
  ++arg_it;

  arg_it->setName("num_rows_per_scan");
  ++arg_it;

  if (hoist_literals) {
    arg_it->setName("literals");
    ++arg_it;
  }

  for (size_t i = 0; i < in_col_count; ++i) {
    arg_it->setName("col_buf" + std::to_string(i));
    ++arg_it;
  }

  arg_it->setName("join_hash_tables");
}

llvm::Function* create_row_function(const size_t in_col_count,
                                    const size_t agg_col_count,
                                    const bool hoist_literals,
                                    llvm::Module* module,
                                    llvm::LLVMContext& context) {
  std::vector<llvm::Type*> row_process_arg_types;

  if (agg_col_count) {
    // output (aggregate) arguments
    for (size_t i = 0; i < agg_col_count; ++i) {
      row_process_arg_types.push_back(llvm::Type::getInt64PtrTy(context));
    }
  } else {
    // group by buffer
    row_process_arg_types.push_back(llvm::Type::getInt64PtrTy(context));
    // current match count
    row_process_arg_types.push_back(llvm::Type::getInt32PtrTy(context));
    // total match count passed from the caller
    row_process_arg_types.push_back(llvm::Type::getInt32PtrTy(context));
    // old total match count returned to the caller
    row_process_arg_types.push_back(llvm::Type::getInt32PtrTy(context));
    // max matched (total number of slots in the output buffer)
    row_process_arg_types.push_back(llvm::Type::getInt32PtrTy(context));
  }

  // aggregate init values
  row_process_arg_types.push_back(llvm::Type::getInt64PtrTy(context));

  // position argument
  row_process_arg_types.push_back(llvm::Type::getInt64Ty(context));

  // fragment row offset argument
  row_process_arg_types.push_back(llvm::Type::getInt64PtrTy(context));

  // number of rows for each scan
  row_process_arg_types.push_back(llvm::Type::getInt64PtrTy(context));

  // literals buffer argument
  if (hoist_literals) {
    row_process_arg_types.push_back(llvm::Type::getInt8PtrTy(context));
  }

  // column buffer arguments
  for (size_t i = 0; i < in_col_count; ++i) {
    row_process_arg_types.emplace_back(llvm::Type::getInt8PtrTy(context));
  }

  // join hash table argument
  row_process_arg_types.push_back(llvm::Type::getInt64PtrTy(context));

  // generate the function
  auto ft =
      llvm::FunctionType::get(get_int_type(32, context), row_process_arg_types, false);

  auto row_func =
      llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "row_func", module);

  // set the row function argument names; for debugging purposes only
  set_row_func_argnames(row_func, in_col_count, agg_col_count, hoist_literals);

  return row_func;
}

std::vector<std::string> get_agg_fnames(const std::vector<Analyzer::Expr*>& target_exprs,
                                        const bool is_group_by) {
  std::vector<std::string> result;
  for (size_t target_idx = 0, agg_col_idx = 0; target_idx < target_exprs.size();
       ++target_idx, ++agg_col_idx) {
    const auto target_expr = target_exprs[target_idx];
    CHECK(target_expr);
    const auto target_type_info = target_expr->get_type_info();
    const auto agg_expr = dynamic_cast<Analyzer::AggExpr*>(target_expr);
    const bool is_varlen =
        (target_type_info.is_string() &&
         target_type_info.get_compression() == kENCODING_NONE) ||
        target_type_info.is_array();  // TODO: should it use is_varlen_array() ?
    if (!agg_expr || agg_expr->get_aggtype() == kSAMPLE) {
      result.emplace_back(target_type_info.is_fp() ? "agg_id_double" : "agg_id");
      if (is_varlen) {
        result.emplace_back("agg_id");
      }
      if (target_type_info.is_geometry()) {
        result.emplace_back("agg_id");
        for (auto i = 2; i < 2 * target_type_info.get_physical_coord_cols(); ++i) {
          result.emplace_back("agg_id");
        }
      }
      continue;
    }
    const auto agg_type = agg_expr->get_aggtype();
    const auto& agg_type_info =
        agg_type != kCOUNT ? agg_expr->get_arg()->get_type_info() : target_type_info;
    switch (agg_type) {
      case kAVG: {
        if (!agg_type_info.is_integer() && !agg_type_info.is_decimal() &&
            !agg_type_info.is_fp()) {
          throw std::runtime_error("AVG is only valid on integer and floating point");
        }
        result.emplace_back((agg_type_info.is_integer() || agg_type_info.is_time())
                                ? "agg_sum"
                                : "agg_sum_double");
        result.emplace_back((agg_type_info.is_integer() || agg_type_info.is_time())
                                ? "agg_count"
                                : "agg_count_double");
        break;
      }
      case kMIN: {
        if (agg_type_info.is_string() || agg_type_info.is_array() ||
            agg_type_info.is_geometry()) {
          throw std::runtime_error(
              "MIN on strings, arrays or geospatial types not supported yet");
        }
        result.emplace_back((agg_type_info.is_integer() || agg_type_info.is_time())
                                ? "agg_min"
                                : "agg_min_double");
        break;
      }
      case kMAX: {
        if (agg_type_info.is_string() || agg_type_info.is_array() ||
            agg_type_info.is_geometry()) {
          throw std::runtime_error(
              "MAX on strings, arrays or geospatial types not supported yet");
        }
        result.emplace_back((agg_type_info.is_integer() || agg_type_info.is_time())
                                ? "agg_max"
                                : "agg_max_double");
        break;
      }
      case kSUM: {
        if (!agg_type_info.is_integer() && !agg_type_info.is_decimal() &&
            !agg_type_info.is_fp()) {
          throw std::runtime_error("SUM is only valid on integer and floating point");
        }
        result.emplace_back((agg_type_info.is_integer() || agg_type_info.is_time())
                                ? "agg_sum"
                                : "agg_sum_double");
        break;
      }
      case kCOUNT:
        result.emplace_back(agg_expr->get_is_distinct() ? "agg_count_distinct"
                                                        : "agg_count");
        break;
      case kSINGLE_VALUE: {
        result.emplace_back(agg_type_info.is_fp() ? "agg_id_double" : "agg_id");
        break;
      }
      case kSAMPLE: {
        // Note that varlen SAMPLE arguments are handled separately above
        result.emplace_back(agg_type_info.is_fp() ? "agg_id_double" : "agg_id");
        break;
      }
      case kAPPROX_COUNT_DISTINCT:
        result.emplace_back("agg_approximate_count_distinct");
        break;
      case kAPPROX_MEDIAN:
        result.emplace_back("agg_approx_median");
        break;
      default:
        CHECK(false);
    }
  }
  return result;
}

// Iterate through multifrag_query_func, replacing calls to query_fname with query_func.
void bind_query(llvm::Function* query_func,
                const std::string& query_fname,
                llvm::Function* multifrag_query_func,
                llvm::Module* module) {
  std::vector<llvm::CallInst*> query_stubs;
  for (auto it = llvm::inst_begin(multifrag_query_func),
            e = llvm::inst_end(multifrag_query_func);
       it != e;
       ++it) {
    if (!llvm::isa<llvm::CallInst>(*it)) {
      continue;
    }
    auto& query_call = llvm::cast<llvm::CallInst>(*it);
    if (std::string(query_call.getCalledFunction()->getName()) == query_fname) {
      query_stubs.push_back(&query_call);
    }
  }
  for (auto& S : query_stubs) {
    std::vector<llvm::Value*> args;
    for (size_t i = 0; i < S->getNumArgOperands(); ++i) {
      args.push_back(S->getArgOperand(i));
    }
    llvm::ReplaceInstWithInst(S, llvm::CallInst::Create(query_func, args, ""));
  }
}
}  // namespace cider

namespace cider_executor {
void nukeOldState(const bool allow_lazy_fetch,
                  const std::vector<InputTableInfo>& query_infos,
                  const PlanState::DeletedColumnsMap& deleted_cols_map,
                  const RelAlgExecutionUnit* ra_exe_unit,
                  std::shared_ptr<CiderMetrics> metrics,
                  std::shared_ptr<CgenState> cgen_state,
                  std::shared_ptr<PlanState> plan_state,
                   Executor* executor) {
  metrics->kernel_queue_time_ms_ = 0;
  metrics->compilation_queue_time_ms_ = 0;
  const bool contains_left_deep_outer_join =
      ra_exe_unit && std::find_if(ra_exe_unit->join_quals.begin(),
                                  ra_exe_unit->join_quals.end(),
                                  [](const JoinCondition& join_condition) {
                                    return join_condition.type == JoinType::LEFT;
                                  }) != ra_exe_unit->join_quals.end();
  cgen_state.reset(new CgenState(query_infos.size(), contains_left_deep_outer_join));
  plan_state.reset(new PlanState(allow_lazy_fetch && !contains_left_deep_outer_join,
                                 query_infos,
                                 deleted_cols_map,
                                 executor));

  unsigned blockSize(Catalog_Namespace::Catalog catalog) const {
    CHECK(catalog);
    const auto cuda_mgr = catalog->getDataMgr().getCudaMgr();
    if (!cuda_mgr) {
      return 0;
    }
    const auto& dev_props = cuda_mgr->getAllDeviceProperties();
    return block_size_x_ ? block_size_x_ : dev_props.front().maxThreadsPerBlock;
  }
};

}  // namespace cider_executor

std::tuple<CompilationResult, std::unique_ptr<QueryMemoryDescriptor>>
CiderCodeGenerator::compileWorkUnit(const std::vector<InputTableInfo>& query_infos,
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
                                    RenderInfo* render_info) {
  auto timer = DEBUG_TIMER(__func__);

  if (co.device_type == ExecutorDeviceType::GPU) {
    const auto cuda_mgr = catalog_->getDataMgr().getCudaMgr();
    if (!cuda_mgr) {
      throw QueryMustRunOnCpu();
    }
  }

#ifndef NDEBUG
  static std::uint64_t counter = 0;
  ++counter;
  VLOG(1) << "CODEGEN #" << counter << ":";
  LOG(IR) << "CODEGEN #" << counter << ":";
  LOG(PTX) << "CODEGEN #" << counter << ":";
  LOG(ASM) << "CODEGEN #" << counter << ":";
#endif

  cider_executor::nukeOldState(allow_lazy_fetch,
                               query_infos,
                               deleted_cols_map,
                               &ra_exe_unit,
                               metrics_,
                               cgen_state_,
                               plan_state_,
                               executor_);

  GroupByAndAggregate group_by_and_aggregate(
      executor_,
      co.device_type,
      ra_exe_unit,
      query_infos,
      row_set_mem_owner,
      has_cardinality_estimation ? std::optional<int64_t>(max_groups_buffer_entry_guess)
                                 : std::nullopt);
  auto query_mem_desc =
      group_by_and_aggregate.initQueryMemoryDescriptor(eo.allow_multifrag,
                                                       max_groups_buffer_entry_guess,
                                                       crt_min_byte_width,
                                                       render_info,
                                                       eo.output_columnar_hint);

  if (query_mem_desc->getQueryDescriptionType() ==
          QueryDescriptionType::GroupByBaselineHash &&
      !has_cardinality_estimation &&
      (!render_info || !render_info->isPotentialInSituRender()) && !eo.just_explain) {
    const auto col_range_info = group_by_and_aggregate.getColRangeInfo();
    throw CardinalityEstimationRequired(col_range_info.max - col_range_info.min);
  }

  const bool output_columnar = query_mem_desc->didOutputColumnar();
  const bool gpu_shared_mem_optimization =
      cider::is_gpu_shared_mem_supported(query_mem_desc.get(),
                                         ra_exe_unit,
                                         cuda_mgr,
                                         co.device_type,
                                         cuda_mgr ? cider_executor::blockSize(catalog_) : 1,
                                         cuda_mgr ? cider_executor::numBlocksPerMP() : 1);
  if (gpu_shared_mem_optimization) {
    // disable interleaved bins optimization on the GPU
    query_mem_desc->setHasInterleavedBinsOnGpu(false);
    LOG(DEBUG1) << "GPU shared memory is used for the " +
                       query_mem_desc->queryDescTypeToString() + " query(" +
                       std::to_string(cider::get_shared_memory_size(
                           gpu_shared_mem_optimization, query_mem_desc.get())) +
                       " out of " + std::to_string(g_gpu_smem_threshold) + " bytes).";
  }

  const GpuSharedMemoryContext gpu_smem_context(
      cider::get_shared_memory_size(gpu_shared_mem_optimization, query_mem_desc.get()));

  if (co.device_type == ExecutorDeviceType::GPU) {
    const size_t num_count_distinct_descs =
        query_mem_desc->getCountDistinctDescriptorsSize();
    for (size_t i = 0; i < num_count_distinct_descs; i++) {
      const auto& count_distinct_descriptor =
          query_mem_desc->getCountDistinctDescriptor(i);
      if (count_distinct_descriptor.impl_type_ == CountDistinctImplType::StdSet ||
          (count_distinct_descriptor.impl_type_ != CountDistinctImplType::Invalid &&
           !co.hoist_literals)) {
        throw QueryMustRunOnCpu();
      }
    }
  }

  // Read the module template and target either CPU or GPU
  // by binding the stream position functions to the right implementation:
  // stride access for GPU, contiguous for CPU
  auto rt_module_copy = llvm::CloneModule(
      *g_rt_module.get(), cgen_state_->vmap_, [](const llvm::GlobalValue* gv) {
        auto func = llvm::dyn_cast<llvm::Function>(gv);
        if (!func) {
          return true;
        }
        return (func->getLinkage() == llvm::GlobalValue::LinkageTypes::PrivateLinkage ||
                func->getLinkage() == llvm::GlobalValue::LinkageTypes::InternalLinkage ||
                CodeGenerator::alwaysCloneRuntimeFunction(func));
      });
  if (co.device_type == ExecutorDeviceType::CPU) {
    if (is_udf_module_present(true)) {
      CodeGenerator::link_udf_module(udf_cpu_module, *rt_module_copy, cgen_state_.get());
    }
    if (is_rt_udf_module_present(true)) {
      CodeGenerator::link_udf_module(
          rt_udf_cpu_module, *rt_module_copy, cgen_state_.get());
    }
  } else {
    rt_module_copy->setDataLayout(cider::get_gpu_data_layout());
    rt_module_copy->setTargetTriple(cider::get_gpu_target_triple_string());
    if (is_udf_module_present()) {
      CodeGenerator::link_udf_module(udf_gpu_module, *rt_module_copy, cgen_state_.get());
    }
    if (is_rt_udf_module_present()) {
      CodeGenerator::link_udf_module(
          rt_udf_gpu_module, *rt_module_copy, cgen_state_.get());
    }
  }

  cgen_state_->module_ = rt_module_copy.release();
  AUTOMATIC_IR_METADATA(cgen_state_.get());

  auto agg_fnames =
      cider::get_agg_fnames(ra_exe_unit.target_exprs, !ra_exe_unit.groupby_exprs.empty());

  const auto agg_slot_count = ra_exe_unit.estimator ? size_t(1) : agg_fnames.size();

  const bool is_group_by{query_mem_desc->isGroupBy()};
  auto [query_func, row_func_call] = is_group_by
                                         ? query_group_by_template(cgen_state_->module_,
                                                                   co.hoist_literals,
                                                                   *query_mem_desc,
                                                                   co.device_type,
                                                                   ra_exe_unit.scan_limit,
                                                                   gpu_smem_context)
                                         : query_template(cgen_state_->module_,
                                                          agg_slot_count,
                                                          co.hoist_literals,
                                                          !!ra_exe_unit.estimator,
                                                          gpu_smem_context);
  cider::bind_pos_placeholders("pos_start", true, query_func, cgen_state_->module_);
  cider::bind_pos_placeholders("group_buff_idx", false, query_func, cgen_state_->module_);
  cider::bind_pos_placeholders("pos_step", false, query_func, cgen_state_->module_);

  cgen_state_->query_func_ = query_func;
  cgen_state_->row_func_call_ = row_func_call;
  cgen_state_->query_func_entry_ir_builder_.SetInsertPoint(
      &query_func->getEntryBlock().front());

  // Generate the function signature and column head fetches s.t.
  // double indirection isn't needed in the inner loop
  auto& fetch_bb = query_func->front();
  llvm::IRBuilder<> fetch_ir_builder(&fetch_bb);
  fetch_ir_builder.SetInsertPoint(&*fetch_bb.begin());
  auto col_heads = generate_column_heads_load(ra_exe_unit.input_col_descs.size(),
                                              query_func->args().begin(),
                                              fetch_ir_builder,
                                              cgen_state_->context_);
  CHECK_EQ(ra_exe_unit.input_col_descs.size(), col_heads.size());

  cgen_state_->row_func_ = cider::create_row_function(ra_exe_unit.input_col_descs.size(),
                                                      is_group_by ? 0 : agg_slot_count,
                                                      co.hoist_literals,
                                                      cgen_state_->module_,
                                                      cgen_state_->context_);
  CHECK(cgen_state_->row_func_);
  cgen_state_->row_func_bb_ =
      llvm::BasicBlock::Create(cgen_state_->context_, "entry", cgen_state_->row_func_);

  if (g_enable_filter_function) {
    auto filter_func_ft =
        llvm::FunctionType::get(get_int_type(32, cgen_state_->context_), {}, false);
    cgen_state_->filter_func_ = llvm::Function::Create(filter_func_ft,
                                                       llvm::Function::ExternalLinkage,
                                                       "filter_func",
                                                       cgen_state_->module_);
    CHECK(cgen_state_->filter_func_);
    cgen_state_->filter_func_bb_ = llvm::BasicBlock::Create(
        cgen_state_->context_, "entry", cgen_state_->filter_func_);
  }

  cgen_state_->current_func_ = cgen_state_->row_func_;
  cgen_state_->ir_builder_.SetInsertPoint(cgen_state_->row_func_bb_);

  // todo: remove executor
  executor_->preloadFragOffsets(ra_exe_unit.input_descs, query_infos);
  RelAlgExecutionUnit body_execution_unit = ra_exe_unit;
  const auto join_loops =
      executor_->buildJoinLoops(body_execution_unit, co, eo, query_infos, column_cache);

  plan_state_->allocateLocalColumnIds(ra_exe_unit.input_col_descs);
  // todo: remove executor
  const auto is_not_deleted_bb =
      executor_->codegenSkipDeletedOuterTableRow(ra_exe_unit, co);
  if (is_not_deleted_bb) {
    cgen_state_->row_func_bb_ = is_not_deleted_bb;
  }
  if (!join_loops.empty()) {
    // todo: remove executor
    executor_->codegenJoinLoops(join_loops,
                                body_execution_unit,
                                group_by_and_aggregate,
                                query_func,
                                cgen_state_->row_func_bb_,
                                *(query_mem_desc.get()),
                                co,
                                eo);
  } else {
    // todo: remove executor
    const bool can_return_error = executor_->compileBody(
        ra_exe_unit, group_by_and_aggregate, *query_mem_desc, co, gpu_smem_context);
    if (can_return_error || cgen_state_->needs_error_check_ || eo.with_dynamic_watchdog ||
        eo.allow_runtime_query_interrupt) {
      // todo: remove executor

      executor_->createErrorCheckControlFlow(query_func,
                                             eo.with_dynamic_watchdog,
                                             eo.allow_runtime_query_interrupt,
                                             co.device_type,
                                             group_by_and_aggregate.query_infos_);
    }
  }
  std::vector<llvm::Value*> hoisted_literals;

  if (co.hoist_literals) {
    VLOG(1) << "number of hoisted literals: "
            << cgen_state_->query_func_literal_loads_.size()
            << " / literal buffer usage: " << cgen_state_->getLiteralBufferUsage(0)
            << " bytes";
  }

  if (co.hoist_literals && !cgen_state_->query_func_literal_loads_.empty()) {
    // we have some hoisted literals...
    hoisted_literals = executor_->inlineHoistedLiterals();
  }

  // replace the row func placeholder call with the call to the actual row func
  std::vector<llvm::Value*> row_func_args;
  for (size_t i = 0; i < cgen_state_->row_func_call_->getNumArgOperands(); ++i) {
    row_func_args.push_back(cgen_state_->row_func_call_->getArgOperand(i));
  }
  row_func_args.insert(row_func_args.end(), col_heads.begin(), col_heads.end());
  row_func_args.push_back(get_arg_by_name(query_func, "join_hash_tables"));
  // push hoisted literals arguments, if any
  row_func_args.insert(
      row_func_args.end(), hoisted_literals.begin(), hoisted_literals.end());
  llvm::ReplaceInstWithInst(
      cgen_state_->row_func_call_,
      llvm::CallInst::Create(cgen_state_->row_func_, row_func_args, ""));

  // replace the filter func placeholder call with the call to the actual filter func
  if (cgen_state_->filter_func_) {
    std::vector<llvm::Value*> filter_func_args;
    for (auto arg_it = cgen_state_->filter_func_args_.begin();
         arg_it != cgen_state_->filter_func_args_.end();
         ++arg_it) {
      filter_func_args.push_back(arg_it->first);
    }
    llvm::ReplaceInstWithInst(
        cgen_state_->filter_func_call_,
        llvm::CallInst::Create(cgen_state_->filter_func_, filter_func_args, ""));
  }

  // Aggregate
  plan_state_->init_agg_vals_ =
      init_agg_val_vec(ra_exe_unit.target_exprs, ra_exe_unit.quals, *query_mem_desc);

  /*
   * If we have decided to use GPU shared memory (decision is not made here), then
   * we generate proper code for extra components that it needs (buffer initialization and
   * gpu reduction from shared memory to global memory). We then replace these functions
   * into the already compiled query_func (replacing two placeholders, write_back_nop and
   * init_smem_nop). The rest of the code should be as before (row_func, etc.).
   */
  if (gpu_smem_context.isSharedMemoryUsed()) {
    if (query_mem_desc->getQueryDescriptionType() ==
        QueryDescriptionType::GroupByPerfectHash) {
      GpuSharedMemCodeBuilder gpu_smem_code(
          cgen_state_->module_,
          cgen_state_->context_,
          *query_mem_desc,
          target_exprs_to_infos(ra_exe_unit.target_exprs, *query_mem_desc),
          plan_state_->init_agg_vals_);
      gpu_smem_code.codegen();
      gpu_smem_code.injectFunctionsInto(query_func);

      // helper functions are used for caching purposes later
      cgen_state_->helper_functions_.push_back(gpu_smem_code.getReductionFunction());
      cgen_state_->helper_functions_.push_back(gpu_smem_code.getInitFunction());
      LOG(IR) << gpu_smem_code.toString();
    }
  }

  auto multifrag_query_func = cgen_state_->module_->getFunction(
      "multifrag_query" + std::string(co.hoist_literals ? "_hoisted_literals" : ""));
  CHECK(multifrag_query_func);

  if (co.device_type == ExecutorDeviceType::GPU && eo.allow_multifrag) {
    executor_->insertErrorCodeChecker(
        multifrag_query_func, co.hoist_literals, eo.allow_runtime_query_interrupt);
  }

  cider::bind_query(
      query_func,
      "query_stub" + std::string(co.hoist_literals ? "_hoisted_literals" : ""),
      multifrag_query_func,
      cgen_state_->module_);

  std::vector<llvm::Function*> root_funcs{query_func, cgen_state_->row_func_};
  if (cgen_state_->filter_func_) {
    root_funcs.push_back(cgen_state_->filter_func_);
  }
  auto live_funcs = CodeGenerator::markDeadRuntimeFuncs(
      *cgen_state_->module_, root_funcs, {multifrag_query_func});

  // Always inline the row function and the filter function.
  // We don't want register spills in the inner loops.
  // LLVM seems to correctly free up alloca instructions
  // in these functions even when they are inlined.
  mark_function_always_inline(cgen_state_->row_func_);
  if (cgen_state_->filter_func_) {
    mark_function_always_inline(cgen_state_->filter_func_);
  }

#ifndef NDEBUG
  // Add helpful metadata to the LLVM IR for debugging.
  AUTOMATIC_IR_METADATA_DONE();
#endif

  // Serialize the important LLVM IR functions to text for SQL EXPLAIN.
  std::string llvm_ir;
  if (eo.just_explain) {
    if (co.explain_type == ExecutorExplainType::Optimized) {
#ifdef WITH_JIT_DEBUG
      throw std::runtime_error(
          "Explain optimized not available when JIT runtime debug symbols are enabled");
#else
      // Note that we don't run the NVVM reflect pass here. Use LOG(IR) to get the
      // optimized IR after NVVM reflect
      llvm::legacy::PassManager pass_manager;
      cider::optimize_ir(query_func, cgen_state_->module_, pass_manager, live_funcs, co);
#endif  // WITH_JIT_DEBUG
    }
    llvm_ir =
        serialize_llvm_object(multifrag_query_func) + serialize_llvm_object(query_func) +
        serialize_llvm_object(cgen_state_->row_func_) +
        (cgen_state_->filter_func_ ? serialize_llvm_object(cgen_state_->filter_func_)
                                   : "");

#ifndef NDEBUG
    llvm_ir += serialize_llvm_metadata_footnotes(query_func, cgen_state_.get());
#endif
  }

  LOG(IR) << "\n\n" << query_mem_desc->toString() << "\n";
  LOG(IR) << "IR for the "
          << (co.device_type == ExecutorDeviceType::CPU ? "CPU:\n" : "GPU:\n");
#ifdef NDEBUG
  LOG(IR) << serialize_llvm_object(query_func)
          << serialize_llvm_object(cgen_state_->row_func_)
          << (cgen_state_->filter_func_ ? serialize_llvm_object(cgen_state_->filter_func_)
                                        : "")
          << "\nEnd of IR";
#else
  LOG(IR) << serialize_llvm_object(cgen_state_->module_) << "\nEnd of IR";
#endif

  // Run some basic validation checks on the LLVM IR before code is generated below.
  verify_function_ir(cgen_state_->row_func_);
  if (cgen_state_->filter_func_) {
    verify_function_ir(cgen_state_->filter_func_);
  }

  // Generate final native code from the LLVM IR.
  return std::make_tuple(
      CompilationResult{
          co.device_type == ExecutorDeviceType::CPU
              ? optimizeAndCodegenCPU(
                    query_func, multifrag_query_func, live_funcs, co, cgen_state_)
              // : optimizeAndCodegenCPU(query_func, multifrag_query_func, live_funcs, co,
              // cgen_state_),
              : optimizeAndCodegenGPU(query_func,
                                      multifrag_query_func,
                                      live_funcs,
                                      is_group_by || ra_exe_unit.estimator,
                                      cuda_mgr,
                                      co,
                                      cgen_state_),
          cgen_state_->getLiterals(),
          output_columnar,
          llvm_ir,
          std::move(gpu_smem_context)},
      std::move(query_mem_desc));
}
