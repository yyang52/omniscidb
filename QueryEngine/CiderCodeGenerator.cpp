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
extern std::unique_ptr<llvm::Module> g_rt_geos_module;
extern std::unique_ptr<llvm::Module> udf_cpu_module;
extern std::unique_ptr<llvm::Module> udf_gpu_module;
extern std::unique_ptr<llvm::Module> rt_udf_cpu_module;
extern std::unique_ptr<llvm::Module> rt_udf_gpu_module;
extern std::unique_ptr<llvm::Module> g_rt_geos_module;
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
#ifdef ENABLE_GEOS
    // load_geos_dynamic_library(); // not support now

    // Read geos runtime module and bind GEOS API function references to GEOS library
    auto rt_geos_module_copy = llvm::CloneModule(
        *g_rt_geos_module.get(), cgen_state_->vmap_, [](const llvm::GlobalValue* gv) {
          auto func = llvm::dyn_cast<llvm::Function>(gv);
          if (!func) {
            return true;
          }
          return (func->getLinkage() == llvm::GlobalValue::LinkageTypes::PrivateLinkage ||
                  func->getLinkage() ==
                      llvm::GlobalValue::LinkageTypes::InternalLinkage ||
                  func->getLinkage() == llvm::GlobalValue::LinkageTypes::ExternalLinkage);
        });
    CodeGenerator::link_udf_module(rt_geos_module_copy,
                                   *module,
                                   cgen_state_.get(),
                                   llvm::Linker::Flags::LinkOnlyNeeded);
#else
    throw std::runtime_error("GEOS is disabled in this build");
#endif
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

  initializeNVPTXBackend();  // todo
  CodeGenerator::GPUTarget gpu_target{
      nvptx_target_machine_.get(),  // todo
      cuda_mgr,
      cider_executor::blockSize(catalog_, block_size_x_),  // todo
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

bool g_enable_left_join_filter_hoisting{true};
static const int32_t ERR_INTERRUPTED{10};

std::string get_table_name(const InputDescriptor& input_desc,
                           const Catalog_Namespace::Catalog& cat) {
  const auto source_type = input_desc.getSourceType();
  if (source_type == InputSourceType::TABLE) {
    const auto td = cat.getMetadataForTable(input_desc.getTableId());
    CHECK(td);
    return td->tableName;
  } else {
    return "$TEMPORARY_TABLE" + std::to_string(-input_desc.getTableId());
  }
}

inline size_t getDeviceBasedScanLimit(const ExecutorDeviceType device_type,
                                      const int device_count) {
  if (device_type == ExecutorDeviceType::GPU) {
    return device_count * Executor::high_scan_limit;
  }
  return Executor::high_scan_limit;
}

void checkWorkUnitWatchdog(const RelAlgExecutionUnit& ra_exe_unit,
                           const std::vector<InputTableInfo>& table_infos,
                           const Catalog_Namespace::Catalog& cat,
                           const ExecutorDeviceType device_type,
                           const int device_count) {
  for (const auto target_expr : ra_exe_unit.target_exprs) {
    if (dynamic_cast<const Analyzer::AggExpr*>(target_expr)) {
      return;
    }
  }
  if (!ra_exe_unit.scan_limit && table_infos.size() == 1 &&
      table_infos.front().info.getPhysicalNumTuples() < Executor::high_scan_limit) {
    // Allow a query with no scan limit to run on small tables
    return;
  }
  if (ra_exe_unit.use_bump_allocator) {
    // Bump allocator removes the scan limit (and any knowledge of the size of the output
    // relative to the size of the input), so we bypass this check for now
    return;
  }
  if (ra_exe_unit.sort_info.algorithm != SortAlgorithm::StreamingTopN &&
      ra_exe_unit.groupby_exprs.size() == 1 && !ra_exe_unit.groupby_exprs.front() &&
      (!ra_exe_unit.scan_limit ||
       ra_exe_unit.scan_limit > getDeviceBasedScanLimit(device_type, device_count))) {
    std::vector<std::string> table_names;
    const auto& input_descs = ra_exe_unit.input_descs;
    for (const auto& input_desc : input_descs) {
      table_names.push_back(get_table_name(input_desc, cat));
    }
    if (!ra_exe_unit.scan_limit) {
      throw WatchdogException(
          "Projection query would require a scan without a limit on table(s): " +
          boost::algorithm::join(table_names, ", "));
    } else {
      throw WatchdogException(
          "Projection query output result set on table(s): " +
          boost::algorithm::join(table_names, ", ") + "  would contain " +
          std::to_string(ra_exe_unit.scan_limit) +
          " rows, which is more than the current system limit of " +
          std::to_string(getDeviceBasedScanLimit(device_type, device_count)));
    }
  }
}

bool has_lazy_fetched_columns(const std::vector<ColumnLazyFetchInfo>& fetched_cols) {
  for (const auto& col : fetched_cols) {
    if (col.is_lazily_fetched) {
      return true;
    }
  }
  return false;
}

std::vector<ColumnLazyFetchInfo> getColLazyFetchInfo(
    const std::vector<Analyzer::Expr*>& target_exprs,
    std::shared_ptr<PlanState> plan_state,
    Catalog_Namespace::Catalog* catalog) {
  CHECK(plan_state);
  CHECK(catalog);
  std::vector<ColumnLazyFetchInfo> col_lazy_fetch_info;
  for (const auto target_expr : target_exprs) {
    if (!plan_state->isLazyFetchColumn(target_expr)) {
      col_lazy_fetch_info.emplace_back(
          ColumnLazyFetchInfo{false, -1, SQLTypeInfo(kNULLT, false)});
    } else {
      const auto col_var = dynamic_cast<const Analyzer::ColumnVar*>(target_expr);
      CHECK(col_var);
      auto col_id = col_var->get_column_id();
      auto rte_idx = (col_var->get_rte_idx() == -1) ? 0 : col_var->get_rte_idx();
      auto cd = (col_var->get_table_id() > 0)
                    ? get_column_descriptor(col_id, col_var->get_table_id(), *catalog)
                    : nullptr;
      if (cd && IS_GEO(cd->columnType.get_type())) {
        // Geo coords cols will be processed in sequence. So we only need to track the
        // first coords col in lazy fetch info.
        {
          auto cd0 = get_column_descriptor(col_id + 1, col_var->get_table_id(), *catalog);
          auto col0_ti = cd0->columnType;
          CHECK(!cd0->isVirtualCol);
          auto col0_var = makeExpr<Analyzer::ColumnVar>(
              col0_ti, col_var->get_table_id(), cd0->columnId, rte_idx);
          auto local_col0_id = plan_state->getLocalColumnId(col0_var.get(), false);
          col_lazy_fetch_info.emplace_back(
              ColumnLazyFetchInfo{true, local_col0_id, col0_ti});
        }
      } else {
        auto local_col_id = plan_state->getLocalColumnId(col_var, false);
        const auto& col_ti = col_var->get_type_info();
        col_lazy_fetch_info.emplace_back(ColumnLazyFetchInfo{true, local_col_id, col_ti});
      }
    }
  }
  return col_lazy_fetch_info;
}

class FetchCacheAnchor {
 public:
  FetchCacheAnchor(CgenState* cgen_state)
      : cgen_state_(cgen_state), saved_fetch_cache(cgen_state_->fetch_cache_) {}
  ~FetchCacheAnchor() { cgen_state_->fetch_cache_.swap(saved_fetch_cache); }

 private:
  CgenState* cgen_state_;
  std::unordered_map<int, std::vector<llvm::Value*>> saved_fetch_cache;
};

struct JoinHashTableOrError {
  std::shared_ptr<HashJoin> hash_table;
  std::string fail_reason;
};

std::vector<llvm::Value*> inlineHoistedLiterals(std::shared_ptr<CgenState> cgen_state) {
  AUTOMATIC_IR_METADATA(cgen_state.get());

  std::vector<llvm::Value*> hoisted_literals;

  // row_func_ is using literals whose defs have been hoisted up to the query_func_,
  // extend row_func_ signature to include extra args to pass these literal values.
  std::vector<llvm::Type*> row_process_arg_types;

  for (llvm::Function::arg_iterator I = cgen_state->row_func_->arg_begin(),
                                    E = cgen_state->row_func_->arg_end();
       I != E;
       ++I) {
    row_process_arg_types.push_back(I->getType());
  }

  for (auto& element : cgen_state->query_func_literal_loads_) {
    for (auto value : element.second) {
      row_process_arg_types.push_back(value->getType());
    }
  }

  auto ft = llvm::FunctionType::get(
      get_int_type(32, cgen_state->context_), row_process_arg_types, false);
  auto row_func_with_hoisted_literals =
      llvm::Function::Create(ft,
                             llvm::Function::ExternalLinkage,
                             "row_func_hoisted_literals",
                             cgen_state->row_func_->getParent());

  auto row_func_arg_it = row_func_with_hoisted_literals->arg_begin();
  for (llvm::Function::arg_iterator I = cgen_state->row_func_->arg_begin(),
                                    E = cgen_state->row_func_->arg_end();
       I != E;
       ++I) {
    if (I->hasName()) {
      row_func_arg_it->setName(I->getName());
    }
    ++row_func_arg_it;
  }

  decltype(row_func_with_hoisted_literals) filter_func_with_hoisted_literals{nullptr};
  decltype(row_func_arg_it) filter_func_arg_it{nullptr};
  if (cgen_state->filter_func_) {
    // filter_func_ is using literals whose defs have been hoisted up to the row_func_,
    // extend filter_func_ signature to include extra args to pass these literal values.
    std::vector<llvm::Type*> filter_func_arg_types;

    for (llvm::Function::arg_iterator I = cgen_state->filter_func_->arg_begin(),
                                      E = cgen_state->filter_func_->arg_end();
         I != E;
         ++I) {
      filter_func_arg_types.push_back(I->getType());
    }

    for (auto& element : cgen_state->query_func_literal_loads_) {
      for (auto value : element.second) {
        filter_func_arg_types.push_back(value->getType());
      }
    }

    auto ft2 = llvm::FunctionType::get(
        get_int_type(32, cgen_state->context_), filter_func_arg_types, false);
    filter_func_with_hoisted_literals =
        llvm::Function::Create(ft2,
                               llvm::Function::ExternalLinkage,
                               "filter_func_hoisted_literals",
                               cgen_state->filter_func_->getParent());

    filter_func_arg_it = filter_func_with_hoisted_literals->arg_begin();
    for (llvm::Function::arg_iterator I = cgen_state->filter_func_->arg_begin(),
                                      E = cgen_state->filter_func_->arg_end();
         I != E;
         ++I) {
      if (I->hasName()) {
        filter_func_arg_it->setName(I->getName());
      }
      ++filter_func_arg_it;
    }
  }

  std::unordered_map<int, std::vector<llvm::Value*>>
      query_func_literal_loads_function_arguments,
      query_func_literal_loads_function_arguments2;

  for (auto& element : cgen_state->query_func_literal_loads_) {
    std::vector<llvm::Value*> argument_values, argument_values2;

    for (auto value : element.second) {
      hoisted_literals.push_back(value);
      argument_values.push_back(&*row_func_arg_it);
      if (cgen_state->filter_func_) {
        argument_values2.push_back(&*filter_func_arg_it);
        cgen_state->filter_func_args_[&*row_func_arg_it] = &*filter_func_arg_it;
      }
      if (value->hasName()) {
        row_func_arg_it->setName("arg_" + value->getName());
        if (cgen_state->filter_func_) {
          filter_func_arg_it->getContext();
          filter_func_arg_it->setName("arg_" + value->getName());
        }
      }
      ++row_func_arg_it;
      ++filter_func_arg_it;
    }

    query_func_literal_loads_function_arguments[element.first] = argument_values;
    query_func_literal_loads_function_arguments2[element.first] = argument_values2;
  }

  // copy the row_func function body over
  // see
  // https://stackoverflow.com/questions/12864106/move-function-body-avoiding-full-cloning/18751365
  row_func_with_hoisted_literals->getBasicBlockList().splice(
      row_func_with_hoisted_literals->begin(),
      cgen_state->row_func_->getBasicBlockList());

  // also replace row_func arguments with the arguments from row_func_hoisted_literals
  for (llvm::Function::arg_iterator I = cgen_state->row_func_->arg_begin(),
                                    E = cgen_state->row_func_->arg_end(),
                                    I2 = row_func_with_hoisted_literals->arg_begin();
       I != E;
       ++I) {
    I->replaceAllUsesWith(&*I2);
    I2->takeName(&*I);
    cgen_state->filter_func_args_.replace(&*I, &*I2);
    ++I2;
  }

  cgen_state->row_func_ = row_func_with_hoisted_literals;

  // and finally replace  literal placeholders
  std::vector<llvm::Instruction*> placeholders;
  std::string prefix("__placeholder__literal_");
  for (auto it = llvm::inst_begin(row_func_with_hoisted_literals),
            e = llvm::inst_end(row_func_with_hoisted_literals);
       it != e;
       ++it) {
    if (it->hasName() && it->getName().startswith(prefix)) {
      auto offset_and_index_entry =
          cgen_state->row_func_hoisted_literals_.find(llvm::dyn_cast<llvm::Value>(&*it));
      CHECK(offset_and_index_entry != cgen_state->row_func_hoisted_literals_.end());

      int lit_off = offset_and_index_entry->second.offset_in_literal_buffer;
      int lit_idx = offset_and_index_entry->second.index_of_literal_load;

      it->replaceAllUsesWith(
          query_func_literal_loads_function_arguments[lit_off][lit_idx]);
      placeholders.push_back(&*it);
    }
  }
  for (auto placeholder : placeholders) {
    placeholder->removeFromParent();
  }

  if (cgen_state->filter_func_) {
    // copy the filter_func function body over
    // see
    // https://stackoverflow.com/questions/12864106/move-function-body-avoiding-full-cloning/18751365
    filter_func_with_hoisted_literals->getBasicBlockList().splice(
        filter_func_with_hoisted_literals->begin(),
        cgen_state->filter_func_->getBasicBlockList());

    // also replace filter_func arguments with the arguments from
    // filter_func_hoisted_literals
    for (llvm::Function::arg_iterator I = cgen_state->filter_func_->arg_begin(),
                                      E = cgen_state->filter_func_->arg_end(),
                                      I2 = filter_func_with_hoisted_literals->arg_begin();
         I != E;
         ++I) {
      I->replaceAllUsesWith(&*I2);
      I2->takeName(&*I);
      ++I2;
    }

    cgen_state->filter_func_ = filter_func_with_hoisted_literals;

    // and finally replace  literal placeholders
    std::vector<llvm::Instruction*> placeholders;
    std::string prefix("__placeholder__literal_");
    for (auto it = llvm::inst_begin(filter_func_with_hoisted_literals),
              e = llvm::inst_end(filter_func_with_hoisted_literals);
         it != e;
         ++it) {
      if (it->hasName() && it->getName().startswith(prefix)) {
        auto offset_and_index_entry = cgen_state->row_func_hoisted_literals_.find(
            llvm::dyn_cast<llvm::Value>(&*it));
        CHECK(offset_and_index_entry != cgen_state->row_func_hoisted_literals_.end());

        int lit_off = offset_and_index_entry->second.offset_in_literal_buffer;
        int lit_idx = offset_and_index_entry->second.index_of_literal_load;

        it->replaceAllUsesWith(
            query_func_literal_loads_function_arguments2[lit_off][lit_idx]);
        placeholders.push_back(&*it);
      }
    }
    for (auto placeholder : placeholders) {
      placeholder->removeFromParent();
    }
  }

  return hoisted_literals;
}

void check_if_loop_join_is_allowed(RelAlgExecutionUnit& ra_exe_unit,
                                   const ExecutionOptions& eo,
                                   const std::vector<InputTableInfo>& query_infos,
                                   const size_t level_idx,
                                   const std::string& fail_reason) {
  if (eo.allow_loop_joins) {
    return;
  }
  if (level_idx + 1 != ra_exe_unit.join_quals.size()) {
    throw std::runtime_error(
        "Hash join failed, reason(s): " + fail_reason +
        " | Cannot fall back to loop join for intermediate join quals");
  }
  if (!is_trivial_loop_join(query_infos, ra_exe_unit)) {
    throw std::runtime_error(
        "Hash join failed, reason(s): " + fail_reason +
        " | Cannot fall back to loop join for non-trivial inner table size");
  }
}

class ExprTableIdVisitor : public ScalarExprVisitor<std::set<int>> {
 protected:
  std::set<int> visitColumnVar(const Analyzer::ColumnVar* col_expr) const final {
    return {col_expr->get_table_id()};
  }

  std::set<int> visitFunctionOper(const Analyzer::FunctionOper* func_expr) const final {
    std::set<int> ret;
    for (size_t i = 0; i < func_expr->getArity(); i++) {
      ret = aggregateResult(ret, visit(func_expr->getArg(i)));
    }
    return ret;
  }

  std::set<int> visitBinOper(const Analyzer::BinOper* bin_oper) const final {
    std::set<int> ret;
    ret = aggregateResult(ret, visit(bin_oper->get_left_operand()));
    return aggregateResult(ret, visit(bin_oper->get_right_operand()));
  }

  std::set<int> visitUOper(const Analyzer::UOper* u_oper) const final {
    return visit(u_oper->get_operand());
  }

  std::set<int> aggregateResult(const std::set<int>& aggregate,
                                const std::set<int>& next_result) const final {
    auto ret = aggregate;  // copy
    for (const auto& el : next_result) {
      ret.insert(el);
    }
    return ret;
  }
};

JoinLoop::HoistedFiltersCallback buildHoistLeftHandSideFiltersCb(
    const RelAlgExecutionUnit& ra_exe_unit,
    const size_t level_idx,
    const int inner_table_id,
    const CompilationOptions& co,
    std::shared_ptr<CgenState> cgen_state,
    std::shared_ptr<PlanState> plan_state,
    Executor* executor) {
  if (!g_enable_left_join_filter_hoisting) {
    return nullptr;
  }

  const auto& current_level_join_conditions = ra_exe_unit.join_quals[level_idx];
  if (current_level_join_conditions.type == JoinType::LEFT) {
    const auto& condition = current_level_join_conditions.quals.front();
    const auto bin_oper = dynamic_cast<const Analyzer::BinOper*>(condition.get());
    CHECK(bin_oper) << condition->toString();
    const auto rhs =
        dynamic_cast<const Analyzer::ColumnVar*>(bin_oper->get_right_operand());
    const auto lhs =
        dynamic_cast<const Analyzer::ColumnVar*>(bin_oper->get_left_operand());
    if (lhs && rhs && lhs->get_table_id() != rhs->get_table_id()) {
      const Analyzer::ColumnVar* selected_lhs{nullptr};
      // grab the left hand side column -- this is somewhat similar to normalize column
      // pair, and a better solution may be to hoist that function out of the join
      // framework and normalize columns at the top of build join loops
      if (lhs->get_table_id() == inner_table_id) {
        selected_lhs = rhs;
      } else if (rhs->get_table_id() == inner_table_id) {
        selected_lhs = lhs;
      }
      if (selected_lhs) {
        std::list<std::shared_ptr<Analyzer::Expr>> hoisted_quals;
        // get all LHS-only filters
        auto should_hoist_qual = [&hoisted_quals](const auto& qual, const int table_id) {
          CHECK(qual);

          ExprTableIdVisitor visitor;
          const auto table_ids = visitor.visit(qual.get());
          if (table_ids.size() == 1 && table_ids.find(table_id) != table_ids.end()) {
            hoisted_quals.push_back(qual);
          }
        };
        for (const auto& qual : ra_exe_unit.simple_quals) {
          should_hoist_qual(qual, selected_lhs->get_table_id());
        }
        for (const auto& qual : ra_exe_unit.quals) {
          should_hoist_qual(qual, selected_lhs->get_table_id());
        }

        // build the filters callback and return it
        if (!hoisted_quals.empty()) {
          return [executor, hoisted_quals, co, plan_state](
                     llvm::BasicBlock* true_bb,
                     llvm::BasicBlock* exit_bb,
                     const std::string& loop_name,
                     llvm::Function* parent_func,
                     CgenState* cgen_state) -> llvm::BasicBlock* {
            // make sure we have quals to hoist
            bool has_quals_to_hoist = false;
            for (const auto& qual : hoisted_quals) {
              // check to see if the filter was previously hoisted. if all filters were
              // previously hoisted, this callback becomes a noop
              if (plan_state->hoisted_filters_.count(qual) == 0) {
                has_quals_to_hoist = true;
                break;
              }
            }

            if (!has_quals_to_hoist) {
              return nullptr;
            }

            AUTOMATIC_IR_METADATA(cgen_state);

            llvm::IRBuilder<>& builder = cgen_state->ir_builder_;
            auto& context = builder.getContext();

            const auto filter_bb =
                llvm::BasicBlock::Create(context,
                                         "hoisted_left_join_filters_" + loop_name,
                                         parent_func,
                                         /*insert_before=*/true_bb);
            builder.SetInsertPoint(filter_bb);

            llvm::Value* filter_lv = cgen_state->llBool(true);
            CodeGenerator code_generator(executor);
            CHECK(plan_state);
            for (const auto& qual : hoisted_quals) {
              if (plan_state->hoisted_filters_.insert(qual).second) {
                // qual was inserted into the hoisted filters map, which means we have not
                // seen this qual before. Generate filter.
                VLOG(1) << "Generating code for hoisted left hand side qualifier "
                        << qual->toString();
                auto cond = code_generator.toBool(
                    code_generator.codegen(qual.get(), true, co).front());
                filter_lv = builder.CreateAnd(filter_lv, cond);
              }
            }
            CHECK(filter_lv->getType()->isIntegerTy(1));

            builder.CreateCondBr(filter_lv, true_bb, exit_bb);
            return filter_bb;
          };
        }
      }
    }
  }
  return nullptr;
}

void insertErrorCodeChecker(llvm::Function* query_func,
                            bool hoist_literals,
                            bool allow_runtime_query_interrupt,
                            std::shared_ptr<CgenState> cgen_state) {
  auto query_stub_func_name =
      "query_stub" + std::string(hoist_literals ? "_hoisted_literals" : "");
  for (auto bb_it = query_func->begin(); bb_it != query_func->end(); ++bb_it) {
    for (auto inst_it = bb_it->begin(); inst_it != bb_it->end(); ++inst_it) {
      if (!llvm::isa<llvm::CallInst>(*inst_it)) {
        continue;
      }
      auto& row_func_call = llvm::cast<llvm::CallInst>(*inst_it);
      if (std::string(row_func_call.getCalledFunction()->getName()) ==
          query_stub_func_name) {
        auto next_inst_it = inst_it;
        ++next_inst_it;
        auto new_bb = bb_it->splitBasicBlock(next_inst_it);
        auto& br_instr = bb_it->back();
        llvm::IRBuilder<> ir_builder(&br_instr);
        llvm::Value* err_lv = &*inst_it;
        auto error_check_bb =
            bb_it->splitBasicBlock(llvm::BasicBlock::iterator(br_instr), ".error_check");
        llvm::Value* error_code_arg = nullptr;
        auto arg_cnt = 0;
        for (auto arg_it = query_func->arg_begin(); arg_it != query_func->arg_end();
             arg_it++, ++arg_cnt) {
          // since multi_frag_* func has anonymous arguments so we use arg_offset
          // explicitly to capture "error_code" argument in the func's argument list
          if (hoist_literals) {
            if (arg_cnt == 9) {
              error_code_arg = &*arg_it;
              break;
            }
          } else {
            if (arg_cnt == 8) {
              error_code_arg = &*arg_it;
              break;
            }
          }
        }
        CHECK(error_code_arg);
        llvm::Value* err_code = nullptr;
        if (allow_runtime_query_interrupt) {
          // decide the final error code with a consideration of interrupt status
          auto& check_interrupt_br_instr = bb_it->back();
          auto interrupt_check_bb = llvm::BasicBlock::Create(
              cgen_state->context_, ".interrupt_check", query_func, error_check_bb);
          llvm::IRBuilder<> interrupt_checker_ir_builder(interrupt_check_bb);
          auto detected_interrupt = interrupt_checker_ir_builder.CreateCall(
              cgen_state->module_->getFunction("check_interrupt"), {});
          auto detected_error = interrupt_checker_ir_builder.CreateCall(
              cgen_state->module_->getFunction("get_error_code"),
              std::vector<llvm::Value*>{error_code_arg});
          err_code = interrupt_checker_ir_builder.CreateSelect(
              detected_interrupt,
              cgen_state->llInt(Executor::ERR_INTERRUPTED),
              detected_error);
          interrupt_checker_ir_builder.CreateBr(error_check_bb);
          llvm::ReplaceInstWithInst(&check_interrupt_br_instr,
                                    llvm::BranchInst::Create(interrupt_check_bb));
          ir_builder.SetInsertPoint(&br_instr);
        } else {
          // uses error code returned from row_func and skip to check interrupt status
          ir_builder.SetInsertPoint(&br_instr);
          err_code =
              ir_builder.CreateCall(cgen_state->module_->getFunction("get_error_code"),
                                    std::vector<llvm::Value*>{error_code_arg});
        }
        err_lv = ir_builder.CreateICmp(
            llvm::ICmpInst::ICMP_NE, err_code, cgen_state->llInt(0));
        auto error_bb = llvm::BasicBlock::Create(
            cgen_state->context_, ".error_exit", query_func, new_bb);
        llvm::CallInst::Create(cgen_state->module_->getFunction("record_error_code"),
                               std::vector<llvm::Value*>{err_code, error_code_arg},
                               "",
                               error_bb);
        llvm::ReturnInst::Create(cgen_state->context_, error_bb);
        llvm::ReplaceInstWithInst(&br_instr,
                                  llvm::BranchInst::Create(error_bb, new_bb, err_lv));
        break;
      }
    }
  }
}

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
};

llvm::BasicBlock* codegenSkipDeletedOuterTableRow(const RelAlgExecutionUnit& ra_exe_unit,
                                                  const CompilationOptions& co,
                                                  std::shared_ptr<CgenState> cgen_state,
                                                  std::shared_ptr<PlanState> plan_state,
                                                  Executor* executor) {
  AUTOMATIC_IR_METADATA(cgen_state.get());
  if (!co.filter_on_deleted_column) {
    return nullptr;
  }
  CHECK(!ra_exe_unit.input_descs.empty());
  const auto& outer_input_desc = ra_exe_unit.input_descs[0];
  if (outer_input_desc.getSourceType() != InputSourceType::TABLE) {
    return nullptr;
  }
  const auto deleted_cd =
      plan_state->getDeletedColForTable(outer_input_desc.getTableId());
  if (!deleted_cd) {
    return nullptr;
  }
  CHECK(deleted_cd->columnType.is_boolean());
  const auto deleted_expr =
      makeExpr<Analyzer::ColumnVar>(deleted_cd->columnType,
                                    outer_input_desc.getTableId(),
                                    deleted_cd->columnId,
                                    outer_input_desc.getNestLevel());
  CodeGenerator code_generator(executor);
  const auto is_deleted =
      code_generator.toBool(code_generator.codegen(deleted_expr.get(), true, co).front());
  const auto is_deleted_bb =
      llvm::BasicBlock::Create(cgen_state->context_, "is_deleted", cgen_state->row_func_);
  llvm::BasicBlock* bb = llvm::BasicBlock::Create(
      cgen_state->context_, "is_not_deleted", cgen_state->row_func_);
  cgen_state->ir_builder_.CreateCondBr(is_deleted, is_deleted_bb, bb);
  cgen_state->ir_builder_.SetInsertPoint(is_deleted_bb);
  cgen_state->ir_builder_.CreateRet(cgen_state->llInt<int32_t>(0));
  cgen_state->ir_builder_.SetInsertPoint(bb);
  return bb;
};

// searches for a particular variable within a specific basic block (or all if bb_name is
// empty)
template <typename InstType>
llvm::Value* find_variable_in_basic_block(llvm::Function* func,
                                          std::string bb_name,
                                          std::string variable_name) {
  llvm::Value* result = nullptr;
  if (func == nullptr || variable_name.empty()) {
    return result;
  }
  bool is_found = false;
  for (auto bb_it = func->begin(); bb_it != func->end() && !is_found; ++bb_it) {
    if (!bb_name.empty() && bb_it->getName() != bb_name) {
      continue;
    }
    for (auto inst_it = bb_it->begin(); inst_it != bb_it->end(); inst_it++) {
      if (llvm::isa<InstType>(*inst_it)) {
        if (inst_it->getName() == variable_name) {
          result = &*inst_it;
          is_found = true;
          break;
        }
      }
    }
  }
  return result;
}

unsigned blockSize(Catalog_Namespace::Catalog* catalog, unsigned block_size_x) {
  CHECK(catalog);
  const auto cuda_mgr = catalog->getDataMgr().getCudaMgr();
  if (!cuda_mgr) {
    return 0;
  }
  const auto& dev_props = cuda_mgr->getAllDeviceProperties();
  return block_size_x ? block_size_x : dev_props.front().maxThreadsPerBlock;
}

unsigned gridSize(Catalog_Namespace::Catalog* catalog, unsigned grid_size_x) {
  CHECK(catalog);
  const auto cuda_mgr = catalog->getDataMgr().getCudaMgr();
  if (!cuda_mgr) {
    return 0;
  }
  return grid_size_x ? grid_size_x : 2 * cuda_mgr->getMinNumMPsForAllDevices();
}

void createErrorCheckControlFlow(llvm::Function* query_func,
                                 bool run_with_dynamic_watchdog,
                                 bool run_with_allowing_runtime_interrupt,
                                 ExecutorDeviceType device_type,
                                 const std::vector<InputTableInfo>& input_table_infos,
                                 std::shared_ptr<CgenState> cgen_state,
                                 Catalog_Namespace::Catalog* catalog,
                                 const unsigned block_size_x = 0,
                                 const unsigned grid_size_x = 0) {
  AUTOMATIC_IR_METADATA(cgen_state.get());

  // check whether the row processing was successful; currently, it can
  // fail by running out of group by buffer slots

  if (run_with_dynamic_watchdog && run_with_allowing_runtime_interrupt) {
    // when both dynamic watchdog and runtime interrupt turns on
    // we use dynamic watchdog
    run_with_allowing_runtime_interrupt = false;
  }

  // TODO: remove this part, will use it???
  // {
  //   // disable injecting query interrupt checker if the session info is invalid
  //   mapd_shared_lock<mapd_shared_mutex> session_read_lock(executor_session_mutex_);
  //   if (current_query_session_.empty()) {
  //     run_with_allowing_runtime_interrupt = false;
  //   }
  // }

  llvm::Value* row_count = nullptr;
  if ((run_with_dynamic_watchdog || run_with_allowing_runtime_interrupt) &&
      device_type == ExecutorDeviceType::GPU) {
    row_count = cider_executor::find_variable_in_basic_block<llvm::LoadInst>(
        query_func, ".entry", "row_count");
  }

  bool done_splitting = false;
  for (auto bb_it = query_func->begin(); bb_it != query_func->end() && !done_splitting;
       ++bb_it) {
    llvm::Value* pos = nullptr;
    for (auto inst_it = bb_it->begin(); inst_it != bb_it->end(); ++inst_it) {
      if ((run_with_dynamic_watchdog || run_with_allowing_runtime_interrupt) &&
          llvm::isa<llvm::PHINode>(*inst_it)) {
        if (inst_it->getName() == "pos") {
          pos = &*inst_it;
        }
        continue;
      }
      if (!llvm::isa<llvm::CallInst>(*inst_it)) {
        continue;
      }
      auto& row_func_call = llvm::cast<llvm::CallInst>(*inst_it);
      if (std::string(row_func_call.getCalledFunction()->getName()) == "row_process") {
        auto next_inst_it = inst_it;
        ++next_inst_it;
        auto new_bb = bb_it->splitBasicBlock(next_inst_it);
        auto& br_instr = bb_it->back();
        llvm::IRBuilder<> ir_builder(&br_instr);
        llvm::Value* err_lv = &*inst_it;
        llvm::Value* err_lv_returned_from_row_func = nullptr;
        if (run_with_dynamic_watchdog) {
          CHECK(pos);
          llvm::Value* call_watchdog_lv = nullptr;
          if (device_type == ExecutorDeviceType::GPU) {
            // In order to make sure all threads within a block see the same barrier,
            // only those blocks whose none of their threads have experienced the critical
            // edge will go through the dynamic watchdog computation
            CHECK(row_count);
            // llvm::Value* crit_edge_rem = nullptr;
            auto crit_edge_rem =
                (cider_executor::blockSize(catalog, block_size_x) &
                 (cider_executor::blockSize(catalog, block_size_x) - 1))
                    ? ir_builder.CreateSRem(
                          row_count,
                          cgen_state->llInt(static_cast<int64_t>(
                              cider_executor::blockSize(catalog, block_size_x))))
                    : ir_builder.CreateAnd(
                          row_count,
                          cgen_state->llInt(static_cast<int64_t>(
                              cider_executor::blockSize(catalog, block_size_x) - 1)));
            auto crit_edge_threshold = ir_builder.CreateSub(row_count, crit_edge_rem);
            crit_edge_threshold->setName("crit_edge_threshold");

            // only those threads where pos < crit_edge_threshold go through dynamic
            // watchdog call
            call_watchdog_lv =
                ir_builder.CreateICmp(llvm::ICmpInst::ICMP_SLT, pos, crit_edge_threshold);
          } else {
            // CPU path: run watchdog for every 64th row
            auto dw_predicate = ir_builder.CreateAnd(pos, uint64_t(0x3f));
            call_watchdog_lv = ir_builder.CreateICmp(
                llvm::ICmpInst::ICMP_EQ, dw_predicate, cgen_state->llInt(int64_t(0LL)));
          }
          CHECK(call_watchdog_lv);
          auto error_check_bb = bb_it->splitBasicBlock(
              llvm::BasicBlock::iterator(br_instr), ".error_check");
          auto& watchdog_br_instr = bb_it->back();

          auto watchdog_check_bb = llvm::BasicBlock::Create(
              cgen_state->context_, ".watchdog_check", query_func, error_check_bb);
          llvm::IRBuilder<> watchdog_ir_builder(watchdog_check_bb);
          auto detected_timeout = watchdog_ir_builder.CreateCall(
              cgen_state->module_->getFunction("dynamic_watchdog"), {});
          auto timeout_err_lv = watchdog_ir_builder.CreateSelect(
              detected_timeout, cgen_state->llInt(Executor::ERR_OUT_OF_TIME), err_lv);
          watchdog_ir_builder.CreateBr(error_check_bb);

          llvm::ReplaceInstWithInst(
              &watchdog_br_instr,
              llvm::BranchInst::Create(
                  watchdog_check_bb, error_check_bb, call_watchdog_lv));
          ir_builder.SetInsertPoint(&br_instr);
          auto unified_err_lv = ir_builder.CreatePHI(err_lv->getType(), 2);

          unified_err_lv->addIncoming(timeout_err_lv, watchdog_check_bb);
          unified_err_lv->addIncoming(err_lv, &*bb_it);
          err_lv = unified_err_lv;
        } else if (run_with_allowing_runtime_interrupt) {
          CHECK(pos);
          llvm::Value* call_check_interrupt_lv = nullptr;
          if (device_type == ExecutorDeviceType::GPU) {
            // approximate how many times the %pos variable
            // is increased --> the number of iteration
            // here we calculate the # bit shift by considering grid/block/fragment sizes
            // since if we use the fixed one (i.e., per 64-th increment)
            // some CUDA threads cannot enter the interrupt checking block depending on
            // the fragment size --> a thread may not take care of 64 threads if an outer
            // table is not sufficiently large, and so cannot be interrupted
            int32_t num_shift_by_gridDim =
                shared::getExpOfTwo(cider_executor::gridSize(catalog, grid_size_x));
            int32_t num_shift_by_blockDim =
                shared::getExpOfTwo(cider_executor::blockSize(catalog, block_size_x));
            int total_num_shift = num_shift_by_gridDim + num_shift_by_blockDim;
            uint64_t interrupt_checking_freq = 32;
            auto freq_control_knob = g_running_query_interrupt_freq;
            CHECK_GT(freq_control_knob, 0);
            CHECK_LE(freq_control_knob, 1.0);
            if (!input_table_infos.empty()) {
              const auto& outer_table_info = *input_table_infos.begin();
              auto num_outer_table_tuples = outer_table_info.info.getNumTuples();
              if (outer_table_info.table_id < 0) {
                auto* rs = (*outer_table_info.info.fragments.begin()).resultSet;
                CHECK(rs);
                num_outer_table_tuples = rs->entryCount();
              } else {
                auto num_frags = outer_table_info.info.fragments.size();
                if (num_frags > 0) {
                  num_outer_table_tuples =
                      outer_table_info.info.fragments.begin()->getNumTuples();
                }
              }
              if (num_outer_table_tuples > 0) {
                // gridSize * blockSize --> pos_step (idx of the next row per thread)
                // we additionally multiply two to pos_step since the number of
                // dispatched blocks are double of the gridSize
                // # tuples (of fragment) / pos_step --> maximum # increment (K)
                // also we multiply 1 / freq_control_knob to K to control the frequency
                // So, needs to check the interrupt status more frequently? make K smaller
                // TODO: refactor gridSize/blockSize
                auto max_inc = uint64_t(
                    floor(num_outer_table_tuples /
                          (gridSize(catalog, grid_size_x) *
                           cider_executor::blockSize(catalog, block_size_x) * 2)));
                if (max_inc < 2) {
                  // too small `max_inc`, so this correction is necessary to make
                  // `interrupt_checking_freq` be valid (i.e., larger than zero)
                  max_inc = 2;
                }
                auto calibrated_inc = uint64_t(floor(max_inc * (1 - freq_control_knob)));
                interrupt_checking_freq =
                    uint64_t(pow(2, shared::getExpOfTwo(calibrated_inc)));
                // add the coverage when interrupt_checking_freq > K
                // if so, some threads still cannot be branched to the interrupt checker
                // so we manually use smaller but close to the max_inc as freq
                if (interrupt_checking_freq > max_inc) {
                  interrupt_checking_freq = max_inc / 2;
                }
                if (interrupt_checking_freq < 8) {
                  // such small freq incurs too frequent interrupt status checking,
                  // so we fixup to the minimum freq value at some reasonable degree
                  interrupt_checking_freq = 8;
                }
              }
            }
            VLOG(1) << "Set the running query interrupt checking frequency: "
                    << interrupt_checking_freq;
            // check the interrupt flag for every interrupt_checking_freq-th iteration
            llvm::Value* pos_shifted_per_iteration =
                ir_builder.CreateLShr(pos, cgen_state->llInt(total_num_shift));
            auto interrupt_predicate =
                ir_builder.CreateAnd(pos_shifted_per_iteration, interrupt_checking_freq);
            call_check_interrupt_lv =
                ir_builder.CreateICmp(llvm::ICmpInst::ICMP_EQ,
                                      interrupt_predicate,
                                      cgen_state->llInt(int64_t(0LL)));
          } else {
            // CPU path: run interrupt checker for every 64th row
            auto interrupt_predicate = ir_builder.CreateAnd(pos, uint64_t(0x3f));
            call_check_interrupt_lv =
                ir_builder.CreateICmp(llvm::ICmpInst::ICMP_EQ,
                                      interrupt_predicate,
                                      cgen_state->llInt(int64_t(0LL)));
          }
          CHECK(call_check_interrupt_lv);
          auto error_check_bb = bb_it->splitBasicBlock(
              llvm::BasicBlock::iterator(br_instr), ".error_check");
          auto& check_interrupt_br_instr = bb_it->back();

          auto interrupt_check_bb = llvm::BasicBlock::Create(
              cgen_state->context_, ".interrupt_check", query_func, error_check_bb);
          llvm::IRBuilder<> interrupt_checker_ir_builder(interrupt_check_bb);
          auto detected_interrupt = interrupt_checker_ir_builder.CreateCall(
              cgen_state->module_->getFunction("check_interrupt"), {});
          auto interrupt_err_lv = interrupt_checker_ir_builder.CreateSelect(
              detected_interrupt, cgen_state->llInt(Executor::ERR_INTERRUPTED), err_lv);
          interrupt_checker_ir_builder.CreateBr(error_check_bb);

          llvm::ReplaceInstWithInst(
              &check_interrupt_br_instr,
              llvm::BranchInst::Create(
                  interrupt_check_bb, error_check_bb, call_check_interrupt_lv));
          ir_builder.SetInsertPoint(&br_instr);
          auto unified_err_lv = ir_builder.CreatePHI(err_lv->getType(), 2);

          unified_err_lv->addIncoming(interrupt_err_lv, interrupt_check_bb);
          unified_err_lv->addIncoming(err_lv, &*bb_it);
          err_lv = unified_err_lv;
        }
        if (!err_lv_returned_from_row_func) {
          err_lv_returned_from_row_func = err_lv;
        }
        if (device_type == ExecutorDeviceType::GPU && g_enable_dynamic_watchdog) {
          // let kernel execution finish as expected, regardless of the observed error,
          // unless it is from the dynamic watchdog where all threads within that block
          // return together.
          err_lv = ir_builder.CreateICmp(llvm::ICmpInst::ICMP_EQ,
                                         err_lv,
                                         cgen_state->llInt(Executor::ERR_OUT_OF_TIME));
        } else {
          err_lv = ir_builder.CreateICmp(llvm::ICmpInst::ICMP_NE,
                                         err_lv,
                                         cgen_state->llInt(static_cast<int32_t>(0)));
        }
        auto error_bb = llvm::BasicBlock::Create(
            cgen_state->context_, ".error_exit", query_func, new_bb);
        const auto error_code_arg = get_arg_by_name(query_func, "error_code");
        llvm::CallInst::Create(
            cgen_state->module_->getFunction("record_error_code"),
            std::vector<llvm::Value*>{err_lv_returned_from_row_func, error_code_arg},
            "",
            error_bb);
        llvm::ReturnInst::Create(cgen_state->context_, error_bb);
        llvm::ReplaceInstWithInst(&br_instr,
                                  llvm::BranchInst::Create(error_bb, new_bb, err_lv));
        done_splitting = true;
        break;
      }
    }
  }
  CHECK(done_splitting);
};

llvm::Value* addJoinLoopIterator(const std::vector<llvm::Value*>& prev_iters,
                                 const size_t level_idx,
                                 std::shared_ptr<CgenState> cgen_state) {
  AUTOMATIC_IR_METADATA(cgen_state.get());
  // Iterators are added for loop-outer joins when the head of the loop is generated,
  // then once again when the body if generated. Allow this instead of special handling
  // of call sites.
  const auto it = cgen_state->scan_idx_to_hash_pos_.find(level_idx);
  if (it != cgen_state->scan_idx_to_hash_pos_.end()) {
    return it->second;
  }
  CHECK(!prev_iters.empty());
  llvm::Value* matching_row_index = prev_iters.back();
  const auto it_ok =
      cgen_state->scan_idx_to_hash_pos_.emplace(level_idx, matching_row_index);
  CHECK(it_ok.second);
  return matching_row_index;
}

void redeclareFilterFunction(std::shared_ptr<CgenState> cgen_state) {
  if (!cgen_state->filter_func_) {
    return;
  }

  // Loop over all the instructions used in the filter func.
  // The filter func instructions were generated as if for row func.
  // Remap any values used by those instructions to filter func args
  // and remember to forward them through the call in the row func.
  for (auto bb_it = cgen_state->filter_func_->begin();
       bb_it != cgen_state->filter_func_->end();
       ++bb_it) {
    for (auto instr_it = bb_it->begin(); instr_it != bb_it->end(); ++instr_it) {
      size_t i = 0;
      for (auto op_it = instr_it->value_op_begin(); op_it != instr_it->value_op_end();
           ++op_it, ++i) {
        llvm::Value* v = *op_it;

        // The last LLVM operand on a call instruction is the function to be called. Never
        // remap it.
        if (llvm::dyn_cast<const llvm::CallInst>(instr_it) &&
            op_it == instr_it->value_op_end() - 1) {
          continue;
        }

        if (auto* instr = llvm::dyn_cast<llvm::Instruction>(v);
            instr && instr->getParent() &&
            instr->getParent()->getParent() == cgen_state->row_func_) {
          // Remember that this filter func arg is needed.
          cgen_state->filter_func_args_[v] = nullptr;
        } else if (auto* argum = llvm::dyn_cast<llvm::Argument>(v);
                   argum && argum->getParent() == cgen_state->row_func_) {
          // Remember that this filter func arg is needed.
          cgen_state->filter_func_args_[v] = nullptr;
        }
      }
    }
  }

  // Create filter_func2 with parameters only for those row func values that are known to
  // be used in the filter func code.
  std::vector<llvm::Type*> filter_func_arg_types;
  filter_func_arg_types.reserve(cgen_state->filter_func_args_.v_.size());
  for (auto& arg : cgen_state->filter_func_args_.v_) {
    filter_func_arg_types.push_back(arg->getType());
  }
  auto ft = llvm::FunctionType::get(
      get_int_type(32, cgen_state->context_), filter_func_arg_types, false);
  cgen_state->filter_func_->setName("old_filter_func");
  auto filter_func2 = llvm::Function::Create(ft,
                                             llvm::Function::ExternalLinkage,
                                             "filter_func",
                                             cgen_state->filter_func_->getParent());
  CHECK_EQ(filter_func2->arg_size(), cgen_state->filter_func_args_.v_.size());
  auto arg_it = cgen_state->filter_func_args_.begin();
  size_t i = 0;
  for (llvm::Function::arg_iterator I = filter_func2->arg_begin(),
                                    E = filter_func2->arg_end();
       I != E;
       ++I, ++arg_it) {
    arg_it->second = &*I;
    if (arg_it->first->hasName()) {
      I->setName(arg_it->first->getName());
    } else {
      I->setName("extra" + std::to_string(i++));
    }
  }

  // copy the filter_func function body over
  // see
  // https://stackoverflow.com/questions/12864106/move-function-body-avoiding-full-cloning/18751365
  filter_func2->getBasicBlockList().splice(filter_func2->begin(),
                                           cgen_state->filter_func_->getBasicBlockList());

  if (cgen_state->current_func_ == cgen_state->filter_func_) {
    cgen_state->current_func_ = filter_func2;
  }
  cgen_state->filter_func_ = filter_func2;

  // loop over all the operands in the filter func
  for (auto bb_it = cgen_state->filter_func_->begin();
       bb_it != cgen_state->filter_func_->end();
       ++bb_it) {
    for (auto instr_it = bb_it->begin(); instr_it != bb_it->end(); ++instr_it) {
      size_t i = 0;
      for (auto op_it = instr_it->op_begin(); op_it != instr_it->op_end(); ++op_it, ++i) {
        llvm::Value* v = op_it->get();
        if (auto arg_it = cgen_state->filter_func_args_.find(v);
            arg_it != cgen_state->filter_func_args_.end()) {
          // replace row func value with a filter func arg
          llvm::Use* use = &*op_it;
          use->set(arg_it->second);
        }
      }
    }
  }
}

bool compileBody(const RelAlgExecutionUnit& ra_exe_unit,
                 GroupByAndAggregate& group_by_and_aggregate,
                 const QueryMemoryDescriptor& query_mem_desc,
                 const CompilationOptions& co,
                 const GpuSharedMemoryContext& gpu_smem_context,
                 std::shared_ptr<CgenState> cgen_state,
                 std::shared_ptr<PlanState> plan_state,
                 Executor* executor) {
  AUTOMATIC_IR_METADATA(cgen_state.get());

  // Switch the code generation into a separate filter function if enabled.
  // Note that accesses to function arguments are still codegenned from the
  // row function's arguments, then later automatically forwarded and
  // remapped into filter function arguments by redeclareFilterFunction().
  cgen_state->row_func_bb_ = cgen_state->ir_builder_.GetInsertBlock();
  llvm::Value* loop_done{nullptr};
  // remove this member since I didn't see anyone use it.
  // std::unique_ptr<Executor::FetchCacheAnchor> fetch_cache_anchor;
  if (cgen_state->filter_func_) {
    if (cgen_state->row_func_bb_->getName() == "loop_body") {
      auto row_func_entry_bb = &cgen_state->row_func_->getEntryBlock();
      cgen_state->ir_builder_.SetInsertPoint(row_func_entry_bb,
                                             row_func_entry_bb->begin());
      loop_done = cgen_state->ir_builder_.CreateAlloca(
          get_int_type(1, cgen_state->context_), nullptr, "loop_done");
      cgen_state->ir_builder_.SetInsertPoint(cgen_state->row_func_bb_);
      cgen_state->ir_builder_.CreateStore(cgen_state->llBool(true), loop_done);
    }
    cgen_state->ir_builder_.SetInsertPoint(cgen_state->filter_func_bb_);
    cgen_state->current_func_ = cgen_state->filter_func_;
    // fetch_cache_anchor =
    // std::make_unique<Executor::FetchCacheAnchor>(cgen_state.get());
  }

  // generate the code for the filter
  std::vector<Analyzer::Expr*> primary_quals;
  std::vector<Analyzer::Expr*> deferred_quals;
  bool short_circuited = CodeGenerator::prioritizeQuals(
      ra_exe_unit, primary_quals, deferred_quals, plan_state->hoisted_filters_);
  if (short_circuited) {
    VLOG(1) << "Prioritized " << std::to_string(primary_quals.size()) << " quals, "
            << "short-circuited and deferred " << std::to_string(deferred_quals.size())
            << " quals";
  }
  llvm::Value* filter_lv = cgen_state->llBool(true);
  CodeGenerator code_generator(executor);
  for (auto expr : primary_quals) {
    // Generate the filter for primary quals
    auto cond = code_generator.toBool(code_generator.codegen(expr, true, co).front());
    filter_lv = cgen_state->ir_builder_.CreateAnd(filter_lv, cond);
  }
  CHECK(filter_lv->getType()->isIntegerTy(1));
  llvm::BasicBlock* sc_false{nullptr};
  if (!deferred_quals.empty()) {
    auto sc_true = llvm::BasicBlock::Create(
        cgen_state->context_, "sc_true", cgen_state->current_func_);
    sc_false = llvm::BasicBlock::Create(
        cgen_state->context_, "sc_false", cgen_state->current_func_);
    cgen_state->ir_builder_.CreateCondBr(filter_lv, sc_true, sc_false);
    cgen_state->ir_builder_.SetInsertPoint(sc_false);
    if (ra_exe_unit.join_quals.empty()) {
      cgen_state->ir_builder_.CreateRet(cgen_state->llInt(int32_t(0)));
    }
    cgen_state->ir_builder_.SetInsertPoint(sc_true);
    filter_lv = cgen_state->llBool(true);
  }
  for (auto expr : deferred_quals) {
    filter_lv = cgen_state->ir_builder_.CreateAnd(
        filter_lv, code_generator.toBool(code_generator.codegen(expr, true, co).front()));
  }

  CHECK(filter_lv->getType()->isIntegerTy(1));
  auto ret = group_by_and_aggregate.codegen(
      filter_lv, sc_false, query_mem_desc, co, gpu_smem_context);

  // Switch the code generation back to the row function if a filter
  // function was enabled.
  if (cgen_state->filter_func_) {
    if (cgen_state->row_func_bb_->getName() == "loop_body") {
      cgen_state->ir_builder_.CreateStore(cgen_state->llBool(false), loop_done);
      cgen_state->ir_builder_.CreateRet(cgen_state->llInt<int32_t>(0));
    }

    cgen_state->ir_builder_.SetInsertPoint(cgen_state->row_func_bb_);
    cgen_state->current_func_ = cgen_state->row_func_;
    cgen_state->filter_func_call_ =
        cgen_state->ir_builder_.CreateCall(cgen_state->filter_func_, {});

    // Create real filter function declaration after placeholder call
    // is emitted.
    cider_executor::redeclareFilterFunction(cgen_state);

    if (cgen_state->row_func_bb_->getName() == "loop_body") {
      auto loop_done_true = llvm::BasicBlock::Create(
          cgen_state->context_, "loop_done_true", cgen_state->row_func_);
      auto loop_done_false = llvm::BasicBlock::Create(
          cgen_state->context_, "loop_done_false", cgen_state->row_func_);
      auto loop_done_flag = cgen_state->ir_builder_.CreateLoad(loop_done);
      cgen_state->ir_builder_.CreateCondBr(
          loop_done_flag, loop_done_true, loop_done_false);
      cgen_state->ir_builder_.SetInsertPoint(loop_done_true);
      cgen_state->ir_builder_.CreateRet(cgen_state->filter_func_call_);
      cgen_state->ir_builder_.SetInsertPoint(loop_done_false);
    } else {
      cgen_state->ir_builder_.CreateRet(cgen_state->filter_func_call_);
    }
  }
  return ret;
};

unsigned numBlocksPerMP(Catalog_Namespace::Catalog* catalog, unsigned grid_size_x) {
  CHECK(catalog);
  const auto cuda_mgr = catalog->getDataMgr().getCudaMgr();
  CHECK(cuda_mgr);
  return grid_size_x ? std::ceil(grid_size_x / cuda_mgr->getMinNumMPsForAllDevices()) : 2;
}

void preloadFragOffsets(const std::vector<InputDescriptor>& input_descs,
                        const std::vector<InputTableInfo>& query_infos,
                        std::shared_ptr<CgenState> cgen_state) {
  AUTOMATIC_IR_METADATA(cgen_state.get());
  const auto ld_count = input_descs.size();
  auto frag_off_ptr = get_arg_by_name(cgen_state->row_func_, "frag_row_off");
  for (size_t i = 0; i < ld_count; ++i) {
    CHECK_LT(i, query_infos.size());
    const auto frag_count = query_infos[i].info.fragments.size();
    if (i > 0) {
      cgen_state->frag_offsets_.push_back(nullptr);
    } else {
      if (frag_count > 1) {
        cgen_state->frag_offsets_.push_back(
            cgen_state->ir_builder_.CreateLoad(frag_off_ptr));
      } else {
        cgen_state->frag_offsets_.push_back(nullptr);
      }
    }
  }
}

inline llvm::Value* get_arg_by_name(llvm::Function* func, const std::string& name) {
  for (auto& arg : func->args()) {
    if (arg.getName() == name) {
      return &arg;
    }
  }
  CHECK(false);
  return nullptr;
}

void add_qualifier_to_execution_unit(RelAlgExecutionUnit& ra_exe_unit,
                                     const std::shared_ptr<Analyzer::Expr>& qual) {
  const auto qual_cf = qual_to_conjunctive_form(qual);
  ra_exe_unit.simple_quals.insert(ra_exe_unit.simple_quals.end(),
                                  qual_cf.simple_quals.begin(),
                                  qual_cf.simple_quals.end());
  ra_exe_unit.quals.insert(
      ra_exe_unit.quals.end(), qual_cf.quals.begin(), qual_cf.quals.end());
}

void check_valid_join_qual(std::shared_ptr<Analyzer::BinOper>& bin_oper) {
  // check whether a join qual is valid before entering the hashtable build and codegen

  auto lhs_cv = dynamic_cast<const Analyzer::ColumnVar*>(bin_oper->get_left_operand());
  auto rhs_cv = dynamic_cast<const Analyzer::ColumnVar*>(bin_oper->get_right_operand());
  if (lhs_cv && rhs_cv && !bin_oper->is_overlaps_oper()) {
    auto lhs_type = lhs_cv->get_type_info().get_type();
    auto rhs_type = rhs_cv->get_type_info().get_type();
    // check #1. avoid a join btw full array columns
    if (lhs_type == SQLTypes::kARRAY && rhs_type == SQLTypes::kARRAY) {
      throw std::runtime_error(
          "Join operation between full array columns (i.e., R.arr = S.arr) instead of "
          "indexed array columns (i.e., R.arr[1] = S.arr[2]) is not supported yet.");
    }
  }
}

int deviceCount(Catalog_Namespace::Catalog* catalog,
                const ExecutorDeviceType device_type) {
  if (device_type == ExecutorDeviceType::GPU) {
    const auto cuda_mgr = catalog->getDataMgr().getCudaMgr();
    CHECK(cuda_mgr);
    return cuda_mgr->getDeviceCount();
  } else {
    return 1;
  }
}

int deviceCountForMemoryLevel(Catalog_Namespace::Catalog* catalog,
                              const Data_Namespace::MemoryLevel memory_level) {
  return memory_level == GPU_LEVEL ? deviceCount(catalog, ExecutorDeviceType::GPU)
                                   : deviceCount(catalog, ExecutorDeviceType::CPU);
}

JoinHashTableOrError buildHashTableForQualifier(
    const std::shared_ptr<Analyzer::BinOper>& qual_bin_oper,
    const std::vector<InputTableInfo>& query_infos,
    const MemoryLevel memory_level,
    const HashType preferred_hash_type,
    ColumnCacheMap& column_cache,
    const RegisteredQueryHint& query_hint,
    Executor* executor,
    Catalog_Namespace::Catalog* catalog) {
  if (!g_enable_overlaps_hashjoin && qual_bin_oper->is_overlaps_oper()) {
    return {nullptr, "Overlaps hash join disabled, attempting to fall back to loop join"};
  }
  if (g_enable_dynamic_watchdog) {
    throw QueryExecutionError(ERR_INTERRUPTED);
  }
  try {
    auto tbl = HashJoin::getInstance(qual_bin_oper,
                                     query_infos,
                                     memory_level,
                                     preferred_hash_type,
                                     deviceCountForMemoryLevel(catalog, memory_level),
                                     column_cache,
                                     executor,
                                     query_hint);
    return {tbl, ""};
  } catch (const HashJoinFail& e) {
    return {nullptr, e.what()};
  }
}

std::shared_ptr<HashJoin> buildCurrentLevelHashTable(
    const JoinCondition& current_level_join_conditions,
    RelAlgExecutionUnit& ra_exe_unit,
    const CompilationOptions& co,
    const std::vector<InputTableInfo>& query_infos,
    ColumnCacheMap& column_cache,
    std::vector<std::string>& fail_reasons,
    std::shared_ptr<CgenState> cgen_state,
    std::shared_ptr<PlanState> plan_state,
    Executor* executor,
    Catalog_Namespace::Catalog* catalog) {
  AUTOMATIC_IR_METADATA(cgen_state.get());
  if (current_level_join_conditions.type != JoinType::INNER &&
      current_level_join_conditions.quals.size() > 1) {
    fail_reasons.emplace_back("No equijoin expression found for outer join");
    return nullptr;
  }
  std::shared_ptr<HashJoin> current_level_hash_table;
  for (const auto& join_qual : current_level_join_conditions.quals) {
    auto qual_bin_oper = std::dynamic_pointer_cast<Analyzer::BinOper>(join_qual);
    if (!qual_bin_oper || !IS_EQUIVALENCE(qual_bin_oper->get_optype())) {
      fail_reasons.emplace_back("No equijoin expression found");
      if (current_level_join_conditions.type == JoinType::INNER) {
        add_qualifier_to_execution_unit(ra_exe_unit, join_qual);
      }
      continue;
    }
    check_valid_join_qual(qual_bin_oper);
    JoinHashTableOrError hash_table_or_error;
    if (!current_level_hash_table) {
      hash_table_or_error = buildHashTableForQualifier(
          qual_bin_oper,
          query_infos,
          co.device_type == ExecutorDeviceType::GPU ? MemoryLevel::GPU_LEVEL
                                                    : MemoryLevel::CPU_LEVEL,
          HashType::OneToOne,
          column_cache,
          ra_exe_unit.query_hint,
          executor,
          catalog);
      current_level_hash_table = hash_table_or_error.hash_table;
    }
    if (hash_table_or_error.hash_table) {
      plan_state->join_info_.join_hash_tables_.push_back(hash_table_or_error.hash_table);
      plan_state->join_info_.equi_join_tautologies_.push_back(qual_bin_oper);
    } else {
      fail_reasons.push_back(hash_table_or_error.fail_reason);
      if (current_level_join_conditions.type == JoinType::INNER) {
        add_qualifier_to_execution_unit(ra_exe_unit, qual_bin_oper);
      }
    }
  }
  return current_level_hash_table;
}

std::function<llvm::Value*(const std::vector<llvm::Value*>&, llvm::Value*)>
buildIsDeletedCb(const RelAlgExecutionUnit& ra_exe_unit,
                 const size_t level_idx,
                 const CompilationOptions& co,
                 std::shared_ptr<CgenState> cgen_state,
                 std::shared_ptr<PlanState> plan_state,
                 Executor* executor) {
  AUTOMATIC_IR_METADATA(cgen_state.get());
  if (!co.filter_on_deleted_column) {
    return nullptr;
  }
  CHECK_LT(level_idx + 1, ra_exe_unit.input_descs.size());
  const auto input_desc = ra_exe_unit.input_descs[level_idx + 1];
  if (input_desc.getSourceType() != InputSourceType::TABLE) {
    return nullptr;
  }

  const auto deleted_cd = plan_state->getDeletedColForTable(input_desc.getTableId());
  if (!deleted_cd) {
    return nullptr;
  }
  CHECK(deleted_cd->columnType.is_boolean());
  const auto deleted_expr = makeExpr<Analyzer::ColumnVar>(deleted_cd->columnType,
                                                          input_desc.getTableId(),
                                                          deleted_cd->columnId,
                                                          input_desc.getNestLevel());
  return [executor, deleted_expr, level_idx, &co, cgen_state](
             const std::vector<llvm::Value*>& prev_iters,
             llvm::Value* have_more_inner_rows) {
    const auto matching_row_index =
        addJoinLoopIterator(prev_iters, level_idx + 1, cgen_state);
    // Avoid fetching the deleted column from a position which is not valid.
    // An invalid position can be returned by a one to one hash lookup (negative)
    // or at the end of iteration over a set of matching values.
    llvm::Value* is_valid_it{nullptr};
    if (have_more_inner_rows) {
      is_valid_it = have_more_inner_rows;
    } else {
      is_valid_it = cgen_state->ir_builder_.CreateICmp(
          llvm::ICmpInst::ICMP_SGE, matching_row_index, cgen_state->llInt<int64_t>(0));
    }
    const auto it_valid_bb = llvm::BasicBlock::Create(
        cgen_state->context_, "it_valid", cgen_state->current_func_);
    const auto it_not_valid_bb = llvm::BasicBlock::Create(
        cgen_state->context_, "it_not_valid", cgen_state->current_func_);
    cgen_state->ir_builder_.CreateCondBr(is_valid_it, it_valid_bb, it_not_valid_bb);
    const auto row_is_deleted_bb = llvm::BasicBlock::Create(
        cgen_state->context_, "row_is_deleted", cgen_state->current_func_);
    cgen_state->ir_builder_.SetInsertPoint(it_valid_bb);
    CodeGenerator code_generator(executor);
    const auto row_is_deleted = code_generator.toBool(
        code_generator.codegen(deleted_expr.get(), true, co).front());
    cgen_state->ir_builder_.CreateBr(row_is_deleted_bb);
    cgen_state->ir_builder_.SetInsertPoint(it_not_valid_bb);
    const auto row_is_deleted_default = cgen_state->llBool(false);
    cgen_state->ir_builder_.CreateBr(row_is_deleted_bb);
    cgen_state->ir_builder_.SetInsertPoint(row_is_deleted_bb);
    auto row_is_deleted_or_default =
        cgen_state->ir_builder_.CreatePHI(row_is_deleted->getType(), 2);
    row_is_deleted_or_default->addIncoming(row_is_deleted, it_valid_bb);
    row_is_deleted_or_default->addIncoming(row_is_deleted_default, it_not_valid_bb);
    return row_is_deleted_or_default;
  };
}

std::vector<JoinLoop> buildJoinLoops(RelAlgExecutionUnit& ra_exe_unit,
                                     const CompilationOptions& co,
                                     const ExecutionOptions& eo,
                                     const std::vector<InputTableInfo>& query_infos,
                                     ColumnCacheMap& column_cache,
                                     Executor* executor,
                                     std::shared_ptr<CgenState> cgen_state,
                                     std::shared_ptr<PlanState> plan_state,
                                     Catalog_Namespace::Catalog* catalog) {
  INJECT_TIMER(buildJoinLoops);
  AUTOMATIC_IR_METADATA(cgen_state.get());
  std::vector<JoinLoop> join_loops;
  for (size_t level_idx = 0, current_hash_table_idx = 0;
       level_idx < ra_exe_unit.join_quals.size();
       ++level_idx) {
    const auto& current_level_join_conditions = ra_exe_unit.join_quals[level_idx];
    std::vector<std::string> fail_reasons;
    const auto build_cur_level_hash_table = [&]() {
      if (current_level_join_conditions.quals.size() > 1) {
        const auto first_qual = *current_level_join_conditions.quals.begin();
        auto qual_bin_oper =
            std::dynamic_pointer_cast<const Analyzer::BinOper>(first_qual);
        if (qual_bin_oper && qual_bin_oper->is_overlaps_oper() &&
            current_level_join_conditions.type == JoinType::LEFT) {
          JoinCondition join_condition{{first_qual}, current_level_join_conditions.type};

          return buildCurrentLevelHashTable(join_condition,
                                            ra_exe_unit,
                                            co,
                                            query_infos,
                                            column_cache,
                                            fail_reasons,
                                            cgen_state,
                                            plan_state,
                                            executor,
                                            catalog);
        }
      }
      return buildCurrentLevelHashTable(current_level_join_conditions,
                                        ra_exe_unit,
                                        co,
                                        query_infos,
                                        column_cache,
                                        fail_reasons,
                                        cgen_state,
                                        plan_state,
                                        executor,
                                        catalog);
    };
    const auto current_level_hash_table = build_cur_level_hash_table();
    const auto found_outer_join_matches_cb = [executor, level_idx, cgen_state](
                                                 llvm::Value* found_outer_join_matches) {
      CHECK_LT(level_idx, cgen_state->outer_join_match_found_per_level_.size());
      CHECK(!cgen_state->outer_join_match_found_per_level_[level_idx]);
      cgen_state->outer_join_match_found_per_level_[level_idx] = found_outer_join_matches;
    };
    const auto is_deleted_cb =
        buildIsDeletedCb(ra_exe_unit, level_idx, co, cgen_state, plan_state, executor);
    const auto outer_join_condition_multi_quals_cb =
        [executor, level_idx, &co, &current_level_join_conditions, cgen_state](
            const std::vector<llvm::Value*>& prev_iters) {
          // The values generated for the match path don't dominate all uses
          // since on the non-match path nulls are generated. Reset the cache
          // once the condition is generated to avoid incorrect reuse.
          FetchCacheAnchor anchor(cgen_state.get());
          addJoinLoopIterator(prev_iters, level_idx + 1, cgen_state);
          llvm::Value* left_join_cond = cgen_state->llBool(true);
          CodeGenerator code_generator(executor);
          // Do not want to look at all quals! only 1..N quals (ignore first qual)
          // Note(jclay): this may need to support cases larger than 2
          // are there any?
          if (current_level_join_conditions.quals.size() >= 2) {
            auto qual_it = std::next(current_level_join_conditions.quals.begin(), 1);
            for (; qual_it != current_level_join_conditions.quals.end();
                 std::advance(qual_it, 1)) {
              left_join_cond = cgen_state->ir_builder_.CreateAnd(
                  left_join_cond,
                  code_generator.toBool(
                      code_generator.codegen(qual_it->get(), true, co).front()));
            }
          }
          return left_join_cond;
        };
    if (current_level_hash_table) {
      const auto hoisted_filters_cb =
          buildHoistLeftHandSideFiltersCb(ra_exe_unit,
                                          level_idx,
                                          current_level_hash_table->getInnerTableId(),
                                          co,
                                          cgen_state,
                                          plan_state,
                                          executor);
      if (current_level_hash_table->getHashType() == HashType::OneToOne) {
        join_loops.emplace_back(
            /*kind=*/JoinLoopKind::Singleton,
            /*type=*/current_level_join_conditions.type,
            /*iteration_domain_codegen=*/
            [executor,
             current_hash_table_idx,
             level_idx,
             current_level_hash_table,
             &co,
             cgen_state](const std::vector<llvm::Value*>& prev_iters) {
              addJoinLoopIterator(prev_iters, level_idx, cgen_state);
              JoinLoopDomain domain{{0}};
              domain.slot_lookup_result =
                  current_level_hash_table->codegenSlot(co, current_hash_table_idx);
              return domain;
            },
            /*outer_condition_match=*/nullptr,
            /*found_outer_matches=*/current_level_join_conditions.type == JoinType::LEFT
                ? std::function<void(llvm::Value*)>(found_outer_join_matches_cb)
                : nullptr,
            /*hoisted_filters=*/hoisted_filters_cb,
            /*is_deleted=*/is_deleted_cb);
      } else {
        join_loops.emplace_back(
            /*kind=*/JoinLoopKind::Set,
            /*type=*/current_level_join_conditions.type,
            /*iteration_domain_codegen=*/
            [executor,
             current_hash_table_idx,
             level_idx,
             current_level_hash_table,
             &co,
             cgen_state](const std::vector<llvm::Value*>& prev_iters) {
              addJoinLoopIterator(prev_iters, level_idx, cgen_state);
              JoinLoopDomain domain{{0}};
              const auto matching_set = current_level_hash_table->codegenMatchingSet(
                  co, current_hash_table_idx);
              domain.values_buffer = matching_set.elements;
              domain.element_count = matching_set.count;
              return domain;
            },
            /*outer_condition_match=*/
            current_level_join_conditions.type == JoinType::LEFT
                ? std::function<llvm::Value*(const std::vector<llvm::Value*>&)>(
                      outer_join_condition_multi_quals_cb)
                : nullptr,
            /*found_outer_matches=*/current_level_join_conditions.type == JoinType::LEFT
                ? std::function<void(llvm::Value*)>(found_outer_join_matches_cb)
                : nullptr,
            /*hoisted_filters=*/hoisted_filters_cb,
            /*is_deleted=*/is_deleted_cb);
      }
      ++current_hash_table_idx;
    } else {
      const auto fail_reasons_str = current_level_join_conditions.quals.empty()
                                        ? "No equijoin expression found"
                                        : boost::algorithm::join(fail_reasons, " | ");
      check_if_loop_join_is_allowed(
          ra_exe_unit, eo, query_infos, level_idx, fail_reasons_str);
      // Callback provided to the `JoinLoop` framework to evaluate the (outer) join
      // condition.
      VLOG(1) << "Unable to build hash table, falling back to loop join: "
              << fail_reasons_str;
      const auto outer_join_condition_cb =
          [executor, level_idx, &co, &current_level_join_conditions, cgen_state](
              const std::vector<llvm::Value*>& prev_iters) {
            // The values generated for the match path don't dominate all uses
            // since on the non-match path nulls are generated. Reset the cache
            // once the condition is generated to avoid incorrect reuse.
            FetchCacheAnchor anchor(cgen_state.get());
            addJoinLoopIterator(prev_iters, level_idx + 1, cgen_state);
            llvm::Value* left_join_cond = cgen_state->llBool(true);
            CodeGenerator code_generator(executor);
            for (auto expr : current_level_join_conditions.quals) {
              left_join_cond = cgen_state->ir_builder_.CreateAnd(
                  left_join_cond,
                  code_generator.toBool(
                      code_generator.codegen(expr.get(), true, co).front()));
            }
            return left_join_cond;
          };
      join_loops.emplace_back(
          /*kind=*/JoinLoopKind::UpperBound,
          /*type=*/current_level_join_conditions.type,
          /*iteration_domain_codegen=*/
          [executor, level_idx, cgen_state](const std::vector<llvm::Value*>& prev_iters) {
            addJoinLoopIterator(prev_iters, level_idx, cgen_state);
            JoinLoopDomain domain{{0}};
            const auto rows_per_scan_ptr = cgen_state->ir_builder_.CreateGEP(
                get_arg_by_name(cgen_state->row_func_, "num_rows_per_scan"),
                cgen_state->llInt(int32_t(level_idx + 1)));
            domain.upper_bound = cgen_state->ir_builder_.CreateLoad(rows_per_scan_ptr,
                                                                    "num_rows_per_scan");
            return domain;
          },
          /*outer_condition_match=*/
          current_level_join_conditions.type == JoinType::LEFT
              ? std::function<llvm::Value*(const std::vector<llvm::Value*>&)>(
                    outer_join_condition_cb)
              : nullptr,
          /*found_outer_matches=*/
          current_level_join_conditions.type == JoinType::LEFT
              ? std::function<void(llvm::Value*)>(found_outer_join_matches_cb)
              : nullptr,
          /*hoisted_filters=*/nullptr,
          /*is_deleted=*/is_deleted_cb);
    }
  }
  return join_loops;
}

}  // namespace cider_executor

// didn't make it in cider_executor namespace because we need call GroupByAndAggregate
// private member, so make it a static method .
void CiderCodeGenerator::codegenJoinLoops(const std::vector<JoinLoop>& join_loops,
                                          const RelAlgExecutionUnit& ra_exe_unit,
                                          GroupByAndAggregate& group_by_and_aggregate,
                                          llvm::Function* query_func,
                                          llvm::BasicBlock* entry_bb,
                                          const QueryMemoryDescriptor& query_mem_desc,
                                          const CompilationOptions& co,
                                          const ExecutionOptions& eo,
                                          std::shared_ptr<CgenState> cgen_state,
                                          std::shared_ptr<PlanState> plan_state,
                                          Executor* executor,
                                          Catalog_Namespace::Catalog* catalog,
                                          const unsigned block_size_x = 0,
                                          const unsigned grid_size_x = 0) {
  AUTOMATIC_IR_METADATA(cgen_state.get());
  const auto exit_bb =
      llvm::BasicBlock::Create(cgen_state->context_, "exit", cgen_state->current_func_);
  cgen_state->ir_builder_.SetInsertPoint(exit_bb);
  cgen_state->ir_builder_.CreateRet(cgen_state->llInt<int32_t>(0));
  cgen_state->ir_builder_.SetInsertPoint(entry_bb);
  CodeGenerator code_generator(executor);
  const auto loops_entry_bb = JoinLoop::codegen(
      join_loops,
      /*body_codegen=*/
      [executor,
       query_func,
       &query_mem_desc,
       &co,
       &eo,
       &group_by_and_aggregate,
       &join_loops,
       &ra_exe_unit,
       &cgen_state,
       &plan_state,
       &catalog,
       &block_size_x,
       &grid_size_x](const std::vector<llvm::Value*>& prev_iters) {
        AUTOMATIC_IR_METADATA(cgen_state.get());
        cider_executor::addJoinLoopIterator(prev_iters, join_loops.size(), cgen_state);
        auto& builder = cgen_state->ir_builder_;
        const auto loop_body_bb = llvm::BasicBlock::Create(
            builder.getContext(), "loop_body", builder.GetInsertBlock()->getParent());
        builder.SetInsertPoint(loop_body_bb);
        const bool can_return_error = cider_executor::compileBody(ra_exe_unit,
                                                                  group_by_and_aggregate,
                                                                  query_mem_desc,
                                                                  co,
                                                                  {},
                                                                  cgen_state,
                                                                  plan_state,
                                                                  executor);
        if (can_return_error || cgen_state->needs_error_check_ ||
            eo.with_dynamic_watchdog || eo.allow_runtime_query_interrupt) {
          cider_executor::createErrorCheckControlFlow(query_func,
                                                      eo.with_dynamic_watchdog,
                                                      eo.allow_runtime_query_interrupt,
                                                      co.device_type,
                                                      group_by_and_aggregate.query_infos_,
                                                      cgen_state,
                                                      catalog,
                                                      block_size_x,
                                                      grid_size_x);
        }
        return loop_body_bb;
      },
      /*outer_iter=*/code_generator.posArg(nullptr),
      exit_bb,
      cgen_state.get());
  cgen_state->ir_builder_.SetInsertPoint(entry_bb);
  cgen_state->ir_builder_.CreateBr(loops_entry_bb);
};

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
  const bool gpu_shared_mem_optimization = cider::is_gpu_shared_mem_supported(
      query_mem_desc.get(),
      ra_exe_unit,
      cuda_mgr,
      co.device_type,
      cuda_mgr ? cider_executor::blockSize(catalog_, block_size_x_) : 1,
      cuda_mgr ? cider_executor::numBlocksPerMP(catalog_, grid_size_x_) : 1);
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

  cider_executor::preloadFragOffsets(ra_exe_unit.input_descs, query_infos, cgen_state_);
  RelAlgExecutionUnit body_execution_unit = ra_exe_unit;
  const auto join_loops = cider_executor::buildJoinLoops(body_execution_unit,
                                                         co,
                                                         eo,
                                                         query_infos,
                                                         column_cache,
                                                         executor_,
                                                         cgen_state_,
                                                         plan_state_,
                                                         catalog_);

  plan_state_->allocateLocalColumnIds(ra_exe_unit.input_col_descs);
  // todo: remove executor
  const auto is_not_deleted_bb = cider_executor::codegenSkipDeletedOuterTableRow(
      ra_exe_unit, co, cgen_state_, plan_state_, executor_);
  if (is_not_deleted_bb) {
    cgen_state_->row_func_bb_ = is_not_deleted_bb;
  }
  if (!join_loops.empty()) {
    // todo: remove executor
    CiderCodeGenerator::codegenJoinLoops(join_loops,
                                         body_execution_unit,
                                         group_by_and_aggregate,
                                         query_func,
                                         cgen_state_->row_func_bb_,
                                         *(query_mem_desc.get()),
                                         co,
                                         eo,
                                         cgen_state_,
                                         plan_state_,
                                         executor_,
                                         catalog_,
                                         block_size_x_,
                                         grid_size_x_);
  } else {
    // todo: remove executor
    const bool can_return_error = cider_executor::compileBody(ra_exe_unit,
                                                              group_by_and_aggregate,
                                                              *query_mem_desc,
                                                              co,
                                                              gpu_smem_context,
                                                              cgen_state_,
                                                              plan_state_,
                                                              executor_);
    if (can_return_error || cgen_state_->needs_error_check_ || eo.with_dynamic_watchdog ||
        eo.allow_runtime_query_interrupt) {
      // todo: remove executor

      cider_executor::createErrorCheckControlFlow(query_func,
                                                  eo.with_dynamic_watchdog,
                                                  eo.allow_runtime_query_interrupt,
                                                  co.device_type,
                                                  group_by_and_aggregate.query_infos_,
                                                  cgen_state_,
                                                  catalog_,
                                                  block_size_x_,
                                                  grid_size_x_);
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
    hoisted_literals = cider_executor::inlineHoistedLiterals(cgen_state_);
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
    cider_executor::insertErrorCodeChecker(multifrag_query_func,
                                           co.hoist_literals,
                                           eo.allow_runtime_query_interrupt,
                                           cgen_state_);
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
    llvm_ir += cider::serialize_llvm_metadata_footnotes(query_func, cgen_state_.get());
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

// std::vector<std::unique_ptr<ExecutionKernel>>
// CiderCodeGenerator::createKernelsWithQueryMemoryDescriptor(
//    std::shared_ptr<QueryMemoryDescriptor> query_mem_desc_owned,
//    size_t& max_groups_buffer_entry_guess,
//    const bool is_agg,
//    const bool allow_single_frag_table_opt,
//    const std::vector<InputTableInfo>& query_infos,
//    const RelAlgExecutionUnit& ra_exe_unit_in,
//    const CompilationOptions& co,
//    const ExecutionOptions& eo,
//    const Catalog_Namespace::Catalog& cat,
//    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
//    RenderInfo* render_info,
//    const bool has_cardinality_estimation,
//    ColumnCacheMap& column_cache) {
//  INJECT_TIMER(Exec_executeWorkUnit);
//  const auto [ra_exe_unit, deleted_cols_map] = addDeletedColumn(ra_exe_unit_in, co);
//  const auto device_type = getDeviceTypeForTargets(ra_exe_unit, co.device_type);
//  CHECK(!query_infos.empty());
//  if (!max_groups_buffer_entry_guess) {
//    // The query has failed the first execution attempt because of running out
//    // of group by slots. Make the conservative choice: allocate fragment size
//    // slots and run on the CPU.
//    CHECK(device_type == ExecutorDeviceType::CPU);
//    max_groups_buffer_entry_guess = compute_buffer_entry_guess(query_infos);
//  }
//
//  int8_t crt_min_byte_width{MAX_BYTE_WIDTH_SUPPORTED};
//  do {
//    SharedKernelContext shared_context(query_infos);
//    ColumnFetcher column_fetcher(this, column_cache);
//    auto query_comp_desc_owned = std::make_unique<QueryCompilationDescriptor>();
//    std::unique_ptr<QueryMemoryDescriptor> query_mem_desc_owned;
//    if (eo.executor_type == ExecutorType::Native) {
//      try {
//        INJECT_TIMER(query_step_compilation);
//        auto clock_begin = timer_start();
//        std::lock_guard<std::mutex> compilation_lock(compilation_mutex_);
//        compilation_queue_time_ms_ += timer_stop(clock_begin);
//
//        query_mem_desc_owned =
//            query_comp_desc_owned->compile(max_groups_buffer_entry_guess,
//                                           crt_min_byte_width,
//                                           has_cardinality_estimation,
//                                           ra_exe_unit,
//                                           query_infos,
//                                           deleted_cols_map,
//                                           column_fetcher,
//                                           {device_type,
//                                            co.hoist_literals,
//                                            co.opt_level,
//                                            co.with_dynamic_watchdog,
//                                            co.allow_lazy_fetch,
//                                            co.filter_on_deleted_column,
//                                            co.explain_type,
//                                            co.register_intel_jit_listener},
//                                           eo,
//                                           render_info,
//                                           this);
//        CHECK(query_mem_desc_owned);
//        crt_min_byte_width = query_comp_desc_owned->getMinByteWidth();
//      } catch (CompilationRetryNoCompaction&) {
//        crt_min_byte_width = MAX_BYTE_WIDTH_SUPPORTED;
//        continue;
//      }
//    } else {
//      plan_state_.reset(new PlanState(false, query_infos, deleted_cols_map, this));
//      plan_state_->allocateLocalColumnIds(ra_exe_unit.input_col_descs);
//      CHECK(!query_mem_desc_owned);
//      query_mem_desc_owned.reset(
//          new QueryMemoryDescriptor(this, 0, QueryDescriptionType::Projection, false));
//    }
//    if (eo.just_explain) {
//      return executeExplain(*query_comp_desc_owned);
//    }
//
//    for (const auto target_expr : ra_exe_unit.target_exprs) {
//      plan_state_->target_exprs_.push_back(target_expr);
//    }
//
//    if (!eo.just_validate) {
//      int available_cpus = cpu_threads();
//      auto available_gpus = get_available_gpus(cat);
//
//      const auto context_count =
//          get_context_count(device_type, available_cpus, available_gpus.size());
//      try {
//        return createKernels(shared_context,
//                                     ra_exe_unit,
//                                     column_fetcher,
//                                     query_infos,
//                                     eo,
//                                     is_agg,
//                                     allow_single_frag_table_opt,
//                                     context_count,
//                                     *query_comp_desc_owned,
//                                     *query_mem_desc_owned,
//                                     render_info,
//                                     available_gpus,
//                                     available_cpus);
//      } catch (QueryExecutionError& e) {
//        if (eo.with_dynamic_watchdog && interrupted_.load() &&
//            e.getErrorCode() == ERR_OUT_OF_TIME) {
//          throw QueryExecutionError(ERR_INTERRUPTED);
//        }
//        if (e.getErrorCode() == ERR_INTERRUPTED) {
//          throw QueryExecutionError(ERR_INTERRUPTED);
//        }
//        if (e.getErrorCode() == ERR_OVERFLOW_OR_UNDERFLOW &&
//            static_cast<size_t>(crt_min_byte_width << 1) <= sizeof(int64_t)) {
//          crt_min_byte_width <<= 1;
//          continue;
//        }
//        throw;
//      }
//    }
//    if (is_agg) {
//      if (eo.allow_runtime_query_interrupt && ra_exe_unit.query_state) {
//        // update query status to let user know we are now in the reduction phase
//        std::string curRunningSession{""};
//        std::string curRunningQuerySubmittedTime{""};
//        bool sessionEnrolled = false;
//        auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
//        {
//          mapd_shared_lock<mapd_shared_mutex> session_read_lock(
//              executor->getSessionLock());
//          curRunningSession = executor->getCurrentQuerySession(session_read_lock);
//          curRunningQuerySubmittedTime =
//          ra_exe_unit.query_state->getQuerySubmittedTime(); sessionEnrolled =
//              executor->checkIsQuerySessionEnrolled(curRunningSession,
//              session_read_lock);
//        }
//        if (!curRunningSession.empty() && !curRunningQuerySubmittedTime.empty() &&
//            sessionEnrolled) {
//          executor->updateQuerySessionStatus(curRunningSession,
//                                             curRunningQuerySubmittedTime,
//                                             QuerySessionStatus::RUNNING_REDUCTION);
//        }
//      }
//      try {
//        return collectAllDeviceResults(shared_context,
//                                       ra_exe_unit,
//                                       *query_mem_desc_owned,
//                                       query_comp_desc_owned->getDeviceType(),
//                                       row_set_mem_owner);
//      } catch (ReductionRanOutOfSlots&) {
//        throw QueryExecutionError(ERR_OUT_OF_SLOTS);
//      } catch (OverflowOrUnderflow&) {
//        crt_min_byte_width <<= 1;
//        continue;
//      } catch (QueryExecutionError& e) {
//        VLOG(1) << "Error received! error_code: " << e.getErrorCode()
//                << ", what(): " << e.what();
//        throw QueryExecutionError(e.getErrorCode());
//      }
//    }
//    return resultsUnion(shared_context, ra_exe_unit);
//
//  } while (static_cast<size_t>(crt_min_byte_width) <= sizeof(int64_t));
//
//  return std::make_shared<ResultSet>(std::vector<TargetInfo>{},
//                                     ExecutorDeviceType::CPU,
//                                     QueryMemoryDescriptor(),
//                                     nullptr,
//                                     catalog_,
//                                     blockSize(),
//                                     gridSize());
//}

std::vector<std::unique_ptr<ExecutionKernel>> CiderCodeGenerator::createKernels(
    SharedKernelContext& shared_context,
    const RelAlgExecutionUnit& ra_exe_unit,
    ColumnFetcher& column_fetcher,
    const std::vector<InputTableInfo>& table_infos,
    const ExecutionOptions& eo,
    const bool is_agg,
    const bool allow_single_frag_table_opt,
    const size_t context_count,
    const QueryCompilationDescriptor& query_comp_desc,
    const QueryMemoryDescriptor& query_mem_desc,
    RenderInfo* render_info,
    std::unordered_set<int>& available_gpus,
    int& available_cpus,
    std::shared_ptr<PlanState> plan_state,
    Catalog_Namespace::Catalog* catalog,
    Executor* executor) {
  std::vector<std::unique_ptr<ExecutionKernel>> execution_kernels;

  QueryFragmentDescriptor fragment_descriptor(
      ra_exe_unit,
      table_infos,
      query_comp_desc.getDeviceType() == ExecutorDeviceType::GPU
          ? catalog->getDataMgr().getMemoryInfo(Data_Namespace::MemoryLevel::GPU_LEVEL)
          : std::vector<Data_Namespace::MemoryInfo>{},
      eo.gpu_input_mem_limit_percent,
      eo.outer_fragment_indices);
  CHECK(!ra_exe_unit.input_descs.empty());

  const auto device_type = query_comp_desc.getDeviceType();
  const bool uses_lazy_fetch =
      plan_state_->allow_lazy_fetch_ &&
      cider_executor::has_lazy_fetched_columns(cider_executor::getColLazyFetchInfo(
          ra_exe_unit.target_exprs, plan_state, catalog));
  const bool use_multifrag_kernel = (device_type == ExecutorDeviceType::GPU) &&
                                    eo.allow_multifrag && (!uses_lazy_fetch || is_agg);
  const auto device_count = cider_executor::deviceCount(catalog, device_type);
  CHECK_GT(device_count, 0);

  fragment_descriptor.buildFragmentKernelMap(ra_exe_unit,
                                             shared_context.getFragOffsets(),
                                             device_count,
                                             device_type,
                                             use_multifrag_kernel,
                                             g_inner_join_fragment_skipping,
                                             executor);
  if (eo.with_watchdog && fragment_descriptor.shouldCheckWorkUnitWatchdog()) {
    cider_executor::checkWorkUnitWatchdog(
        ra_exe_unit, table_infos, *catalog, device_type, device_count);
  }

  if (use_multifrag_kernel) {
    VLOG(1) << "Creating multifrag execution kernels";
    VLOG(1) << query_mem_desc.toString();

    // NB: We should never be on this path when the query is retried because of running
    // out of group by slots; also, for scan only queries on CPU we want the
    // high-granularity, fragment by fragment execution instead. For scan only queries on
    // GPU, we want the multifrag kernel path to save the overhead of allocating an output
    // buffer per fragment.
    auto multifrag_kernel_dispatch = [&ra_exe_unit,
                                      &execution_kernels,
                                      &column_fetcher,
                                      &eo,
                                      &query_comp_desc,
                                      &query_mem_desc,
                                      render_info](const int device_id,
                                                   const FragmentsList& frag_list,
                                                   const int64_t rowid_lookup_key) {
      execution_kernels.emplace_back(
          std::make_unique<ExecutionKernel>(ra_exe_unit,
                                            ExecutorDeviceType::GPU,
                                            device_id,
                                            eo,
                                            column_fetcher,
                                            query_comp_desc,
                                            query_mem_desc,
                                            frag_list,
                                            ExecutorDispatchMode::MultifragmentKernel,
                                            render_info,
                                            rowid_lookup_key));
    };
    fragment_descriptor.assignFragsToMultiDispatch(multifrag_kernel_dispatch);
  } else {
    VLOG(1) << "Creating one execution kernel per fragment";
    VLOG(1) << query_mem_desc.toString();

    if (!ra_exe_unit.use_bump_allocator && allow_single_frag_table_opt &&
        (query_mem_desc.getQueryDescriptionType() == QueryDescriptionType::Projection) &&
        table_infos.size() == 1 && table_infos.front().table_id > 0) {
      const auto max_frag_size =
          table_infos.front().info.getFragmentNumTuplesUpperBound();
      if (max_frag_size < query_mem_desc.getEntryCount()) {
        LOG(INFO) << "Lowering scan limit from " << query_mem_desc.getEntryCount()
                  << " to match max fragment size " << max_frag_size
                  << " for kernel per fragment execution path.";
        throw CompilationRetryNewScanLimit(max_frag_size);
      }
    }

    size_t frag_list_idx{0};
    auto fragment_per_kernel_dispatch = [&ra_exe_unit,
                                         &execution_kernels,
                                         &column_fetcher,
                                         &eo,
                                         &frag_list_idx,
                                         &device_type,
                                         &query_comp_desc,
                                         &query_mem_desc,
                                         render_info](const int device_id,
                                                      const FragmentsList& frag_list,
                                                      const int64_t rowid_lookup_key) {
      if (!frag_list.size()) {
        return;
      }
      CHECK_GE(device_id, 0);

      execution_kernels.emplace_back(
          std::make_unique<ExecutionKernel>(ra_exe_unit,
                                            device_type,
                                            device_id,
                                            eo,
                                            column_fetcher,
                                            query_comp_desc,
                                            query_mem_desc,
                                            frag_list,
                                            ExecutorDispatchMode::KernelPerFragment,
                                            render_info,
                                            rowid_lookup_key));
      ++frag_list_idx;
    };

    fragment_descriptor.assignFragsToKernelDispatch(fragment_per_kernel_dispatch,
                                                    ra_exe_unit);
  }

  return execution_kernels;
}
namespace cider {
// Returns true iff the execution unit contains window functions.
bool is_window_execution_unit(const CiderRelAlgExecutionUnit& ra_exe_unit) {
  return std::any_of(ra_exe_unit.target_exprs.begin(),
                     ra_exe_unit.target_exprs.end(),
                     [](const Analyzer::Expr* expr) {
                       return dynamic_cast<const Analyzer::WindowFunction*>(expr);
                     });
}

inline size_t get_count_distinct_sub_bitmap_count(
    const size_t bitmap_sz_bits,
    const CiderRelAlgExecutionUnit& ra_exe_unit,
    const ExecutorDeviceType device_type) {
  // For count distinct on a column with a very small number of distinct values
  // contention can be very high, especially for non-grouped queries. We'll split
  // the bitmap into multiple sub-bitmaps which are unified to get the full result.
  // The threshold value for bitmap_sz_bits works well on Kepler.
  return bitmap_sz_bits < 50000 && ra_exe_unit.groupby_exprs.empty() &&
                 (device_type == ExecutorDeviceType::GPU || g_cluster)
             ? 64  // NB: must be a power of 2 to keep runtime offset computations cheap
             : 1;
}

CiderRelAlgExecutionUnit decide_approx_count_distinct_implementation(
    const CiderRelAlgExecutionUnit& ra_exe_unit_in,
    const std::vector<InputTableInfo>& table_infos,
    const Executor* executor,
    const ExecutorDeviceType device_type_in,
    std::vector<std::shared_ptr<Analyzer::Expr>>& target_exprs_owned) {
  CiderRelAlgExecutionUnit ra_exe_unit = ra_exe_unit_in;
  for (size_t i = 0; i < ra_exe_unit.target_exprs.size(); ++i) {
    const auto target_expr = ra_exe_unit.target_exprs[i];
    const auto agg_info = get_target_info(target_expr, g_bigint_count);
    if (agg_info.agg_kind != kAPPROX_COUNT_DISTINCT) {
      continue;
    }
    CHECK(dynamic_cast<const Analyzer::AggExpr*>(target_expr));
    const auto arg = static_cast<Analyzer::AggExpr*>(target_expr)->get_own_arg();
    CHECK(arg);
    const auto& arg_ti = arg->get_type_info();
    // Avoid calling getExpressionRange for variable length types (string and array),
    // it'd trigger an assertion since that API expects to be called only for types
    // for which the notion of range is well-defined. A bit of a kludge, but the
    // logic to reject these types anyway is at lower levels in the stack and not
    // really worth pulling into a separate function for now.
    if (!(arg_ti.is_number() || arg_ti.is_boolean() || arg_ti.is_time() ||
          (arg_ti.is_string() && arg_ti.get_compression() == kENCODING_DICT))) {
      continue;
    }
    const auto arg_range = getExpressionRange(arg.get(), table_infos, executor);
    if (arg_range.getType() != ExpressionRangeType::Integer) {
      continue;
    }
    // When running distributed, the threshold for using the precise implementation
    // must be consistent across all leaves, otherwise we could have a mix of precise
    // and approximate bitmaps and we cannot aggregate them.
    const auto device_type = g_cluster ? ExecutorDeviceType::GPU : device_type_in;
    const auto bitmap_sz_bits = arg_range.getIntMax() - arg_range.getIntMin() + 1;
    const auto sub_bitmap_count = cider::get_count_distinct_sub_bitmap_count(
        bitmap_sz_bits, ra_exe_unit, device_type);
    int64_t approx_bitmap_sz_bits{0};
    const auto error_rate =
        static_cast<Analyzer::AggExpr*>(target_expr)->get_error_rate();
    if (error_rate) {
      CHECK(error_rate->get_type_info().get_type() == kINT);
      CHECK_GE(error_rate->get_constval().intval, 1);
      approx_bitmap_sz_bits = hll_size_for_rate(error_rate->get_constval().intval);
    } else {
      approx_bitmap_sz_bits = g_hll_precision_bits;
    }
    CountDistinctDescriptor approx_count_distinct_desc{CountDistinctImplType::Bitmap,
                                                       arg_range.getIntMin(),
                                                       approx_bitmap_sz_bits,
                                                       true,
                                                       device_type,
                                                       sub_bitmap_count};
    CountDistinctDescriptor precise_count_distinct_desc{CountDistinctImplType::Bitmap,
                                                        arg_range.getIntMin(),
                                                        bitmap_sz_bits,
                                                        false,
                                                        device_type,
                                                        sub_bitmap_count};
    if (approx_count_distinct_desc.bitmapPaddedSizeBytes() >=
        precise_count_distinct_desc.bitmapPaddedSizeBytes()) {
      auto precise_count_distinct = makeExpr<Analyzer::AggExpr>(
          get_agg_type(kCOUNT, arg.get()), kCOUNT, arg, true, nullptr);
      target_exprs_owned.push_back(precise_count_distinct);
      ra_exe_unit.target_exprs[i] = precise_count_distinct.get();
    }
  }
  return ra_exe_unit;
}

Fragmenter_Namespace::TableInfo copy_table_info(
    const Fragmenter_Namespace::TableInfo& table_info) {
  Fragmenter_Namespace::TableInfo table_info_copy;
  table_info_copy.chunkKeyPrefix = table_info.chunkKeyPrefix;
  table_info_copy.fragments = table_info.fragments;
  table_info_copy.setPhysicalNumTuples(table_info.getPhysicalNumTuples());
  return table_info_copy;
}

Fragmenter_Namespace::TableInfo synthesize_table_info(const ResultSetPtr& rows) {
  std::vector<Fragmenter_Namespace::FragmentInfo> result;
  if (rows) {
    result.resize(1);
    auto& fragment = result.front();
    fragment.fragmentId = 0;
    fragment.deviceIds.resize(3);
    fragment.resultSet = rows.get();
    fragment.resultSetMutex.reset(new std::mutex());
  }
  Fragmenter_Namespace::TableInfo table_info;
  table_info.fragments = result;
  return table_info;
}

void collect_table_infos(std::vector<InputTableInfo>& table_infos,
                         const std::vector<InputDescriptor>& input_descs,
                         Executor* executor) {
  const auto temporary_tables = executor->getTemporaryTables();
  const auto cat = executor->getCatalog();
  CHECK(cat);
  std::unordered_map<int, size_t> info_cache;
  for (const auto& input_desc : input_descs) {
    const auto table_id = input_desc.getTableId();
    const auto cached_index_it = info_cache.find(table_id);
    if (cached_index_it != info_cache.end()) {
      CHECK_LT(cached_index_it->second, table_infos.size());
      table_infos.push_back(
          {table_id, copy_table_info(table_infos[cached_index_it->second].info)});
      continue;
    }
    if (input_desc.getSourceType() == InputSourceType::RESULT) {
      CHECK_LT(table_id, 0);
      CHECK(temporary_tables);
      const auto it = temporary_tables->find(table_id);
      LOG_IF(FATAL, it == temporary_tables->end())
          << "Failed to find previous query result for node " << -table_id;
      table_infos.push_back({table_id, synthesize_table_info(it->second)});
    } else {
      CHECK(input_desc.getSourceType() == InputSourceType::TABLE);
      table_infos.push_back({table_id, executor->getTableInfo(table_id)});
    }
    CHECK(!table_infos.empty());
    info_cache.insert(std::make_pair(table_id, table_infos.size() - 1));
  }
}

std::vector<InputTableInfo> get_table_infos(const RelAlgExecutionUnit& ra_exe_unit,
                                            Executor* executor) {
  INJECT_TIMER(get_table_infos);
  std::vector<InputTableInfo> table_infos;
  collect_table_infos(table_infos, ra_exe_unit.input_descs, executor);
  return table_infos;
}

/**
 * Determines whether a query needs to compute the size of its output buffer. Returns
 * true for projection queries with no LIMIT or a LIMIT that exceeds the high scan limit
 * threshold (meaning it would be cheaper to compute the number of rows passing or use
 * the bump allocator than allocate the current scan limit per GPU)
 */
bool compute_output_buffer_size(const CiderRelAlgExecutionUnit& ra_exe_unit) {
  for (const auto target_expr : ra_exe_unit.target_exprs) {
    if (dynamic_cast<const Analyzer::AggExpr*>(target_expr)) {
      return false;
    }
  }
  if (ra_exe_unit.groupby_exprs.size() == 1 && !ra_exe_unit.groupby_exprs.front() &&
      (!ra_exe_unit.scan_limit || ra_exe_unit.scan_limit > Executor::high_scan_limit)) {
    return true;
  }
  return false;
}

inline bool exe_unit_has_quals(const CiderRelAlgExecutionUnit ra_exe_unit) {
  return !(ra_exe_unit.quals.empty() && ra_exe_unit.join_quals.empty() &&
           ra_exe_unit.simple_quals.empty());
}

bool isArchPascalOrLater(const ExecutorDeviceType dt,
                         Catalog_Namespace::Catalog* catalog) {
  if (dt == ExecutorDeviceType::GPU) {
    const auto cuda_mgr = catalog->getDataMgr().getCudaMgr();
    LOG_IF(FATAL, cuda_mgr == nullptr)
        << "No CudaMgr instantiated, unable to check device architecture";
    return cuda_mgr->isArchPascalOrLater();
  }
  return false;
}

ExecutorDeviceType getDeviceTypeForTargets(const CiderRelAlgExecutionUnit& ra_exe_unit,
                                           const ExecutorDeviceType requested_device_type,
                                           Catalog_Namespace::Catalog* catalog) {
  for (const auto target_expr : ra_exe_unit.target_exprs) {
    const auto agg_info = get_target_info(target_expr, g_bigint_count);
    if (!ra_exe_unit.groupby_exprs.empty() &&
        !isArchPascalOrLater(requested_device_type, catalog)) {
      if ((agg_info.agg_kind == kAVG || agg_info.agg_kind == kSUM) &&
          agg_info.agg_arg_type.get_type() == kDOUBLE) {
        return ExecutorDeviceType::CPU;
      }
    }
    if (dynamic_cast<const Analyzer::RegexpExpr*>(target_expr)) {
      return ExecutorDeviceType::CPU;
    }
  }
  return requested_device_type;
}

// Compute a very conservative entry count for the output buffer entry count using no
// other information than the number of tuples in each table and multiplying them
// together.
size_t compute_buffer_entry_guess(const std::vector<InputTableInfo>& query_infos) {
  using Fragmenter_Namespace::FragmentInfo;
  // Check for overflows since we're multiplying potentially big table sizes.
  using checked_size_t = boost::multiprecision::number<
      boost::multiprecision::cpp_int_backend<64,
                                             64,
                                             boost::multiprecision::unsigned_magnitude,
                                             boost::multiprecision::checked,
                                             void>>;
  checked_size_t max_groups_buffer_entry_guess = 1;
  for (const auto& query_info : query_infos) {
    CHECK(!query_info.info.fragments.empty());
    auto it = std::max_element(query_info.info.fragments.begin(),
                               query_info.info.fragments.end(),
                               [](const FragmentInfo& f1, const FragmentInfo& f2) {
                                 return f1.getNumTuples() < f2.getNumTuples();
                               });
    max_groups_buffer_entry_guess *= it->getNumTuples();
  }
  // Cap the rough approximation to 100M entries, it's unlikely we can do a great job for
  // baseline group layout with that many entries anyway.
  constexpr size_t max_groups_buffer_entry_guess_cap = 100000000;
  try {
    return std::min(static_cast<size_t>(max_groups_buffer_entry_guess),
                    max_groups_buffer_entry_guess_cap);
  } catch (...) {
    return max_groups_buffer_entry_guess_cap;
  }
}

bool isRowidLookup(const CiderWorkUnit& work_unit) {
  // TODO:
  // const auto& ra_exe_unit = work_unit.exe_unit;
  // if (ra_exe_unit.input_descs.size() != 1) {
  //   return false;
  // }
  // const auto& table_desc = ra_exe_unit.input_descs.front();
  // if (table_desc.getSourceType() != InputSourceType::TABLE) {
  //   return false;
  // }
  // const int table_id = table_desc.getTableId();
  // for (const auto& simple_qual : ra_exe_unit.simple_quals) {
  //   const auto comp_expr =
  //       std::dynamic_pointer_cast<const Analyzer::BinOper>(simple_qual);
  //   if (!comp_expr || comp_expr->get_optype() != kEQ) {
  //     return false;
  //   }
  //   const auto lhs = comp_expr->get_left_operand();
  //   const auto lhs_col = dynamic_cast<const Analyzer::ColumnVar*>(lhs);
  //   if (!lhs_col || !lhs_col->get_table_id() || lhs_col->get_rte_idx()) {
  //     return false;
  //   }
  //   const auto rhs = comp_expr->get_right_operand();
  //   const auto rhs_const = dynamic_cast<const Analyzer::Constant*>(rhs);
  //   if (!rhs_const) {
  //     return false;
  //   }
  //   auto cd = get_column_descriptor(lhs_col->get_column_id(), table_id, cat_);
  //   if (cd->isVirtualCol) {
  //     CHECK_EQ("rowid", cd->columnName);
  //     return true;
  //   }
  // }
  return false;
}

std::tuple<CiderRelAlgExecutionUnit, PlanState::DeletedColumnsMap> addDeletedColumn(
    const CiderRelAlgExecutionUnit& ra_exe_unit,
    const CompilationOptions& co) {
  // TODO: I didn't understand what is deleted_column...
  return std::make_tuple(ra_exe_unit, PlanState::DeletedColumnsMap{});

  // if (!co.filter_on_deleted_column) {
  //   return std::make_tuple(ra_exe_unit, PlanState::DeletedColumnsMap{});
  // }
  // auto ra_exe_unit_with_deleted = ra_exe_unit;
  // PlanState::DeletedColumnsMap deleted_cols_map;
  // for (const auto& input_table : ra_exe_unit_with_deleted.input_descs) {
  //   if (input_table.getSourceType() != InputSourceType::TABLE) {
  //     continue;
  //   }
  //   const auto td = catalog_->getMetadataForTable(input_table.getTableId());
  //   CHECK(td);
  //   const auto deleted_cd = catalog_->getDeletedColumnIfRowsDeleted(td);
  //   if (!deleted_cd) {
  //     continue;
  //   }
  //   CHECK(deleted_cd->columnType.is_boolean());
  //   // check deleted column is not already present
  //   bool found = false;
  //   for (const auto& input_col : ra_exe_unit_with_deleted.input_col_descs) {
  //     if (input_col.get()->getColId() == deleted_cd->columnId &&
  //         input_col.get()->getScanDesc().getTableId() == deleted_cd->tableId &&
  //         input_col.get()->getScanDesc().getNestLevel() == input_table.getNestLevel())
  //         {
  //       found = true;
  //       add_deleted_col_to_map(deleted_cols_map, deleted_cd);
  //       break;
  //     }
  //   }
  //   if (!found) {
  //     // add deleted column
  //     ra_exe_unit_with_deleted.input_col_descs.emplace_back(new InputColDescriptor(
  //         deleted_cd->columnId, deleted_cd->tableId, input_table.getNestLevel()));
  //     add_deleted_col_to_map(deleted_cols_map, deleted_cd);
  //   }
  // }
  // return std::make_tuple(ra_exe_unit_with_deleted, deleted_cols_map);
}

// TODO: a work around to constrcut a RelAlgExecutionUnit .
RelAlgExecutionUnit makeRelAlgExecutionUnit(const CiderRelAlgExecutionUnit& in) {
  return {
      std::vector<InputDescriptor>(),
      std::list<std::shared_ptr<const InputColDescriptor>>(),
      in.simple_quals,
      in.quals,
      in.join_quals,
      in.groupby_exprs,
      in.target_exprs,
      in.estimator,
      in.sort_info,
      in.scan_limit,
      in.query_hint,
      in.use_bump_allocator,
      in.union_all,
      nullptr  // query_state
  };
}

}  // namespace cider

void CiderCodeGenerator::compileWorkUnit1(const CiderWorkUnit& work_unit,
                                          const std::vector<TargetMetaInfo>& targets_meta,
                                          const bool is_agg,
                                          const CompilationOptions& co_in,
                                          const ExecutionOptions& eo,
                                          RenderInfo* render_info,
                                          const int64_t queue_time_ms,
                                          const std::optional<size_t> previous_count) {
  INJECT_TIMER(executeWorkUnit);
  auto timer = DEBUG_TIMER(__func__);

  auto co = co_in;
  ColumnCacheMap column_cache;
  if (cider::is_window_execution_unit(work_unit.exe_unit)) {
    // TODO: support window execution.
    // if (!g_enable_window_functions) {
    //   throw std::runtime_error("Window functions support is disabled");
    // }
    // co.device_type = ExecutorDeviceType::CPU;
    // co.allow_lazy_fetch = false;
    // computeWindow(work_unit.exe_unit, co, eo, column_cache, queue_time_ms);

    throw std::runtime_error("Window functions support is disabled");
  }
  // ideally, join filter PD should be done in earlier step(eg. Spark )
  if (!eo.just_explain && eo.find_push_down_candidates) {
    // find potential candidates:
    // auto selected_filters = selectFiltersToBePushedDown(work_unit, co, eo);
    // if (!selected_filters.empty() || eo.just_calcite_explain) {
    //   return ExecutionResult(selected_filters, eo.find_push_down_candidates);
    // }
    throw std::runtime_error("Not support Filter push down");
  }
  if (render_info && render_info->isPotentialInSituRender()) {
    co.allow_lazy_fetch = false;
  }
  const auto body = work_unit.body;
  CHECK(body);
  auto it = leaf_results_.find(body->getId());
  VLOG(3) << "body->getId()=" << body->getId() << " body->toString()=" << body->toString()
          << " it==leaf_results_.end()=" << (it == leaf_results_.end());
  if (it != leaf_results_.end()) {
    // TODO: we should not step into here???
    // GroupByAndAggregate::addTransientStringLiterals(
    //     work_unit.exe_unit, executor_, executor_->row_set_mem_owner_);
    // auto& aggregated_result = it->second;
    // auto& result_rows = aggregated_result.rs;
    // ExecutionResult result(result_rows, aggregated_result.targets_meta);
    // body->setOutputMetainfo(aggregated_result.targets_meta);
    // if (render_info) {
    //   build_render_targets(*render_info, work_unit.exe_unit.target_exprs,
    //   targets_meta);
    // }
    // return result;
  }
  // TODO!!!!!
  auto tmp = cider::makeRelAlgExecutionUnit(work_unit.exe_unit);
  const auto table_infos = cider::get_table_infos(tmp, executor_);

  // TODO: suuport APPROX_COUNT_DISTINCT syntax
  auto ra_exe_unit = work_unit.exe_unit;
  // auto ra_exe_unit = cider::decide_approx_count_distinct_implementation(
  // work_unit.exe_unit, table_infos, executor_, co.device_type, target_exprs_owned_);

  // TODO: use defaults query hint.
  ra_exe_unit.query_hint = RegisteredQueryHint::defaults();
  // register query hint if query_dag_ is valid
  // ra_exe_unit.query_hint =
  //    query_dag_ ? query_dag_->getQueryHints() : RegisteredQueryHint::defaults();

  auto max_groups_buffer_entry_guess = work_unit.max_groups_buffer_entry_guess;
  if (cider::is_window_execution_unit(ra_exe_unit)) {
    CHECK_EQ(table_infos.size(), size_t(1));
    CHECK_EQ(table_infos.front().info.fragments.size(), size_t(1));
    max_groups_buffer_entry_guess =
        table_infos.front().info.fragments.front().getNumTuples();
    ra_exe_unit.scan_limit = max_groups_buffer_entry_guess;
  } else if (cider::compute_output_buffer_size(ra_exe_unit) &&
             !cider::isRowidLookup(work_unit)) {
    // TODO: in this clause, just need set `ra_exe_unit.scan_limit`
    ra_exe_unit.scan_limit = 0;
    // if (previous_count && !exe_unit_has_quals(ra_exe_unit)) {
    //   ra_exe_unit.scan_limit = *previous_count;
    // } else {
    //   // TODO(adb): enable bump allocator path for render queries
    //   if (can_use_bump_allocator(ra_exe_unit, co, eo) && !render_info) {
    //     ra_exe_unit.scan_limit = 0;
    //     ra_exe_unit.use_bump_allocator = true;
    //   } else if (eo.executor_type == ::ExecutorType::Extern) {
    //     ra_exe_unit.scan_limit = 0;
    //   } else if (!eo.just_explain) {
    //     const auto filter_count_all = getFilteredCountAll(work_unit, true, co, eo);
    //     if (filter_count_all) {
    //       ra_exe_unit.scan_limit = std::max(*filter_count_all, size_t(1));
    //     }
    //   }
    // }
  }

  // max_groups_buffer_entry_guess_in and has_cardinality_estimation is pass from lambda
  size_t max_groups_buffer_entry_guess_in = 0;
  bool has_cardinality_estimation = false;
  auto local_groups_buffer_entry_guess = max_groups_buffer_entry_guess_in;

  const auto [ra_exe_unit_a, deleted_cols_map] = cider::addDeletedColumn(ra_exe_unit, co);
  const auto device_type =
      cider::getDeviceTypeForTargets(ra_exe_unit, co.device_type, catalog_);
  CHECK(!table_infos.empty());
  if (!local_groups_buffer_entry_guess) {
    // The query has failed the first execution attempt because of running out
    // of group by slots. Make the conservative choice: allocate fragment size
    // slots and run on the CPU.
    CHECK(device_type == ExecutorDeviceType::CPU);
    local_groups_buffer_entry_guess = cider::compute_buffer_entry_guess(table_infos);
  }

  int8_t crt_min_byte_width{MAX_BYTE_WIDTH_SUPPORTED};

  SharedKernelContext shared_context(table_infos);
  ColumnFetcher column_fetcher(executor_, column_cache);
  auto query_comp_desc_owned = std::make_unique<QueryCompilationDescriptor>();
  std::unique_ptr<QueryMemoryDescriptor> query_mem_desc_owned;

  INJECT_TIMER(query_step_compilation);  //??
  auto clock_begin = timer_start();
  // TODO : seems not used
  // std::lock_guard<std::mutex> compilation_lock(compilation_mutex_);
  metrics_->compilation_queue_time_ms_ += timer_stop(clock_begin);  //

  auto ra_exe_unit_b = cider::makeRelAlgExecutionUnit(ra_exe_unit_a);
  std::tuple<CompilationResult, std::unique_ptr<QueryMemoryDescriptor>> ret =
      compileWorkUnit(table_infos,
                      deleted_cols_map,
                      ra_exe_unit_b,
                      co,
                      eo,
                      catalog_->getDataMgr().getCudaMgr(),
                      false,
                      executor_->row_set_mem_owner_,
                      local_groups_buffer_entry_guess,
                      crt_min_byte_width,
                      has_cardinality_estimation,
                      column_fetcher.columnarized_table_cache_,  // private
                      render_info);
};
