#include "CiderCodeGenerator.h"

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
// void optimize_ir(llvm::Function* query_func,
//                  llvm::Module* module,
//                  llvm::legacy::PassManager& pass_manager,
//                  const std::unordered_set<llvm::Function*>& live_funcs,
//                  const CompilationOptions& co) {
//   pass_manager.add(llvm::createAlwaysInlinerLegacyPass());
//   pass_manager.add(llvm::createPromoteMemoryToRegisterPass());
//   pass_manager.add(llvm::createInstSimplifyLegacyPass());
//   pass_manager.add(llvm::createInstructionCombiningPass());
//   pass_manager.add(llvm::createGlobalOptimizerPass());

//   pass_manager.add(llvm::createLICMPass());
//   if (co.opt_level == ExecutorOptLevel::LoopStrengthReduction) {
//     pass_manager.add(llvm::createLoopStrengthReducePass());
//   }
//   pass_manager.run(*module);

//   eliminate_dead_self_recursive_funcs(*module, live_funcs);
// }

}


// std::tuple<CompilationResult, std::unique_ptr<QueryMemoryDescriptor>>
// CiderCodeGenerator::compileWorkUnit(const std::vector<InputTableInfo>& query_infos,
//                                     const PlanState::DeletedColumnsMap& deleted_cols_map,
//                                     const RelAlgExecutionUnit& ra_exe_unit,
//                                     const CompilationOptions& co,
//                                     const ExecutionOptions& eo,
//                                     const CudaMgr_Namespace::CudaMgr* cuda_mgr,
//                                     const bool allow_lazy_fetch,
//                                     std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
//                                     const size_t max_groups_buffer_entry_guess,
//                                     const int8_t crt_min_byte_width,
//                                     const bool has_cardinality_estimation,
//                                     ColumnCacheMap& column_cache,
//                                     RenderInfo* render_info,
//                                     std::shared_ptr<CgenState> cgen_state) {
//   auto timer = DEBUG_TIMER(__func__);

// #ifndef NDEBUG
//   static std::uint64_t counter = 0;
//   ++counter;
//   VLOG(1) << "CODEGEN #" << counter << ":";
//   LOG(IR) << "CODEGEN #" << counter << ":";
//   LOG(PTX) << "CODEGEN #" << counter << ":";
//   LOG(ASM) << "CODEGEN #" << counter << ":";
// #endif

//   nukeOldState(allow_lazy_fetch, query_infos, deleted_cols_map, &ra_exe_unit);

//   GroupByAndAggregate group_by_and_aggregate(
//       this,
//       co.device_type,
//       ra_exe_unit,
//       query_infos,
//       row_set_mem_owner,
//       has_cardinality_estimation ? std::optional<int64_t>(max_groups_buffer_entry_guess)
//                                  : std::nullopt);
//   auto query_mem_desc =
//       group_by_and_aggregate.initQueryMemoryDescriptor(eo.allow_multifrag,
//                                                        max_groups_buffer_entry_guess,
//                                                        crt_min_byte_width,
//                                                        render_info,
//                                                        eo.output_columnar_hint);

//   if (query_mem_desc->getQueryDescriptionType() ==
//           QueryDescriptionType::GroupByBaselineHash &&
//       !has_cardinality_estimation &&
//       (!render_info || !render_info->isPotentialInSituRender()) && !eo.just_explain) {
//     const auto col_range_info = group_by_and_aggregate.getColRangeInfo();
//     throw CardinalityEstimationRequired(col_range_info.max - col_range_info.min);
//   }

//   const bool output_columnar = query_mem_desc->didOutputColumnar();

//   // Read the module template and target either CPU or GPU
//   // by binding the stream position functions to the right implementation:
//   // stride access for GPU, contiguous for CPU
//   auto rt_module_copy = llvm::CloneModule(
//       *g_rt_module.get(), cgen_state->vmap_, [](const llvm::GlobalValue* gv) {
//         auto func = llvm::dyn_cast<llvm::Function>(gv);
//         if (!func) {
//           return true;
//         }
//         return (func->getLinkage() == llvm::GlobalValue::LinkageTypes::PrivateLinkage ||
//                 func->getLinkage() == llvm::GlobalValue::LinkageTypes::InternalLinkage ||
//                 CodeGenerator::alwaysCloneRuntimeFunction(func));
//       });
//   if (co.device_type == ExecutorDeviceType::CPU) {
//     if (is_udf_module_present(true)) {
//       CodeGenerator::link_udf_module(udf_cpu_module, *rt_module_copy, cgen_state.get());
//     }
//     if (is_rt_udf_module_present(true)) {
//       CodeGenerator::link_udf_module(
//           rt_udf_cpu_module, *rt_module_copy, cgen_state.get());
//     }
//   }

//   cgen_state->module_ = rt_module_copy.release();
//   AUTOMATIC_IR_METADATA(cgen_state.get());

//   auto agg_fnames =
//       get_agg_fnames(ra_exe_unit.target_exprs, !ra_exe_unit.groupby_exprs.empty());

//   const auto agg_slot_count = ra_exe_unit.estimator ? size_t(1) : agg_fnames.size();

//   const bool is_group_by{query_mem_desc->isGroupBy()};
//   auto [query_func, row_func_call] = is_group_by
//                                          ? query_group_by_template(cgen_state->module_,
//                                                                    co.hoist_literals,
//                                                                    *query_mem_desc,
//                                                                    co.device_type,
//                                                                    ra_exe_unit.scan_limit,
//                                                                    gpu_smem_context)
//                                          : query_template(cgen_state->module_,
//                                                           agg_slot_count,
//                                                           co.hoist_literals,
//                                                           !!ra_exe_unit.estimator,
//                                                           gpu_smem_context);
//   bind_pos_placeholders("pos_start", true, query_func, cgen_state->module_);
//   bind_pos_placeholders("group_buff_idx", false, query_func, cgen_state->module_);
//   bind_pos_placeholders("pos_step", false, query_func, cgen_state->module_);

//   cgen_state->query_func_ = query_func;
//   cgen_state->row_func_call_ = row_func_call;
//   cgen_state->query_func_entry_ir_builder_.SetInsertPoint(
//       &query_func->getEntryBlock().front());

//   // Generate the function signature and column head fetches s.t.
//   // double indirection isn't needed in the inner loop
//   auto& fetch_bb = query_func->front();
//   llvm::IRBuilder<> fetch_ir_builder(&fetch_bb);
//   fetch_ir_builder.SetInsertPoint(&*fetch_bb.begin());
//   auto col_heads = generate_column_heads_load(ra_exe_unit.input_col_descs.size(),
//                                               query_func->args().begin(),
//                                               fetch_ir_builder,
//                                               cgen_state->context_);
//   CHECK_EQ(ra_exe_unit.input_col_descs.size(), col_heads.size());

//   cgen_state->row_func_ = create_row_function(ra_exe_unit.input_col_descs.size(),
//                                               is_group_by ? 0 : agg_slot_count,
//                                               co.hoist_literals,
//                                               cgen_state->module_,
//                                               cgen_state->context_);
//   CHECK(cgen_state->row_func_);
//   cgen_state->row_func_bb_ =
//       llvm::BasicBlock::Create(cgen_state->context_, "entry", cgen_state->row_func_);

//   if (g_enable_filter_function) {
//     auto filter_func_ft =
//         llvm::FunctionType::get(get_int_type(32, cgen_state->context_), {}, false);
//     cgen_state->filter_func_ = llvm::Function::Create(filter_func_ft,
//                                                       llvm::Function::ExternalLinkage,
//                                                       "filter_func",
//                                                       cgen_state->module_);
//     CHECK(cgen_state->filter_func_);
//     cgen_state->filter_func_bb_ =
//         llvm::BasicBlock::Create(cgen_state->context_, "entry", cgen_state->filter_func_);
//   }

//   cgen_state->current_func_ = cgen_state->row_func_;
//   cgen_state->ir_builder_.SetInsertPoint(cgen_state->row_func_bb_);

//   preloadFragOffsets(ra_exe_unit.input_descs, query_infos);
//   RelAlgExecutionUnit body_execution_unit = ra_exe_unit;
//   const auto join_loops =
//       buildJoinLoops(body_execution_unit, co, eo, query_infos, column_cache);

//   plan_state_->allocateLocalColumnIds(ra_exe_unit.input_col_descs);
//   const auto is_not_deleted_bb = codegenSkipDeletedOuterTableRow(ra_exe_unit, co);
//   if (is_not_deleted_bb) {
//     cgen_state->row_func_bb_ = is_not_deleted_bb;
//   }
//   if (!join_loops.empty()) {
//     codegenJoinLoops(join_loops,
//                      body_execution_unit,
//                      group_by_and_aggregate,
//                      query_func,
//                      cgen_state->row_func_bb_,
//                      *(query_mem_desc.get()),
//                      co,
//                      eo);
//   } else {
//     const bool can_return_error = compileBody(
//         ra_exe_unit, group_by_and_aggregate, *query_mem_desc, co, gpu_smem_context);
//     if (can_return_error || cgen_state->needs_error_check_ || eo.with_dynamic_watchdog ||
//         eo.allow_runtime_query_interrupt) {
//       createErrorCheckControlFlow(query_func,
//                                   eo.with_dynamic_watchdog,
//                                   eo.allow_runtime_query_interrupt,
//                                   co.device_type,
//                                   group_by_and_aggregate.query_infos_);
//     }
//   }
//   std::vector<llvm::Value*> hoisted_literals;

//   if (co.hoist_literals) {
//     VLOG(1) << "number of hoisted literals: "
//             << cgen_state->query_func_literal_loads_.size()
//             << " / literal buffer usage: " << cgen_state->getLiteralBufferUsage(0)
//             << " bytes";
//   }

//   if (co.hoist_literals && !cgen_state->query_func_literal_loads_.empty()) {
//     // we have some hoisted literals...
//     hoisted_literals = inlineHoistedLiterals();
//   }

//   // replace the row func placeholder call with the call to the actual row func
//   std::vector<llvm::Value*> row_func_args;
//   for (size_t i = 0; i < cgen_state->row_func_call_->getNumArgOperands(); ++i) {
//     row_func_args.push_back(cgen_state->row_func_call_->getArgOperand(i));
//   }
//   row_func_args.insert(row_func_args.end(), col_heads.begin(), col_heads.end());
//   row_func_args.push_back(get_arg_by_name(query_func, "join_hash_tables"));
//   // push hoisted literals arguments, if any
//   row_func_args.insert(
//       row_func_args.end(), hoisted_literals.begin(), hoisted_literals.end());
//   llvm::ReplaceInstWithInst(
//       cgen_state->row_func_call_,
//       llvm::CallInst::Create(cgen_state->row_func_, row_func_args, ""));

//   // replace the filter func placeholder call with the call to the actual filter func
//   if (cgen_state->filter_func_) {
//     std::vector<llvm::Value*> filter_func_args;
//     for (auto arg_it = cgen_state->filter_func_args_.begin();
//          arg_it != cgen_state->filter_func_args_.end();
//          ++arg_it) {
//       filter_func_args.push_back(arg_it->first);
//     }
//     llvm::ReplaceInstWithInst(
//         cgen_state->filter_func_call_,
//         llvm::CallInst::Create(cgen_state->filter_func_, filter_func_args, ""));
//   }

//   // Aggregate
//   plan_state_->init_agg_vals_ =
//       init_agg_val_vec(ra_exe_unit.target_exprs, ra_exe_unit.quals, *query_mem_desc);

//   auto multifrag_query_func = cgen_state->module_->getFunction(
//       "multifrag_query" + std::string(co.hoist_literals ? "_hoisted_literals" : ""));
//   CHECK(multifrag_query_func);

//   bind_query(query_func,
//              "query_stub" + std::string(co.hoist_literals ? "_hoisted_literals" : ""),
//              multifrag_query_func,
//              cgen_state->module_);

//   std::vector<llvm::Function*> root_funcs{query_func, cgen_state->row_func_};
//   if (cgen_state->filter_func_) {
//     root_funcs.push_back(cgen_state->filter_func_);
//   }
//   auto live_funcs = CodeGenerator::markDeadRuntimeFuncs(
//       *cgen_state->module_, root_funcs, {multifrag_query_func});

// #include "LLVMFunctionAttributesUtil.h"
//   // Always inline the row function and the filter function.
//   // We don't want register spills in the inner loops.
//   // LLVM seems to correctly free up alloca instructions
//   // in these functions even when they are inlined.
//   mark_function_always_inline(cgen_state->row_func_);
//   if (cgen_state->filter_func_) {
//     mark_function_always_inline(cgen_state->filter_func_);
//   }

// #ifndef NDEBUG
//   // Add helpful metadata to the LLVM IR for debugging.
//   AUTOMATIC_IR_METADATA_DONE();
// #endif

//   // Serialize the important LLVM IR functions to text for SQL EXPLAIN.
//   std::string llvm_ir;
//   if (eo.just_explain) {
//     if (co.explain_type == ExecutorExplainType::Optimized) {
// #ifdef WITH_JIT_DEBUG
//       throw std::runtime_error(
//           "Explain optimized not available when JIT runtime debug symbols are
//           enabled");
// #else
//       // Note that we don't run the NVVM reflect pass here. Use LOG(IR) to get the
//       // optimized IR after NVVM reflect
//       llvm::legacy::PassManager pass_manager;
//       cider::optimize_ir(query_func, cgen_state->module_, pass_manager, live_funcs, co);
// #endif  // WITH_JIT_DEBUG
//     }
//     llvm_ir =
//         serialize_llvm_object(multifrag_query_func) + serialize_llvm_object(query_func) +
//         serialize_llvm_object(cgen_state->row_func_) +
//         (cgen_state->filter_func_ ? serialize_llvm_object(cgen_state->filter_func_) : "");

// #ifndef NDEBUG
//     llvm_ir += serialize_llvm_metadata_footnotes(query_func, cgen_state.get());
// #endif
//   }

//   LOG(IR) << "\n\n" << query_mem_desc->toString() << "\n";
//   LOG(IR) << "IR for the "
//           << (co.device_type == ExecutorDeviceType::CPU ? "CPU:\n" : "GPU:\n");
// #ifdef NDEBUG
//   LOG(IR) << serialize_llvm_object(query_func)
//           << serialize_llvm_object(cgen_state->row_func_)
//           << (cgen_state->filter_func_ ? serialize_llvm_object(cgen_state->filter_func_)
//                                        : "")
//           << "\nEnd of IR";
// #else
//   LOG(IR) << serialize_llvm_object(cgen_state->module_) << "\nEnd of IR";
// #endif

//   // Run some basic validation checks on the LLVM IR before code is generated below.
//   verify_function_ir(cgen_state->row_func_);
//   if (cgen_state->filter_func_) {
//     verify_function_ir(cgen_state->filter_func_);
//   }

//   // Generate final native code from the LLVM IR.
//   return std::make_tuple(
//       CompilationResult{optimizeAndCodegenCPU(
//                             query_func, multifrag_query_func, live_funcs, co, cgen_state),
//                         cgen_state->getLiterals(),
//                         output_columnar,
//                         llvm_ir,
//                         {}},
//       std::move(query_mem_desc));
// }
