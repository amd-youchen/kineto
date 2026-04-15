/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "RoctracerLogger.h"

#include <time.h>
#include <unistd.h>
#include <chrono>
#include <cstring>
#include <mutex>

#include "ApproximateClock.h"
#include "Demangle.h"
#include "Logger.h"
#include "ThreadUtil.h"

using namespace libkineto;
using namespace std::chrono;

class Flush {
 public:
  std::mutex mutex_;
  std::atomic<uint64_t> maxCorrelationId_;
  uint64_t maxCompletedCorrelationId_{0};
  void reportCorrelation(const uint64_t& cid) {
    uint64_t prev = maxCorrelationId_;
    while (prev < cid && !maxCorrelationId_.compare_exchange_weak(prev, cid)) {
    }
  }
};
static Flush s_flush;

RoctracerLogger& RoctracerLogger::singleton() {
  static RoctracerLogger instance;
  return instance;
}

RoctracerLogger::RoctracerLogger() {}

RoctracerLogger::~RoctracerLogger() {
  stopLogging();
  endTracing();
}

namespace {
thread_local std::deque<uint64_t>
    t_externalIds[RoctracerLogger::CorrelationDomain::size];
}

void RoctracerLogger::pushCorrelationID(uint64_t id, CorrelationDomain type) {
  if (!singleton().externalCorrelationEnabled_) {
    return;
  }
  t_externalIds[type].push_back(id);
}

void RoctracerLogger::popCorrelationID(CorrelationDomain type) {
  if (!singleton().externalCorrelationEnabled_) {
    return;
  }
  if (!t_externalIds[type].empty()) {
    t_externalIds[type].pop_back();
  } else {
    LOG(ERROR)
        << "Attempt to popCorrelationID from an empty external Ids stack";
  }
}

void RoctracerLogger::clearLogs() {
  size_t rowsBeforeClear = 0;
  {
    std::lock_guard<std::mutex> lock(rowsMutex_);
    rowsBeforeClear = rows_.size();
    rows_.clear();
  }

  size_t externalTotalBeforeClear = 0;
  size_t externalPerDomain[CorrelationDomain::size] = {0};
  {
    std::lock_guard<std::mutex> lock(externalCorrelationsMutex_);
    for (int i = 0; i < CorrelationDomain::size; ++i) {
      externalPerDomain[i] = externalCorrelations_[i].size();
      externalTotalBeforeClear += externalPerDomain[i];
      externalCorrelations_[i].clear();
    }
  }

  const auto clearCount = debugState_.clearLogsCalls.fetch_add(1) + 1;
  LOG(INFO) << "ROCtracer clearLogs #" << clearCount
            << ": rows_before=" << rowsBeforeClear
            << ", external_correlations_before={total="
            << externalTotalBeforeClear << ", domain0="
            << externalPerDomain[CorrelationDomain::Domain0] << ", domain1="
            << externalPerDomain[CorrelationDomain::Domain1] << "}"
            << ", callback_stats={batches="
            << debugState_.activityCallbackBatches.load()
            << ", rows=" << debugState_.activityCallbackRows.load()
            << ", during_stop_batches="
            << debugState_.stopPhaseCallbackBatches.load()
            << ", during_stop_rows="
            << debugState_.stopPhaseCallbackRows.load()
            << ", after_stop_batches="
            << debugState_.postStopCallbackBatches.load()
            << ", after_stop_rows="
            << debugState_.postStopCallbackRows.load() << "}";
}

void RoctracerLogger::flushActivities() {
  if (hccPool_ != nullptr) {
    roctracer_flush_activity_expl(hccPool_);
  }
}

void RoctracerLogger::insert_row_to_buffer(roctracerBase* row) {
  RoctracerLogger* dis = &singleton();
  std::lock_guard<std::mutex> lock(dis->rowsMutex_);
  if (dis->rows_.size() >= dis->maxBufferSize_) {
    dis->debugState_.droppedRowsByBufferLimit.fetch_add(1);
    LOG_FIRST_N(WARNING, 10)
        << "Exceeded max GPU buffer count (" << dis->rows_.size() << " > "
        << dis->maxBufferSize_ << ") - dropping row id=" << row->id
        << ", type=" << row->type;
    return;
  }
  dis->rows_.push_back(row);
}

void RoctracerLogger::api_callback(
    uint32_t domain,
    uint32_t cid,
    const void* callback_data,
    void* arg) {
  RoctracerLogger* dis = &singleton();

  if (domain == ACTIVITY_DOMAIN_HIP_API && dis->loggedIds_.contains(cid)) {
    const hip_api_data_t* data = (const hip_api_data_t*)(callback_data);

    // Pack callbacks into row structures

    thread_local std::unordered_map<activity_correlation_id_t, uint64_t>
        timestamps;

    if (data->phase == ACTIVITY_API_PHASE_ENTER) {
      timestamps[data->correlation_id] = getApproximateTime();
    } else { // (data->phase == ACTIVITY_API_PHASE_EXIT)
      uint64_t startTime = timestamps[data->correlation_id];
      timestamps.erase(data->correlation_id);
      uint64_t endTime = getApproximateTime();

      switch (cid) {
        case HIP_API_ID_hipLaunchKernel:
        case HIP_API_ID_hipExtLaunchKernel:
        case HIP_API_ID_hipLaunchCooperativeKernel: // Should work here
        {
          s_flush.reportCorrelation(data->correlation_id);
          auto& args = data->args.hipLaunchKernel;
          roctracerKernelRow* row = new roctracerKernelRow(
              data->correlation_id,
              domain,
              cid,
              processId(),
              systemThreadId(),
              startTime,
              endTime,
              args.function_address,
              nullptr,
              args.numBlocks.x,
              args.numBlocks.y,
              args.numBlocks.z,
              args.dimBlocks.x,
              args.dimBlocks.y,
              args.dimBlocks.z,
              args.sharedMemBytes,
              args.stream);
          insert_row_to_buffer(row);
        } break;
        case HIP_API_ID_hipHccModuleLaunchKernel:
        case HIP_API_ID_hipModuleLaunchKernel:
        case HIP_API_ID_hipExtModuleLaunchKernel: {
          s_flush.reportCorrelation(data->correlation_id);
          auto& args = data->args.hipModuleLaunchKernel;
          roctracerKernelRow* row = new roctracerKernelRow(
              data->correlation_id,
              domain,
              cid,
              processId(),
              systemThreadId(),
              startTime,
              endTime,
              nullptr,
              args.f,
              args.gridDimX,
              args.gridDimY,
              args.gridDimZ,
              args.blockDimX,
              args.blockDimY,
              args.blockDimZ,
              args.sharedMemBytes,
              args.stream);
          insert_row_to_buffer(row);
        } break;
        case HIP_API_ID_hipLaunchCooperativeKernelMultiDevice:
        case HIP_API_ID_hipExtLaunchMultiKernelMultiDevice:
#if 0
          {
            auto &args = data->args.hipLaunchCooperativeKernelMultiDevice.launchParamsList__val;
            roctracerKernelRow* row = new roctracerKernelRow(
              data->correlation_id,
              domain,
              cid,
              processId(),
              systemThreadId(),
              startTime,
              endTime,
              args.function_address,
              nullptr,
              args.numBlocks.x,
              args.numBlocks.y,
              args.numBlocks.z,
              args.dimBlocks.x,
              args.dimBlocks.y,
              args.dimBlocks.z,
              args.sharedMemBytes,
              args.stream
            );
            insert_row_to_buffer(row);
          }
#endif
          break;
        case HIP_API_ID_hipMalloc: {
          roctracerMallocRow* row = new roctracerMallocRow(
              data->correlation_id,
              domain,
              cid,
              processId(),
              systemThreadId(),
              startTime,
              endTime,
              data->args.hipMalloc.ptr__val,
              data->args.hipMalloc.size);
          insert_row_to_buffer(row);
        } break;
        case HIP_API_ID_hipFree: {
          roctracerMallocRow* row = new roctracerMallocRow(
              data->correlation_id,
              domain,
              cid,
              processId(),
              systemThreadId(),
              startTime,
              endTime,
              data->args.hipFree.ptr,
              0);
          insert_row_to_buffer(row);
        } break;
        case HIP_API_ID_hipMemcpy: {
          auto& args = data->args.hipMemcpy;
          roctracerCopyRow* row = new roctracerCopyRow(
              data->correlation_id,
              domain,
              cid,
              processId(),
              systemThreadId(),
              startTime,
              endTime,
              args.src,
              args.dst,
              args.sizeBytes,
              args.kind,
              static_cast<hipStream_t>(0) // use placeholder?
          );
          insert_row_to_buffer(row);
        } break;
        case HIP_API_ID_hipMemcpyAsync:
        case HIP_API_ID_hipMemcpyWithStream: {
          auto& args = data->args.hipMemcpyAsync;
          roctracerCopyRow* row = new roctracerCopyRow(
              data->correlation_id,
              domain,
              cid,
              processId(),
              systemThreadId(),
              startTime,
              endTime,
              args.src,
              args.dst,
              args.sizeBytes,
              args.kind,
              args.stream);
          insert_row_to_buffer(row);
        } break;
        default: {
          roctracerRow* row = new roctracerRow(
              data->correlation_id,
              domain,
              cid,
              processId(),
              systemThreadId(),
              startTime,
              endTime);
          insert_row_to_buffer(row);
        } break;
      } // switch

      if (dis->hipGraphLaunchOpId_ != std::numeric_limits<uint32_t>::max() &&
          cid == dis->hipGraphLaunchOpId_) {
        size_t rowsBuffered = 0;
        {
          std::lock_guard<std::mutex> rowsLock(dis->rowsMutex_);
          rowsBuffered = dis->rows_.size();
        }
        GraphLaunchDebugEvent event;
        {
          std::lock_guard<std::mutex> graphLock(dis->graphLaunchDebugMutex_);
          event.sequence = ++dis->graphLaunchSequence_;
          event.correlationId = data->correlation_id;
          event.apiBegin = startTime;
          event.apiEnd = endTime;
          event.rowsBufferedAtApiExit = rowsBuffered;
          event.activityBatchesAtApiExit =
              dis->debugState_.activityCallbackBatches.load();
          event.activityRowsAtApiExit =
              dis->debugState_.activityCallbackRows.load();
          event.stopPhaseRowsAtApiExit =
              dis->debugState_.stopPhaseCallbackRows.load();
          dis->recentGraphLaunches_.push_back(event);
          while (
              dis->recentGraphLaunches_.size() >
              RoctracerLogger::kGraphLaunchDebugHistory) {
            dis->recentGraphLaunches_.pop_front();
          }
          dis->trackedGraphLaunches_.store(dis->recentGraphLaunches_.size());
        }
        LOG(INFO) << "ROCtracer observed hipGraphLaunch: correlation="
                  << data->correlation_id << ", cid=" << cid
                  << ", rows_buffered_at_api_exit=" << rowsBuffered
                  << ", callback_snapshot={batches="
                  << dis->debugState_.activityCallbackBatches.load()
                  << ", rows=" << dis->debugState_.activityCallbackRows.load()
                  << ", stop_rows="
                  << dis->debugState_.stopPhaseCallbackRows.load() << "}"
                  << ", flush_state={reported="
                  << s_flush.maxCorrelationId_.load() << ", completed="
                  << s_flush.maxCompletedCorrelationId_ << "}";
      }

      // External correlation
      for (int it = CorrelationDomain::begin; it < CorrelationDomain::end;
           ++it) {
        if (t_externalIds[it].size() > 0) {
          std::lock_guard<std::mutex> lock(dis->externalCorrelationsMutex_);
          dis->externalCorrelations_[it].emplace_back(
              data->correlation_id, t_externalIds[it].back());
        }
      }
    } // phase exit
  }
}

void RoctracerLogger::activity_callback(
    const char* begin,
    const char* end,
    void* arg) {
  RoctracerLogger* dis = &singleton();
  // Log latest completed correlation id.  Used to ensure we have flushed all
  // data on stop
  std::unique_lock<std::mutex> lock(s_flush.mutex_);
  const roctracer_record_t* record = (const roctracer_record_t*)(begin);
  const roctracer_record_t* end_record = (const roctracer_record_t*)(end);
  bool sawRecord = false;
  uint64_t batchRows = 0;
  uint64_t minCorrelationId = 0;
  uint64_t maxCorrelationId = 0;
  uint64_t minBeginNs = 0;
  uint64_t maxEndNs = 0;

  while (record < end_record) {
    if (!sawRecord) {
      sawRecord = true;
      minCorrelationId = record->correlation_id;
      maxCorrelationId = record->correlation_id;
      minBeginNs = record->begin_ns;
      maxEndNs = record->end_ns;
    } else {
      if (record->correlation_id < minCorrelationId) {
        minCorrelationId = record->correlation_id;
      }
      if (record->correlation_id > maxCorrelationId) {
        maxCorrelationId = record->correlation_id;
      }
      if (record->begin_ns < minBeginNs) {
        minBeginNs = record->begin_ns;
      }
      if (record->end_ns > maxEndNs) {
        maxEndNs = record->end_ns;
      }
    }
    ++batchRows;
    if (record->correlation_id > s_flush.maxCompletedCorrelationId_) {
      s_flush.maxCompletedCorrelationId_ = record->correlation_id;
    }
    roctracerAsyncRow* row = new roctracerAsyncRow(
        record->correlation_id,
        record->domain,
        record->kind,
        record->op,
        record->device_id,
        record->queue_id,
        record->begin_ns,
        record->end_ns,
        ((record->kind == HIP_OP_DISPATCH_KIND_KERNEL_) ||
         (record->kind == HIP_OP_DISPATCH_KIND_TASK_))
            ? demangle(record->kernel_name)
            : std::string());
    insert_row_to_buffer(row);
    roctracer_next_record(record, &record);
  }

  const auto batchIndex = dis->debugState_.activityCallbackBatches.fetch_add(1) +
      1;
  dis->debugState_.activityCallbackRows.fetch_add(batchRows);
  const bool stopRequested = dis->debugState_.stopRequested.load();
  const bool roctracerStopIssued = dis->debugState_.roctracerStopIssued.load();
  if (roctracerStopIssued) {
    dis->debugState_.postStopCallbackBatches.fetch_add(1);
    dis->debugState_.postStopCallbackRows.fetch_add(batchRows);
  } else if (stopRequested) {
    dis->debugState_.stopPhaseCallbackBatches.fetch_add(1);
    dis->debugState_.stopPhaseCallbackRows.fetch_add(batchRows);
  }

  if (sawRecord && dis->trackedGraphLaunches_.load() > 0) {
    uint64_t graphCorrelationHits = 0;
    {
      std::lock_guard<std::mutex> graphLock(dis->graphLaunchDebugMutex_);
      for (auto& event : dis->recentGraphLaunches_) {
        if (batchIndex > event.activityBatchesAtApiExit) {
          event.asyncBatchesAfterApi++;
          event.asyncRowsAfterApi += batchRows;
          if (stopRequested) {
            event.stopPhaseBatchesAfterApi++;
            event.stopPhaseRowsAfterApi += batchRows;
          }
        }
        if (event.correlationId >= minCorrelationId &&
            event.correlationId <= maxCorrelationId) {
          event.correlationCoverBatches++;
          event.correlationCoverRows += batchRows;
          event.lastCorrelationCoverBatchIndex = batchIndex;
          event.lastCorrelationCoverMinId = minCorrelationId;
          event.lastCorrelationCoverMaxId = maxCorrelationId;
          ++graphCorrelationHits;
        }
      }
    }
    if (graphCorrelationHits > 0) {
      size_t rowsBuffered = 0;
      {
        std::lock_guard<std::mutex> rowsLock(dis->rowsMutex_);
        rowsBuffered = dis->rows_.size();
      }
      LOG_FIRST_N(INFO, 20)
          << "ROCtracer graph-launch correlated activity batch: batch="
          << batchIndex << ", graph_launch_hits=" << graphCorrelationHits
          << ", rows=" << batchRows << ", corr_range=[" << minCorrelationId
          << ", " << maxCorrelationId << "]"
          << ", stop_requested=" << stopRequested
          << ", rows_buffered=" << rowsBuffered;
    }
  }

  if (sawRecord) {
    if (roctracerStopIssued) {
      size_t rowsBuffered = 0;
      {
        std::lock_guard<std::mutex> rowsLock(dis->rowsMutex_);
        rowsBuffered = dis->rows_.size();
      }
      LOG_FIRST_N(WARNING, 20)
          << "ROCtracer activity callback after roctracer_stop: batch="
          << batchIndex << ", rows=" << batchRows << ", corr_range=["
          << minCorrelationId << ", " << maxCorrelationId << "]"
          << ", time_range_ns=[" << minBeginNs << ", " << maxEndNs << "]"
          << ", max_completed_correlation="
          << s_flush.maxCompletedCorrelationId_ << ", rows_buffered="
          << rowsBuffered;
    } else if (stopRequested) {
      size_t rowsBuffered = 0;
      {
        std::lock_guard<std::mutex> rowsLock(dis->rowsMutex_);
        rowsBuffered = dis->rows_.size();
      }
      LOG_FIRST_N(INFO, 20)
          << "ROCtracer activity callback while stop is in progress: batch="
          << batchIndex << ", rows=" << batchRows << ", corr_range=["
          << minCorrelationId << ", " << maxCorrelationId << "]"
          << ", time_range_ns=[" << minBeginNs << ", " << maxEndNs << "]"
          << ", max_completed_correlation="
          << s_flush.maxCompletedCorrelationId_ << ", rows_buffered="
          << rowsBuffered;
    } else {
      LOG_FIRST_N(INFO, 10)
          << "ROCtracer activity callback: batch=" << batchIndex
          << ", rows=" << batchRows << ", corr_range=[" << minCorrelationId
          << ", " << maxCorrelationId << "]"
          << ", time_range_ns=[" << minBeginNs << ", " << maxEndNs << "]"
          << ", max_completed_correlation="
          << s_flush.maxCompletedCorrelationId_;
    }
  }
}

void RoctracerLogger::setMaxEvents(uint32_t maxBufferSize) {
#ifdef HAS_ROCTRACER
  RoctracerLogger* dis = &singleton();
  std::lock_guard<std::mutex> lock(dis->rowsMutex_);
  maxBufferSize_ = maxBufferSize;
#endif
}

void RoctracerLogger::startLogging() {
  size_t rowsBeforeStart = 0;
  {
    std::lock_guard<std::mutex> lock(rowsMutex_);
    rowsBeforeStart = rows_.size();
  }
  size_t externalCorrelationsBeforeStart = 0;
  {
    std::lock_guard<std::mutex> lock(externalCorrelationsMutex_);
    for (int i = 0; i < CorrelationDomain::size; ++i) {
      externalCorrelationsBeforeStart += externalCorrelations_[i].size();
    }
  }
  debugState_.resetForNewRun();
  {
    std::lock_guard<std::mutex> graphLock(graphLaunchDebugMutex_);
    recentGraphLaunches_.clear();
    graphLaunchSequence_ = 0;
  }
  trackedGraphLaunches_.store(0);
  if (!hipGraphLaunchOpIdInitialized_ ||
      hipGraphLaunchOpId_ == std::numeric_limits<uint32_t>::max()) {
    uint32_t graphLaunchCid = 0;
    const auto graphLaunchStatus = roctracer_op_code(
        ACTIVITY_DOMAIN_HIP_API, "hipGraphLaunch", &graphLaunchCid, nullptr);
    if (graphLaunchStatus == ROCTRACER_STATUS_SUCCESS) {
      hipGraphLaunchOpIdInitialized_ = true;
      hipGraphLaunchOpId_ = graphLaunchCid;
      LOG(INFO) << "ROCtracer graph debug: resolved hipGraphLaunch cid="
                << hipGraphLaunchOpId_;
    } else {
      hipGraphLaunchOpIdInitialized_ = false;
      hipGraphLaunchOpId_ = std::numeric_limits<uint32_t>::max();
      LOG(WARNING)
          << "ROCtracer graph debug: failed to resolve hipGraphLaunch cid, status="
          << graphLaunchStatus;
    }
  }
  LOG(INFO) << "ROCtracer startLogging debug_v2: registered=" << registered_
            << ", logging=" << logging_ << ", maxBufferSize="
            << maxBufferSize_ << ", rows_buffered_before_start="
            << rowsBeforeStart
            << ", external_correlations_before_start="
            << externalCorrelationsBeforeStart << ", flush_state={reported="
            << s_flush.maxCorrelationId_.load()
            << ", completed=" << s_flush.maxCompletedCorrelationId_ << "}";

  if (!registered_) {
    roctracer_set_properties(
        ACTIVITY_DOMAIN_HIP_API, nullptr); // Magic encantation

    // Set some api calls to ignore
    loggedIds_.setInvertMode(true); // Omit the specified api
    loggedIds_.add("hipGetDevice");
    loggedIds_.add("hipSetDevice");
    loggedIds_.add("hipGetLastError");
    loggedIds_.add("__hipPushCallConfiguration");
    loggedIds_.add("__hipPopCallConfiguration");
    loggedIds_.add("hipCtxSetCurrent");
    loggedIds_.add("hipEventRecord");
    loggedIds_.add("hipEventQuery");
    loggedIds_.add("hipGetDeviceProperties");
    loggedIds_.add("hipPeekAtLastError");
    loggedIds_.add("hipModuleGetFunction");
    loggedIds_.add("hipEventCreateWithFlags");
    loggedIds_.add("hipGetDeviceCount");
    loggedIds_.add("hipDevicePrimaryCtxGetState");

    // Enable API callbacks
    if (loggedIds_.invertMode() == true) {
      // exclusion list - enable entire domain and turn off things in list
      roctracer_enable_domain_callback(
          ACTIVITY_DOMAIN_HIP_API, api_callback, nullptr);
      const std::unordered_map<uint32_t, uint32_t>& filter =
          loggedIds_.filterList();
      for (auto it = filter.begin(); it != filter.end(); ++it) {
        roctracer_disable_op_callback(ACTIVITY_DOMAIN_HIP_API, it->first);
      }
    } else {
      // inclusion list - only enable things in the list
      const std::unordered_map<uint32_t, uint32_t>& filter =
          loggedIds_.filterList();
      roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API);
      for (auto it = filter.begin(); it != filter.end(); ++it) {
        roctracer_enable_op_callback(
            ACTIVITY_DOMAIN_HIP_API, it->first, api_callback, nullptr);
      }
    }
    // roctracer_enable_domain_callback(ACTIVITY_DOMAIN_ROCTX, api_callback,
    // nullptr);

    // Allocate default tracing pool
    roctracer_properties_t properties;
    memset(&properties, 0, sizeof(roctracer_properties_t));
    properties.buffer_size = 0x1000;
    roctracer_open_pool(&properties);

    // Enable async op collection
    roctracer_properties_t hcc_cb_properties;
    memset(&hcc_cb_properties, 0, sizeof(roctracer_properties_t));
    hcc_cb_properties.buffer_size = 0x4000;
    hcc_cb_properties.buffer_callback_fun = activity_callback;
    roctracer_open_pool_expl(&hcc_cb_properties, &hccPool_);
    roctracer_enable_domain_activity_expl(ACTIVITY_DOMAIN_HCC_OPS, hccPool_);

    registered_ = true;
  }

  externalCorrelationEnabled_ = true;
  logging_ = true;
  roctracer_start();
  LOG(INFO) << "ROCtracer startLogging active: hccPool=" << hccPool_
            << ", externalCorrelationEnabled="
            << externalCorrelationEnabled_;
}

void RoctracerLogger::stopLogging() {
  if (logging_ == false) {
    LOG(INFO) << "ROCtracer stopLogging skipped: logging already disabled"
              << ", flush_state={reported="
              << s_flush.maxCorrelationId_.load() << ", completed="
              << s_flush.maxCompletedCorrelationId_ << "}"
              << ", callback_stats={batches="
              << debugState_.activityCallbackBatches.load()
              << ", rows=" << debugState_.activityCallbackRows.load()
              << ", during_stop_batches="
              << debugState_.stopPhaseCallbackBatches.load()
              << ", during_stop_rows="
              << debugState_.stopPhaseCallbackRows.load()
              << ", after_stop_batches="
              << debugState_.postStopCallbackBatches.load()
              << ", after_stop_rows="
              << debugState_.postStopCallbackRows.load() << "}";
    return;
  }

  logging_ = false;
  debugState_.stopRequested.store(true);

  auto countExternalCorrelations = [this]() {
    size_t total = 0;
    std::lock_guard<std::mutex> lock(externalCorrelationsMutex_);
    for (int i = 0; i < CorrelationDomain::size; ++i) {
      total += externalCorrelations_[i].size();
    }
    return total;
  };
  auto countRows = [this]() {
    std::lock_guard<std::mutex> lock(rowsMutex_);
    return rows_.size();
  };

  const auto correlationIdBeforeSync = s_flush.maxCorrelationId_.load();
  const auto completedBeforeSync = s_flush.maxCompletedCorrelationId_;
  LOG(INFO) << "ROCtracer stopLogging begin: rows_buffered=" << countRows()
            << ", external_correlations=" << countExternalCorrelations()
            << ", flush_state={target=" << correlationIdBeforeSync
            << ", completed=" << completedBeforeSync
            << ", pending="
            << (correlationIdBeforeSync >= completedBeforeSync
                    ? correlationIdBeforeSync - completedBeforeSync
                    : 0)
            << "}, callback_stats={batches="
            << debugState_.activityCallbackBatches.load()
            << ", rows=" << debugState_.activityCallbackRows.load()
            << ", dropped_rows="
            << debugState_.droppedRowsByBufferLimit.load() << "}";

  const auto syncStart = steady_clock::now();
  hipError_t err = hipDeviceSynchronize();
  const auto syncDurationUs =
      duration_cast<microseconds>(steady_clock::now() - syncStart).count();
  if (err != hipSuccess) {
    LOG(ERROR) << "hipDeviceSynchronize failed with code " << err;
  }
  LOG(INFO) << "ROCtracer stopLogging after hipDeviceSynchronize: err="
            << static_cast<int>(err) << ", duration_us=" << syncDurationUs
            << ", rows_buffered=" << countRows() << ", flush_state={target="
            << s_flush.maxCorrelationId_.load() << ", completed="
            << s_flush.maxCompletedCorrelationId_ << "}";

  roctracer_flush_activity_expl(hccPool_);
  LOG(INFO) << "ROCtracer stopLogging after initial flush: rows_buffered="
            << countRows() << ", flush_state={target="
            << s_flush.maxCorrelationId_.load() << ", completed="
            << s_flush.maxCompletedCorrelationId_ << "}";

  // If we are stopping the tracer, implement reliable flushing
  std::unique_lock<std::mutex> lock(s_flush.mutex_);

  auto correlationId =
      s_flush.maxCorrelationId_.load(); // load ending id from the running max
  constexpr int kCorrelationFlushPollLimit = 50;
  constexpr int kStabilizationFlushPollLimit = 250;
  constexpr int kRequiredStableFlushPolls = 5;
  constexpr useconds_t kFlushPollSleepUsec = 1000;
  constexpr auto kMinAdditionalDrainDuration = milliseconds(50);
  auto flushOnce = [&]() {
    lock.unlock();
    roctracer_flush_activity_expl(hccPool_);
    usleep(kFlushPollSleepUsec);
    lock.lock();
    const auto latestTarget = s_flush.maxCorrelationId_.load();
    if (latestTarget > correlationId) {
      correlationId = latestTarget;
    }
  };

  // Poll on the worker finding the final correlation id
  int timeout = kCorrelationFlushPollLimit;
  int flushPolls = 0;
  while ((s_flush.maxCompletedCorrelationId_ < correlationId) && --timeout) {
    ++flushPolls;
    flushOnce();
  }

  const auto completedAfterFlush = s_flush.maxCompletedCorrelationId_;
  const bool flushTimedOut = completedAfterFlush < correlationId;
  if (flushTimedOut) {
    LOG(WARNING) << "ROCtracer stopLogging flush timed out: polls="
                 << flushPolls << ", target_correlation=" << correlationId
                 << ", completed_correlation=" << completedAfterFlush
                 << ", rows_buffered=" << countRows()
                 << ", during_stop_callback_stats={batches="
                 << debugState_.stopPhaseCallbackBatches.load()
                 << ", rows=" << debugState_.stopPhaseCallbackRows.load()
                 << "}";
  } else {
    LOG(INFO) << "ROCtracer stopLogging flush completed: polls=" << flushPolls
              << ", target_correlation=" << correlationId
              << ", completed_correlation=" << completedAfterFlush
              << ", rows_buffered=" << countRows()
              << ", during_stop_callback_stats={batches="
              << debugState_.stopPhaseCallbackBatches.load()
              << ", rows=" << debugState_.stopPhaseCallbackRows.load()
              << "}";
  }

  int stabilizationPolls = 0;
  int stableFlushPolls = 0;
  bool stabilizationTimedOut = false;
  if (!flushTimedOut) {
    // ROCm 7.2 can deliver multiple async batches for the same correlation id
    // after maxCompletedCorrelationId_ has already caught up with the target.
    // Keep flushing until callback/row growth stays stable for several polls.
    const auto stabilizationStart = steady_clock::now();
    size_t stableRows = countRows();
    uint64_t stableCallbackRows = debugState_.activityCallbackRows.load();
    uint64_t stableStopPhaseRows = debugState_.stopPhaseCallbackRows.load();
    while (stabilizationPolls < kStabilizationFlushPollLimit) {
      ++stabilizationPolls;
      flushOnce();
      const auto currentCompleted = s_flush.maxCompletedCorrelationId_;
      const auto currentRows = countRows();
      const auto currentCallbackRows =
          debugState_.activityCallbackRows.load();
      const auto currentStopPhaseRows =
          debugState_.stopPhaseCallbackRows.load();
      const bool stateStable = currentCompleted >= correlationId &&
          currentRows == stableRows &&
          currentCallbackRows == stableCallbackRows &&
          currentStopPhaseRows == stableStopPhaseRows;
      if (stateStable) {
        ++stableFlushPolls;
      } else {
        stableFlushPolls = 0;
        stableRows = currentRows;
        stableCallbackRows = currentCallbackRows;
        stableStopPhaseRows = currentStopPhaseRows;
      }
      const bool minDrainElapsed =
          (steady_clock::now() - stabilizationStart) >=
          kMinAdditionalDrainDuration;
      if (minDrainElapsed &&
          stableFlushPolls >= kRequiredStableFlushPolls) {
        break;
      }
    }
    const bool minDrainElapsed =
        (steady_clock::now() - stabilizationStart) >=
        kMinAdditionalDrainDuration;
    stabilizationTimedOut =
        !minDrainElapsed || stableFlushPolls < kRequiredStableFlushPolls;
    const auto completedAfterStabilization = s_flush.maxCompletedCorrelationId_;
    const auto rowsAfterStabilization = countRows();
    if (stabilizationTimedOut) {
      LOG(WARNING)
          << "ROCtracer stopLogging stabilization timed out: polls="
          << stabilizationPolls << ", stable_polls=" << stableFlushPolls
          << ", required_stable_polls=" << kRequiredStableFlushPolls
          << ", min_drain_ms=" << kMinAdditionalDrainDuration.count()
          << ", target_correlation=" << correlationId
          << ", completed_correlation=" << completedAfterStabilization
          << ", rows_buffered=" << rowsAfterStabilization
          << ", callback_stats={total_rows="
          << debugState_.activityCallbackRows.load()
          << ", during_stop_rows="
          << debugState_.stopPhaseCallbackRows.load() << "}";
    } else {
      LOG(INFO) << "ROCtracer stopLogging stabilization completed: polls="
                << stabilizationPolls
                << ", stable_polls=" << stableFlushPolls
                << ", required_stable_polls=" << kRequiredStableFlushPolls
                << ", min_drain_ms=" << kMinAdditionalDrainDuration.count()
                << ", target_correlation=" << correlationId
                << ", completed_correlation=" << completedAfterStabilization
                << ", rows_buffered=" << rowsAfterStabilization
                << ", callback_stats={total_rows="
                << debugState_.activityCallbackRows.load()
                << ", during_stop_rows="
                << debugState_.stopPhaseCallbackRows.load() << "}";
    }
  }

  debugState_.roctracerStopIssued.store(true);
  roctracer_stop();
  LOG(INFO) << "ROCtracer stopLogging end: rows_buffered=" << countRows()
            << ", external_correlations=" << countExternalCorrelations()
            << ", flush_state={target=" << correlationId
            << ", completed=" << s_flush.maxCompletedCorrelationId_ << "}"
            << ", drain_state={flush_timed_out=" << flushTimedOut
            << ", stabilization_timed_out=" << stabilizationTimedOut
            << ", stabilization_polls=" << stabilizationPolls
            << ", stable_polls=" << stableFlushPolls << "}"
            << ", callback_stats={total_batches="
            << debugState_.activityCallbackBatches.load()
            << ", total_rows=" << debugState_.activityCallbackRows.load()
            << ", during_stop_batches="
            << debugState_.stopPhaseCallbackBatches.load()
            << ", during_stop_rows="
            << debugState_.stopPhaseCallbackRows.load()
            << ", after_stop_batches="
            << debugState_.postStopCallbackBatches.load()
            << ", after_stop_rows="
            << debugState_.postStopCallbackRows.load()
            << ", dropped_rows="
            << debugState_.droppedRowsByBufferLimit.load() << "}";
  if (trackedGraphLaunches_.load() > 0) {
    std::lock_guard<std::mutex> graphLock(graphLaunchDebugMutex_);
    for (const auto& event : recentGraphLaunches_) {
      LOG(INFO) << "ROCtracer hipGraphLaunch summary: seq=" << event.sequence
                << ", correlation=" << event.correlationId
                << ", api_window=[" << event.apiBegin << ", " << event.apiEnd
                << "]"
                << ", rows_buffered_at_api_exit="
                << event.rowsBufferedAtApiExit
                << ", callback_snapshot={batches="
                << event.activityBatchesAtApiExit << ", rows="
                << event.activityRowsAtApiExit << ", stop_rows="
                << event.stopPhaseRowsAtApiExit << "}"
                << ", batches_after_api={total=" << event.asyncBatchesAfterApi
                << ", stop=" << event.stopPhaseBatchesAfterApi
                << ", corr_cover=" << event.correlationCoverBatches << "}"
                << ", rows_after_api={total=" << event.asyncRowsAfterApi
                << ", stop=" << event.stopPhaseRowsAfterApi
                << ", corr_cover=" << event.correlationCoverRows << "}"
                << ", last_corr_cover_batch="
                << event.lastCorrelationCoverBatchIndex
                << ", last_corr_range=[" << event.lastCorrelationCoverMinId
                << ", " << event.lastCorrelationCoverMaxId << "]";
    }
  }
}

void RoctracerLogger::endTracing() {
  if (registered_ == true) {
    roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API);
    // roctracer_disable_domain_callback(ACTIVITY_DOMAIN_ROCTX);

    roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HCC_OPS);
    roctracer_close_pool_expl(hccPool_);
    hccPool_ = nullptr;
  }
}

ApiIdList::ApiIdList() : invert_(true) {}

void ApiIdList::add(const std::string& apiName) {
  uint32_t cid = 0;
  if (roctracer_op_code(
          ACTIVITY_DOMAIN_HIP_API, apiName.c_str(), &cid, nullptr) ==
      ROCTRACER_STATUS_SUCCESS) {
    filter_[cid] = 1;
  }
}
void ApiIdList::remove(const std::string& apiName) {
  uint32_t cid = 0;
  if (roctracer_op_code(
          ACTIVITY_DOMAIN_HIP_API, apiName.c_str(), &cid, nullptr) ==
      ROCTRACER_STATUS_SUCCESS) {
    filter_.erase(cid);
  }
}

bool ApiIdList::loadUserPrefs() {
  // placeholder
  return false;
}
bool ApiIdList::contains(uint32_t apiId) {
  return (filter_.find(apiId) != filter_.end()) ? !invert_ : invert_; // XOR
}
