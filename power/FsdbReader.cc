#include <cmath>

#include "FsdbReader.hh"
#include "GlobalConfig.hh"
#include "ScopedTimer.hh"
#include "ThreadPool.hh"
#include "StringUtil.hh"
#include "PowerUtils.hh"

namespace sta {

FsdbReader::FsdbReader(StaState* sta, PeriodVal clk_period) :
  StaState(sta),
  clk_period_(clk_period),
  n_cycle_(0),
  vcd_time_unit_per_cycle_(0),
  file_name_(),
  n_var_(0),
  vcd_(nullptr),
  n_make_var_ready_thread_(0),
  all_threads_make_var_ready_(false)
{
}

Vcd
FsdbReader::read(const char* file_name)
{
  file_name_ = file_name;
  if (!ffrObject::ffrIsFSDB(file_name_.data())) {
    LOG_ERROR << "File " << file_name_ << " is not a valid FSDB file.";
  }
  Vcd vcd(this);
  vcd_ = &vcd;
  getFsdbInfo();
  readFsdbHeader();

  utils::ScopedTimer timer_read_fsdb("Read FSDB");
  std::vector<NEeventVal> cyc_idx_to_n_event(n_cycle_, 0);
  if (G_CONFIG.flags.enable_multi_threaded_cpu) {
    LOG_INFO << "Starting multi threaded FSDB parser...";
    // --------------------------------init--------------------------------------
    std::vector<ffrObject*> ffr_objs;
    for (int t_id = 0; t_id < G_CONFIG.nums.multi_thread_number; ++t_id) {
      ffrObject* ffr_obj = ffrObject::ffrOpenNonSharedObj(file_name_.data());
      if (NULL == ffr_obj) {
        LOG_ERROR << "FSDB file " << file_name << " is unable to be opened.";
      }
      ffr_objs.push_back(ffr_obj);
    }
    // --------------------------------end of init--------------------------------------

    // --------------------------------launch threads--------------------------------------
    NVarVal base = n_var_ / G_CONFIG.nums.multi_thread_number;
    NVarVal remainder = n_var_ % G_CONFIG.nums.multi_thread_number;
    ThreadPool thread_pool(G_CONFIG.nums.multi_thread_number);
    std::vector<std::future<void>> job_futures;
    std::vector<std::vector<NEeventVal>> per_threads_cyc_idx_to_n_event(G_CONFIG.nums.multi_thread_number, std::vector<NEeventVal>(n_cycle_, 0));
    NVarVal start = 1;
    for (int t_id = 0; t_id < G_CONFIG.nums.multi_thread_number; ++t_id) {
      ffrObject* current_thread_ffr_obj = ffr_objs.at(t_id);
      auto& cur_thread_cyc_idx_to_n_event = per_threads_cyc_idx_to_n_event.at(t_id);
      NVarVal end = start + base - 1;
      if (t_id < remainder) {
        end += 1;
      }
      LOG_INFO << "Thread " << t_id << " handles vars in range[" << start << ", " << end << "].";
      job_futures.push_back(thread_pool.enqueue(
        [current_thread_ffr_obj, start, end, &cur_thread_cyc_idx_to_n_event, this]() {
          readVarChanges(current_thread_ffr_obj, start, end, cur_thread_cyc_idx_to_n_event);
          closeFfrObj(current_thread_ffr_obj);
        }
      ));
      start = end + 1;
    }
    while (true) {  // wait until all threads make vars ready
      {
        std::lock_guard<std::mutex> lock(make_var_mtx_);
        if (n_make_var_ready_thread_ == G_CONFIG.nums.multi_thread_number) {
          all_threads_make_var_ready_ = true;
          break;
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    // All vars have been created by worker threads and they are waiting on make_var_cv_.
    // Sort deterministically before releasing them.
    vcd_->sortVarsById();
    make_var_cv_.notify_all();
    for (int i = 0; i < G_CONFIG.nums.multi_thread_number; ++i) {
      job_futures.at(i).wait();
    }
    // --------------------------------end of launch threads--------------------------------------

    // --------------------------------wrap up--------------------------------------
    // for (ffrObject* ffr_obj : ffr_objs) {
    //   closeFfrObj(ffr_obj);
    // }
    for (int t_id = 0; t_id < G_CONFIG.nums.multi_thread_number; ++t_id) {  // merge per_threads_cyc_idx_to_n_event
      for (NPeriodVal cyc_idx = 0; cyc_idx < n_cycle_; ++cyc_idx) {
        cyc_idx_to_n_event.at(cyc_idx) += per_threads_cyc_idx_to_n_event.at(t_id).at(cyc_idx);
      }
    }
    // --------------------------------end of wrap up--------------------------------------
  } else {
    ffrObject* fsdb_obj = ffrObject::ffrOpen3(file_name_.data());
    readVarChanges(fsdb_obj, 1, n_var_, cyc_idx_to_n_event);
    closeFfrObj(fsdb_obj);
  }
  timeSlicing(cyc_idx_to_n_event);

  return vcd;
}

bool_T
FsdbReader::scopeTreeCallback(fsdbTreeCBType cb_type, void* client_data, void* tree_cb_data)
{
  switch (cb_type) {
    case FSDB_TREE_CBT_BEGIN_TREE:
      break;

    case FSDB_TREE_CBT_SCOPE:
      parseScope((fsdbTreeCBDataScope*)tree_cb_data);
      break;

    case FSDB_TREE_CBT_VAR:
      parseVar((fsdbTreeCBDataVar*)tree_cb_data);
      break;

    case FSDB_TREE_CBT_UPSCOPE:
      scopes_.pop_back();
      break;

    case FSDB_TREE_CBT_END_TREE:
      break;

    case FSDB_TREE_CBT_FILE_TYPE:
      break;

    case FSDB_TREE_CBT_SIMULATOR_VERSION:
      break;

    case FSDB_TREE_CBT_SIMULATION_DATE:
      break;

    case FSDB_TREE_CBT_X_AXIS_SCALE:
      break;

    case FSDB_TREE_CBT_END_ALL_TREE:
      break;

    case FSDB_TREE_CBT_ARRAY_BEGIN:
      break;

    case FSDB_TREE_CBT_ARRAY_END:
      break;

    case FSDB_TREE_CBT_RECORD_BEGIN:
      break;

    case FSDB_TREE_CBT_RECORD_END:
      break;

    default:
      return 0;
  }

  return 1;
}

void
FsdbReader::parseVar(fsdbTreeCBDataVar* var)
{
  std::string var_name;
  for (const std::string& scope : scopes_) {
    var_name.append(scope);
    var_name.append("/");
  }
  std::string tmp_var_name(var->name);
  if (tmp_var_name.size() != 0 && tmp_var_name.at(0) == '\\') {
    tmp_var_name.erase(0, 1);  // remove leading \ to keep the same with verilog parser
  }
  replaceAll(tmp_var_name, "/", "\\/");  // convert / to \\/ to keep the same with verilog parser
  var_name.append(tmp_var_name);
  var_id_to_names_map_[var->u.idcode].push_back(var_name);
}

bool_T
FsdbReader::parseScope(fsdbTreeCBDataScope* scope)
{
  switch (scope->type) {
    case FSDB_ST_VCD_MODULE:
      scopes_.push_back(scope->name);
      break;
    case FSDB_ST_VCD_TASK:
      break;
    case FSDB_ST_VCD_FUNCTION:
      break;
    case FSDB_ST_VCD_BEGIN:
      break;
    case FSDB_ST_VCD_FORK:
      break;
    default:
      break;
  }

  return 1;
}

void
FsdbReader::readFsdbHeader()
{
  ffrObject* tmp_fsdb_obj = ffrObject::ffrOpen3(file_name_.data());
  tmp_fsdb_obj->ffrSetTreeCBFunc(scopeTreeCallback, NULL);
  tmp_fsdb_obj->ffrReadScopeVarTree();
  closeFfrObj(tmp_fsdb_obj);
}

void
FsdbReader::getFsdbInfo()
{
  ffrFSDBInfo fsdb_info;
  ffrObject::ffrGetFSDBInfo(file_name_.data(), fsdb_info);
  ffrObject* tmp_fsdb_obj = ffrObject::ffrOpen3(file_name_.data());
  fsdbXTagType x_tag_type = tmp_fsdb_obj->ffrGetXTagType();

  LOG_INFO << "unique_var_count: " << fsdb_info.unique_var_count << " total_var_count: " << fsdb_info.total_var_count  // unsigned long long
           << " scale_unit: " << fsdb_info.scale_unit
           << " x_tag_type: " << x_tag_type
           << " min_xtag: " << fsdb_info.min_xtag.hltag.L << " " << fsdb_info.min_xtag.hltag.H  // unsigned int
           << " max_xtag: " << fsdb_info.max_xtag.hltag.L << " " << fsdb_info.max_xtag.hltag.H;
  // if (fsdb_info.unique_var_count != fsdb_info.total_var_count) {
  //   LOG_ERROR << "unique_var_count != total_var_count unique_var_count: " << fsdb_info.unique_var_count << " total_var_count: " << fsdb_info.total_var_count;
  // }
  n_var_ = fsdb_info.unique_var_count;
  setTimeUnit(fsdb_info.scale_unit);
  setMaxTime(fsdb_info.max_xtag, x_tag_type);

  closeFfrObj(tmp_fsdb_obj);
}

void
FsdbReader::setTimeUnit(const string& time_unit)
{  // TODO distinguish time unit and time scale
  double time_unit_scale = 1.0;
  if (time_unit == "1fs")
    time_unit_scale = 1e-15;
  else if (time_unit == "1ps")
    time_unit_scale = 1e-12;
  else if (time_unit == "1ns")
    time_unit_scale = 1e-9;
  else
    LOG_ERROR << "Unknown timescale unit.";

  LOG_INFO << "Time unit scale in FSDB file: " << time_unit_scale;
  vcd_->setTimeUnit(time_unit, time_unit_scale);
  vcd_->setTimeScale(time_unit_scale);
}

void
FsdbReader::setMaxTime(const fsdbXTag& max_time_tag, const fsdbXTagType x_tag_type)
{
  if (x_tag_type == FSDB_XTAG_TYPE_HL) {
    VcdTime raw_max_time = getValueChangeTime(max_time_tag.hltag);
    vcd_time_unit_per_cycle_ = static_cast<VcdTime>(ceil(clk_period_ / vcd_->timeScale()));
    VcdTime max_time = raw_max_time;
    const VcdTime complete_cycle_max_time = (raw_max_time / vcd_time_unit_per_cycle_) * vcd_time_unit_per_cycle_;
    if (complete_cycle_max_time > 0) {
      max_time = complete_cycle_max_time;
    }
    vcd_->setTimeMax(max_time);
    n_cycle_ = (max_time + vcd_time_unit_per_cycle_ - 1) / vcd_time_unit_per_cycle_;
    LOG_INFO << "Max time in FSDB file: " << raw_max_time
             << " effective max time: " << max_time
             << " ignored tail time: " << (raw_max_time - max_time)
             << " vcd_time_unit_per_cycle: " << vcd_time_unit_per_cycle_
             << " n_cycle_: " << n_cycle_;
  }
  else {
    LOG_ERROR << "Not supported fsdbXTagType: " << x_tag_type;
  }
}

void
FsdbReader::readVarChanges(ffrObject* ffr_obj, NVarVal start_var_id, NVarVal end_var_id, std::vector<NEeventVal>& cyc_idx_to_n_event)
{  // NOTE the var id start from 1 here, we iterate vars in range [start_var_id, end_var_id]
  // --------------------------------init--------------------------------------
  // Each unique var is represented by a unique idcode in fsdb
  // file, these idcodes are positive integer and continuous from
  // the smallest to the biggest one. So the maximum idcode also
  // means that how many unique vars are there in this fsdb file.
  fsdbVarIdcode max_var_idcode = ffr_obj->ffrGetMaxVarIdcode();  // id codes range from 1 to max_var_idcode
  if (max_var_idcode != n_var_) {
    LOG_ERROR << "max_var_idcode != n_var_, max_var_idcode: " << max_var_idcode << " n_var_: " << n_var_;
  }
  // Read all vars
  // ffr_obj->ffrReadScopeVarTree();
  for (NVarVal var_id = start_var_id; var_id <= end_var_id; ++var_id) {  // [start_var_id, end_var_id]
    // Load vc
    ffr_obj->ffrAddToSignalList(var_id);
  }
  ffr_obj->ffrLoadSignals();
  char vc_buffer[FSDB_MAX_BIT_SIZE + 1];
  // --------------------------------end of init--------------------------------------

  // --------------------------------make var--------------------------------------
  // make var first to avoid conflicts in map
  for (NVarVal var_id = start_var_id; var_id <= end_var_id; ++var_id) {  // [start_var_id, end_var_id]
    ffrVCTrvsHdl vc_trvs_hdl = ffr_obj->ffrCreateVCTraverseHandle(var_id);
    if (NULL == vc_trvs_hdl) {
      LOG_ERROR << "Failed to to create a traverse handle for var id: " << var_id;
    }
    fsdbTag64 time;
    // Get the minimum time(xtag) where has value change.
    vc_trvs_hdl->ffrGetMinXTag((void*)&time);
    // Jump to the minimum time(xtag).
    vc_trvs_hdl->ffrGotoXTag((void*)&time);
    uint_T bit_width = vc_trvs_hdl->ffrGetBitSize();
    std::string var_id_str = std::to_string(var_id);

    if (var_id_to_names_map_.count(var_id) != 0) {  // TODO see why some ids not added to var_id_to_names_map_
      std::lock_guard<std::mutex> lock(make_var_mtx_);  // add lock to avoid conflict when making new var
      for (auto& var_name: var_id_to_names_map_.at(var_id)) {
        vcd_->makeVar(var_name, VcdVarType::wire, bit_width, var_id_str);  // TODO decide VcdVarType here
      }
    }

    vc_trvs_hdl->ffrFree();
  }
  if (G_CONFIG.flags.enable_multi_threaded_cpu) {
    {
      std::lock_guard<std::mutex> lock(make_var_mtx_);
      ++n_make_var_ready_thread_;
    }
    {  // sync all threads here
      std::unique_lock<std::mutex> lock(make_var_mtx_);
      make_var_cv_.wait(lock, [this]{ return all_threads_make_var_ready_; });
    }
  }
  // --------------------------------end of make var--------------------------------------

  // --------------------------------iteration--------------------------------------
  for (NVarVal var_id = start_var_id; var_id <= end_var_id; ++var_id) {  // [start_var_id, end_var_id]
    if (var_id_to_names_map_.count(var_id) == 0) {
      LOG_WARN << "var_id: " << var_id << " not in var_id_to_names_map_";
      continue;
    }

    ffrVCTrvsHdl vc_trvs_hdl = ffr_obj->ffrCreateVCTraverseHandle(var_id);
    if (NULL == vc_trvs_hdl) {
      LOG_ERROR << "Failed to to create a traverse handle for var id: " << var_id;
    }
    fsdbTag64 time;
    byte_T* vc_ptr;
    // Get the minimum time(xtag) where has value change.
    vc_trvs_hdl->ffrGetMinXTag((void*)&time);
    // Jump to the minimum time(xtag).
    vc_trvs_hdl->ffrGotoXTag((void*)&time);
    uint_T bit_width = vc_trvs_hdl->ffrGetBitSize();
    std::string var_id_str = std::to_string(var_id);

    do {
      vc_trvs_hdl->ffrGetXTag(&time);
      vc_trvs_hdl->ffrGetVC(&vc_ptr);
      VcdTime cur_time = getValueChangeTime(time);
      if (vcd_->timeMax() > 0 && cur_time >= vcd_->timeMax()) {
        break;
      }
      if (bit_width == 1) {
        char cur_vc;
        switch (vc_ptr[0]) {  // TODO consider use map instead
          case FSDB_BT_VCD_0:
            cur_vc = '0';
            break;
          case FSDB_BT_VCD_1:
            cur_vc = '1';
            break;
          case FSDB_BT_VCD_X:
            cur_vc = 'X';
            break;
          case FSDB_BT_VCD_Z:
            cur_vc = 'Z';
            break;
          default:
            cur_vc = '-';
            LOG_ERROR << "Unsupported bit (not 0/1/X): " << vc_ptr[0];
        }
        vcd_->varAppendValue(var_id_str, cur_time, cur_vc);  // TODO batch append instead one by one
      } else {
        if (vc_trvs_hdl->ffrGetBytesPerBit() != FSDB_BYTES_PER_BIT_1B) {
          LOG_ERROR << "Unsupported ffrGetBytesPerBit: " << vc_trvs_hdl->ffrGetBytesPerBit();
        }
        uint_T bit_i;
        for (bit_i = 0; bit_i < bit_width; ++bit_i) {
          switch (vc_ptr[bit_i]) {  // TODO consider use map instead
            case FSDB_BT_VCD_0:
              vc_buffer[bit_i] = '0';
              break;
            case FSDB_BT_VCD_1:
              vc_buffer[bit_i] = '1';
              break;
            case FSDB_BT_VCD_X:
            case FSDB_BT_VCD_Z:
            default:
              vc_buffer[bit_i] = '0';
              LOG_WARN << "Bus mixed 0/1/X/U not supported for " << var_id << ", use 0 instead.";
          }
        }
        vc_buffer[bit_i] = '\0';
        int64_t bus_value = strtol(vc_buffer, nullptr, 2);
        vcd_->varAppendBusValue(var_id_str, cur_time, bus_value, bit_width);  // TODO batch append instead one by one
      }
      cyc_idx_to_n_event.at(power::utils::clkedWaveformIdx(cur_time, vcd_->timeScale(), clk_period_)) += bit_width;
    } while (FSDB_RC_SUCCESS == vc_trvs_hdl->ffrGotoNextVC());
    vc_trvs_hdl->ffrFree();
  }
  // --------------------------------end of iteration--------------------------------------
}

// new slicing logic by linear scanning
void
FsdbReader::timeSlicing(const std::vector<NEeventVal>& cyc_idx_to_n_event) const
{
  std::vector<std::pair<VcdEventTime, VcdEventTime>> time_intervals;
  const NEeventVal max_events = static_cast<NEeventVal>(G_CONFIG.nums.max_event_num);

  // Defensive guards
  if (n_cycle_ == 0 || max_events <= 0) {
    LOG_ERROR << "n_cycle_ == 0 || max_events <= 0, n_cycle_: " << n_cycle_ << " max_events: " << max_events;
  }

  NPeriodVal start_cycle = 0;  // inclusive
  NEeventVal acc = 0;

  for (NPeriodVal cycle = 0; cycle < n_cycle_; ++cycle) {
    acc += cyc_idx_to_n_event.at(cycle);

    if (acc >= max_events) {
      const NPeriodVal end_cycle = cycle + 1;  // exclusive
      if (end_cycle > start_cycle) {
        time_intervals.emplace_back(
          static_cast<VcdEventTime>(start_cycle) * vcd_time_unit_per_cycle_,
          static_cast<VcdEventTime>(end_cycle) * vcd_time_unit_per_cycle_
        );
      }
      start_cycle = end_cycle;
      acc = 0;
    }
  }

  // Tail
  if (start_cycle < n_cycle_) {
    time_intervals.emplace_back(
      static_cast<VcdEventTime>(start_cycle) * vcd_time_unit_per_cycle_,
      static_cast<VcdEventTime>(n_cycle_) * vcd_time_unit_per_cycle_
    );
  }

  vcd_->setTimeIntervals(time_intervals);
}

void
FsdbReader::closeFfrObj(ffrObject* ffr_obj)
{
  static std::mutex close_mtx;          // 全进程共享
  std::lock_guard<std::mutex> lk(close_mtx);

  if (!ffr_obj) return;
  ffr_obj->ffrResetSignalList();
  ffr_obj->ffrUnloadSignals();
  ffr_obj->ffrClose();
}

VcdTime
FsdbReader::getValueChangeTime(const fsdbTag64& tag)
{
  return (static_cast<VcdTime>(tag.H) << 32) + tag.L;
}

VcdTime
FsdbReader::getValueChangeTime(const fsdbHLTag& tag)
{
  return (static_cast<VcdTime>(tag.H) << 32) + tag.L;
}

Vcd
readFsdbFile(const char* file_name,
             StaState* sta,
             PeriodVal clk_period)
{
  FsdbReader reader(sta, clk_period);
  return reader.read(file_name);
}

}  // namespace sta
