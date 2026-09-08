#pragma once

#include <unordered_map>
#include <condition_variable>
#include <mutex>

#include "Vcd.hh"
#include "ffrAPI.h"
#include "Types.hh"

namespace sta {

class StaState;

Vcd
readFsdbFile(const char *file_name,
            StaState *sta,
            PeriodVal clk_period);

class FsdbReader : public StaState
{
public:
  FsdbReader(StaState *sta, PeriodVal clk_period);
  Vcd read(const char *file_name);

private:
  void getFsdbInfo();
  void readFsdbHeader();
  static bool_T scopeTreeCallback(fsdbTreeCBType cb_type, void *client_data, void *tree_cb_data);
  static void parseVar(fsdbTreeCBDataVar *var);
  static bool_T parseScope(fsdbTreeCBDataScope* scope);
  void setTimeUnit(const string &time_unit);
  void setMaxTime(const fsdbXTag& max_time_tag, const fsdbXTagType x_tag_type);
  void readVarChanges(ffrObject* ffr_obj, NVarVal start_var_id, NVarVal end_var_id, std::vector<NEeventVal>& cyc_idx_to_n_event);  // NOTE the var id start from 1 here, we iterate vars in range [start_var_id, end_var_id]
  void timeSlicing(const std::vector<NEeventVal>& cyc_idx_to_n_event) const;
  // NPeriodVal getCycleIntervalBoundaryByCycleIdxToAccuEvents(NEeventVal target_accu_event_count, const std::vector<NEeventVal>& cycle_idx_to_accu_n_event) const;
  void closeFfrObj(ffrObject* ffr_obj);
  VcdTime getValueChangeTime(const fsdbTag64& tag);
  VcdTime getValueChangeTime(const fsdbHLTag& tag);

  PeriodVal clk_period_;
  NPeriodVal n_cycle_;
  VcdTime vcd_time_unit_per_cycle_;
  inline static std::unordered_map<NVarVal, std::vector<std::string>> var_id_to_names_map_;
  inline static std::vector<std::string> scopes_;
  std::string file_name_;
  NVarVal n_var_;
  Vcd *vcd_;
  std::mutex make_var_mtx_;
  std::condition_variable make_var_cv_;
  int n_make_var_ready_thread_;
  bool all_threads_make_var_ready_;
};

}  // namespace sta