#include "./ScopedTimer.hh"

namespace utils {
// ---------timer----------
class GlobalTimeStats::Node {
  private:
  std::string task_;
  std::vector<Node*> children_;
  std::vector<double> wall_times_;
  std::vector<double> cpu_times_;

  public:
  // Node() {}
  Node(std::string val)
  {
    task_ = val;
  }

  const std::string& GetTask()
  {
    return task_;
  }

  std::vector<Node*>& GetChildren()
  {
    return children_;
  }

  std::vector<double>& GetCPUTimes()
  {
    return cpu_times_;
  }

  std::vector<double>& GetWallTimes()
  {
    return wall_times_;
  }

  void AppendChild(Node* child)
  {
    children_.push_back(child);
  }

  void AddWallTime(double wall_time)
  {
    wall_times_.push_back(wall_time);
  }

  void AddCPUTime(double cpu_time)
  {
    cpu_times_.push_back(cpu_time);
  }
};

thread_local std::vector<std::pair<std::string, std::pair<double, double>>> GlobalTimeStats::task_stack_; //  <task, <start_cpu_time, start_wall_time>>
thread_local GlobalTimeStats::Node* GlobalTimeStats::root_ = nullptr; // dummy root
thread_local std::string GlobalTimeStats::unit_;
thread_local size_t GlobalTimeStats::max_task_name_len_ = 0;

GlobalTimeStats::GlobalTimeStats(token) { 
  //root_ = new Node("root");
}

int GlobalTimeStats::Init() { return 0;}

GlobalTimeStats::~GlobalTimeStats()
{
}

void GlobalTimeStats::Print()
{
  if(!root_) return;
  SetUnit();
  LayerwiseSortAndResetTimeByUnit();
  std::cout << std::endl;
  std::cout << "---------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
  std::cout << std::setiosflags(std::ios::left) << std::setw(max_task_name_len_ + task_name_padding_) << "task" << std::setw(15) << "number of call"
    << std::setiosflags(std::ios::left) << std::setw(20) << ("total cpu time(" + unit_ + ")") << std::setw(20) << ("total wall time(" + unit_ + ")")
    << std::setiosflags(std::ios::left) << std::setw(20) << ("avg cpu time(" + unit_ + ")") << std::setw(20) << ("avg wall time(" + unit_ + ")") << std::endl;
  std::cout << "---------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
  PreOrderPrint(root_, 0);
  std::cout << "---------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
  std::cout << std::endl;
}

void GlobalTimeStats::StartTaskTiming(const std::string& task_name)
{
  task_stack_.push_back(std::make_pair(task_name, std::make_pair(utils::CpuTime(), utils::WallTime())));
}

void GlobalTimeStats::EndCurTaskTiming()
{
  const auto& start_time = task_stack_.back().second;
  AddRecord((utils::CpuTime() - start_time.first) * 1000, (utils::WallTime() - start_time.second) * 1000); // *1000 for ms

  task_stack_.pop_back();
}

void GlobalTimeStats::AddRecord(double cpu_time, double wall_time)
{
  if(!root_) root_ = new Node("root");
  Node* cur = root_;
  size_t n_level = task_stack_.size();

  for (size_t i = 0; i < n_level; ++i) {
    const auto& task = task_stack_[i].first;

    bool found = false;
    const auto& children = cur->GetChildren();

    for (const auto& node : children) {
      if (task == node->GetTask()) {
        found = true;
        cur = node;

        break;
      }
    }

    if (!found) {
      Node* new_node = new Node(task);
      cur->AppendChild(new_node);
      cur = new_node;
    }

    // add time to the task on the stack top
    if (i == n_level - 1) {
      cur->AddCPUTime(cpu_time);
      cur->AddWallTime(wall_time);
    }
  }
}

void GlobalTimeStats::SetUnit()
{
  double first_layer_total_time = 0.0;
  for (const auto& child: root_->GetChildren()) {
    first_layer_total_time += std::accumulate(child->GetCPUTimes().begin(), child->GetCPUTimes().end(), first_layer_total_time);
  }

  if (first_layer_total_time <= 1e5) {
    unit_ = "ms";
  } else if (first_layer_total_time <= 1e9) {
    unit_ = "s";
  } else {
    unit_ = "h";
  }
}

// TODO use macro to replace timer call
void GlobalTimeStats::LayerwiseSortAndResetTimeByUnit() 
{
  std::queue<Node*> q;
  q.push(root_);
  size_t cur_layer_rest = q.size();
  size_t depth = 0;  // 0 for no indentation 
  max_task_name_len_ = 0;

  while (!q.empty()) {
    Node* cur = q.front();
    q.pop();
    cur_layer_rest--;

    // get max task name length
    max_task_name_len_ = std::max(max_task_name_len_, cur->GetTask().size() + depth * indentation_.size());

    // layer traverse all nodes to reset time by unit
    double unit_scale = 1;
    if (unit_ == "s") {
      unit_scale = 1e-3;
    } else if (unit_ == "h") {
      unit_scale = 1e-3 / 3600.0;
    }
    for (double& time: cur->GetCPUTimes()) {
      time *= unit_scale;
    }
    for (double& time: cur->GetWallTimes()) {
      time *= unit_scale;
    }

    // sort time in descending order
    // std::sort(cur->GetChildren().begin(), cur->GetChildren().end(), [](Node* a, Node* b) {
    //   double accuA = std::accumulate(a->GetCPUTimes().begin(), a->GetCPUTimes().end(), 0.0);
    //   double accuB = std::accumulate(b->GetCPUTimes().begin(), b->GetCPUTimes().end(), 0.0);
    //   return accuA > accuB;
    // });

    for (const auto& child: cur->GetChildren()) {
      q.push(child);
    }

    // traverse to a new layer
    if (cur_layer_rest == 0) {
      cur_layer_rest = q.size();
      depth++;
    }
  }
}

void GlobalTimeStats::PreOrderPrint(Node* node, int depth)
{
  if (node == nullptr)
    return;

  const auto& cpu_times = node->GetCPUTimes();
  const auto& wall_times = node->GetWallTimes();
  double total_cpu_time = std::accumulate(cpu_times.begin(), cpu_times.end(), 0.0);
  double total_wall_time = std::accumulate(wall_times.begin(), wall_times.end(), 0.0);
 
  std::string task_name = node->GetTask();
  for (int i = 0; i < depth - 1; i++) {
    // LOG(TIMER) << "  ";
    task_name = "  " + task_name;
  }
  if (depth) {
    // if (depth > 1) {
      // LOG(TIMER) << "-";
    // }
    std::cout << std::setiosflags(std::ios::left) << std::setw(max_task_name_len_ + task_name_padding_) << task_name << std::setw(15) << cpu_times.size()
      << std::setiosflags(std::ios::left) << std::setw(20) << total_cpu_time << std::setw(20) << total_wall_time
      << std::setiosflags(std::ios::left) << std::setw(20) << total_cpu_time / cpu_times.size() << std::setw(20) << total_wall_time / wall_times.size() << std::endl;
  }

  const auto& children = node->GetChildren();
  for (const auto& child : children) {
    PreOrderPrint(child, depth + 1);
  }

  // only print "-" for layer 0
  // if (depth == 1) {
  //   LOG(TIMER) << "-----------------------";
  // }

  // finally we delete the node
  delete node;
}
// ---------timer----------
}