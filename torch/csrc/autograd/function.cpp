#include <torch/csrc/autograd/function.h>

#include <c10/util/ThreadLocal.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/variable.h>

#include <ATen/ATen.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static int32_t global_fn_uid_count = 0;

namespace torch { namespace autograd {

// The current evaluating node. This is useful to assign the current node as a
// parent of new nodes created during the evaluation of this node in anomaly
// mode.
C10_DEFINE_TLS_static(std::shared_ptr<Node>, tls_current_evaluating_node);
#define current_evaluating_node (tls_current_evaluating_node.get())

NodeGuard::NodeGuard(std::shared_ptr<Node> node) {
  last_evaluating_node_ = std::move(current_evaluating_node);
  current_evaluating_node = std::move(node);
}
NodeGuard::~NodeGuard() {
  // restore the previous evaluating node
  current_evaluating_node = std::move(last_evaluating_node_);
}

Node::Node(uint64_t sequence_nr)
      : sequence_nr_(sequence_nr) {
  // set id global
  next_edges_ = edge_list();

  self_id_ = global_fn_uid_count++;

  for (const Edge& edge: next_edges_) {
    update_topological_nr(edge);
  }

  if (AnomalyMode::is_enabled()) {
    metadata()->store_stack();

    // If anomaly mode is enabled and graph is constructed, then assign the
    // currently evaluating node as the parent of this node.
    // A parent is a Node where this Node is created.
    // We are tracking the parents to track multiple backward operations.
    assign_parent();
  }

  // Store the thread_id of the forward operator.
  // See NOTE [ Sequence Numbers ]
  thread_id_ = at::RecordFunction::currentThreadId();
}

Node::Node(
      uint64_t sequence_nr,
      edge_list&& next_edges)
      : sequence_nr_(sequence_nr),
      next_edges_(std::move(next_edges)) {
  // set id global
  self_id_ = global_fn_uid_count++;

  for (const Edge& edge: next_edges_) {
    update_topological_nr(edge);
  }

  if (AnomalyMode::is_enabled()) {
    metadata()->store_stack();

    // If anomaly mode is enabled and graph is constructed, then assign the
    // currently evaluating node as the parent of this node.
    // A parent is a Node where this Node is created.
    // We are tracking the parents to track multiple backward operations.
    assign_parent();
  }

  // Store the thread_id of the forward operator.
  // See NOTE [ Sequence Numbers ]
  thread_id_ = at::RecordFunction::currentThreadId();
}

void Node::assign_parent() {
  metadata()->assign_parent(current_evaluating_node);
}

auto Node::name() const -> std::string {
  return c10::demangle(typeid(*this).name());
}

AnomalyMetadata* Node::metadata() noexcept {
  if (!anomaly_metadata_) {
    anomaly_metadata_ = Engine::get_default_engine().make_anomaly_metadata();
  }
  return anomaly_metadata_.get();
}

static void gatherFunctions(
    Node* func,
    std::vector<std::shared_ptr<Node>>& stack) {
  func->release_variables();

  for (auto& edge : func->next_edges()) {
    if (edge.function.use_count() == 1) {
      stack.emplace_back(std::move(edge.function));
    } else {
      edge.function.reset();
    }
  }
}

/*
  * Fix for #5534: prevent stack overflow on deletion of deep computation graph
  *
  * Sometimes one can end up with a very big computation graph of Nodes
  * and Edges. Each std::shared_ptr<Node> contains a list of Edge, and
  * each Edge contains a std::shared_ptr<Node>. Deleting a
  * std::shared_ptr<Node> can trigger the recursive deletion of other
  * std::shared_ptr<Node>'s: this can stack overflow if the graph
  * is deep enough. Here is an example of such a graph:
  *
  * shared_ptr<Node> -> Edge -> shared_ptr<Node> -> Edge -> ... -> shared_ptr<Node>
  *
  * The solution here is to detect when we are decrementing away the last
  * reference to a Node, and when doing so to buffer up the Node's
  * that will be recursively decremented.  We can then decrement (and free)
  * the original Node without causing a recursive cascade, before
  * draining the buffer applying the same behavior.  This is, in effect,
  * converting recursion to a loop, using a heap buffer in place of the
  * recursive call stack.
  */
void deleteNode(Node* function) {
  // To avoid stack overflow on large computational graphs,
  // we need to track reference decrementing and freeing
  // on the heap.
  function->release_variables();
  std::vector<std::shared_ptr<Node>> stack;
  gatherFunctions(function, stack);
  delete function;

  while (!stack.empty()) {
    auto func = std::move(stack.back());
    stack.pop_back();
    gatherFunctions(func.get(), stack);
    // Reference count is decremented on the loop backedge.
  }
}

}} // namespace torch::autograd
