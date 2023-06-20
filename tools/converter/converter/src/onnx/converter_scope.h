#pragma once

#include <map>
#include <memory>
#include <vector>

#include "ace/schema/ace_generated.h"

namespace ace {
namespace converter {

class ConverterScope {
 public:
  ConverterScope();
  ConverterScope(ace::NetT* net);
  ConverterScope(ace::SubGraphProtoT* subnet, ace::NetT* parentNet,
                 ConverterScope* parentScope);

 public:
  // declare a tensor in this scope, get it's idx
  int declareTensor(std::string name);
  // lookup tensor idx by name in this scope
  virtual int lookupTensor(std::string name) {
    const auto iter = mTensorIdx.find(name);
    if (iter != mTensorIdx.end()) {
      return iter->second;
    }
    return -1;
  }
  // lookup tensor name by idx in this scope
  std::string lookupTensorByIdx(int idx);
  // build an input op in this scope
  int buildIntInputOp(std::string name);
  // build an int const op in this scope
  int buildIntConstOp(std::vector<int> data, std::string name);
  // add input tensor for op, add depend in parent scope
  void addInputForOp(ace::OpT* op, std::string inputName,
                     bool allowSameInput = false);
  // deal with subgraph input depend
  void dealSubgraphDeps();
  void dealSubgraphDepsForOp(ace::OpT* op);
  // build a cond subgraph for while_op in this scope
  // name: subgraph name; iName: i; mName: M; kName: keep_going
  // cond_graph is : for (int i = 0; i < M && keep_going; i++)
  void buildCondGraph(const std::string& name, const std::string& iName,
                      const std::string& mName, const std::string& kName);
  // build increment instruction for while_op body
  // name: subgraph name; iName: i;
  // increment is : i = i + 1
  void buildIncrement(std::string name, std::string iName);
  // get tensors
  std::vector<std::string>& tensors();
  // get oplists
  std::vector<std::unique_ptr<ace::OpT>>& oplists();
  // get deps
  std::vector<std::string>& deps();

 protected:
  std::map<std::string, int> mTensorIdx;
  ace::NetT* mNet;
  ace::SubGraphProtoT* mSubNet;
  ConverterScope* mParent;
  std::vector<std::string> subgraphDeps;
};

}  // namespace converter
}  // namespace ace