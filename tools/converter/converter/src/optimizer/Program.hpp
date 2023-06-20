#ifndef Program_hpp
#define Program_hpp

#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>

#include "express/Expr.hpp"

namespace ace {
namespace Express {

class Program {
 public:
  static std::shared_ptr<Program> create(const ace::NetT* net,
                                         bool supportExtra);
  std::vector<VARP> outputs() const { return mOutputs; }
  void removeDeadNodes();

  void input(const std::unordered_map<std::string, VARP>& inputs);
  static void createUnit(std::map<int, VARP>& varMap,
                         std::vector<int>& inputIndexes,
                         const std::vector<std::unique_ptr<OpT>>& oplists,
                         ace::OpT* op, const ace::NetT* net,
                         std::set<OpT*>& invalidSet,
                         std::set<int>& extraInputIndexes);

  const std::map<int, VARP>& vars() const { return mVars; }

 private:
  Program() {}
  std::map<int, VARP> mVars;
  std::vector<VARP> mOutputs;
};

}  // namespace Express
};  // namespace ace

#endif
