#pragma once

#include <map>
#include <set>
#include <string>

namespace ace {

class OpCount {
 public:
  OpCount();
  ~OpCount();
  static OpCount* get();
  void insertOp(const std::string& framework, const std::string& name);
  const std::map<std::string, std::set<std::string>>& getMap();

 private:
  std::map<std::string, std::set<std::string>> mOps;
};

};  // namespace ace
