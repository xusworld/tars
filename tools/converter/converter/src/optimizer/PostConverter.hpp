#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <fstream>
#include <memory>
#include <sstream>

#include "ace/schema/ace_generated.h"
#include "config.hpp"
#include "flatbuffers/idl.h"
#include "flatbuffers/minireflect.h"
#include "flatbuffers/util.h"

std::unique_ptr<ace::NetT> optimizeNet(std::unique_ptr<ace::NetT>& netT,
                                       bool forTraining, modelConfig& config);

#endif  // OPTIMIZER_HPP
