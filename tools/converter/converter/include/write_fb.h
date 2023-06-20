#pragma once

#include <map>

#include "ace_generated.h"
#include "config.h"

int writeFb(std::unique_ptr<ace::NetT>& netT, const std::string& MNNModelFile,
            const AceModelConfig& config);
