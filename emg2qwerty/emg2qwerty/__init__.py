# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import types
from emg2qwerty import lr_scheduler

# Backward-compatibility alias for pl_bolts scheduler
if "pl_bolts" not in sys.modules or not hasattr(sys.modules.get("pl_bolts"), "optimizers"):
    b = types.ModuleType("pl_bolts")
    b_opt = types.ModuleType("pl_bolts.optimizers")
    b_opt.lr_scheduler = lr_scheduler
    b.optimizers = b_opt
    sys.modules["pl_bolts"] = b
    sys.modules["pl_bolts.optimizers"] = b_opt
    sys.modules["pl_bolts.optimizers.lr_scheduler"] = lr_scheduler

