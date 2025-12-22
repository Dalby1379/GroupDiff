# Copyright © 2025, Adobe Inc. and its licensors. 
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

# --------------------------------------------------------------------------
# model size configurations
SIZE_DICT = {
    "small": {"width": 512, "layers": 8, "heads": 8},
    "base": {"width": 768, "layers": 12, "heads": 12},
    "large": {"width": 1024, "layers": 24, "heads": 16},
    "xl": {"width": 1152, "layers": 28, "heads": 16},
    "huge": {"width": 1280, "layers": 32, "heads": 16},
}