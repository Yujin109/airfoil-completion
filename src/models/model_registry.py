from typing import Dict

import torch.nn as nn

from .model_250416_Baseline import ConditionalUNet as BaselineConditionalUNet
from .model_250418_Baseline_CFG_zero import (
    ConditionalUNet as BaselineConditionalUNet_CFG_zero,
)
from .model_250422_Baseline_ResUNet_2layer_CFG_none import (
    ConditionalResidualUNet as BaselineResUNet_2layer_CFG_none,
)
from .model_250423_Baseline_ResUNet_3layer_CFG_none import (
    ConditionalResidualUNet as BaselineResUNet_3layer_CFG_none,
)
from .model_250425_NewBase_ResUNet_3layer import (
    ConditionalResidualUNet_PosEnc as NewBaseResUNet_3layer,
)
from .model_250426_NewBase_CFG_zero import (
    ConditionalUNet_PosEnc as NewBaseConditionalUNet_CFG_zero,
)
from .model_250430_NewBase_CFG_none import (
    ConditionalUNet_PosEnc as NewBaseConditionalUNet_CFG_none,
)
from .model_250506_NewBase_ResUNet_3layer_CFG_zero import (
    ConditionalResidualUNet_PosEnc as NewBaseResUNet_3layer_CFG_zero,
)
from .model_250626_NewBase_ResUNet_3layer_CFG_zero_widekernel import (
    ConditionalResidualUNet_PosEnc as NewBaseResUNet_3layer_CFG_zero_widekernel,
)
from .model_250628_NewBase_ResUNet_3layer_CFG_zero_widekernel2 import (
    ConditionalResidualUNet_PosEnc as NewBaseResUNet_3layer_CFG_zero_widekernel_2,
)
from .model_25041819_Baseline_CFG_none import (
    ConditionalUNet as BaselineConditionalUNet_CFG_none,
)
from .model_25042224_NewBase import ConditionalUNet_PosEnc as NewBaseConditionalUNet

MODEL_REGISTRY: Dict[str, nn.Module] = {
    "BaselineConditionalUNet": BaselineConditionalUNet,
    "BaselineConditionalUNet_CFG_zero": BaselineConditionalUNet_CFG_zero,
    "BaselineConditionalUNet_CFG_none": BaselineConditionalUNet_CFG_none,
    "BaselineResUNet_2layer_CFG_none": BaselineResUNet_2layer_CFG_none,
    "BaselineResUNet_3layer_CFG_none": BaselineResUNet_3layer_CFG_none,
    "NewBaseConditionalUNet": NewBaseConditionalUNet,
    "NewBaseResUNet_3layer": NewBaseResUNet_3layer,
    "NewBaseConditionalUNet_CFG_zero": NewBaseConditionalUNet_CFG_zero,
    "NewBaseConditionalUNet_CFG_none": NewBaseConditionalUNet_CFG_none,
    "NewBaseResUNet_3layer_CFG_zero": NewBaseResUNet_3layer_CFG_zero,
    "NewBaseResUNet_3layer_CFG_zero_widekernel": NewBaseResUNet_3layer_CFG_zero_widekernel,
    "NewBaseResUNet_3layer_CFG_zero_widekernel_2": NewBaseResUNet_3layer_CFG_zero_widekernel_2,
}
