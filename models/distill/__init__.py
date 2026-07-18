from .methods.base import DistillMethod, DistillMethodOutput
from .methods.components.losses import LogitsLoss, CELoss, HiddenLoss, LossOutput
from .methods.components.modules import create_layer_mapping, MultiLayerAdaptationLayer
from .methods.featkd import FeatKDMethod
from .methods.kd import KDMethod
from .methods.seqkd import SeqKDMethod
from .plugins.base import Plugin, Scope, PluginConfig
from .plugins.contrastive import (
    ContrastivePlugin,
    ContrastivePluginConfig,
    ProjectionMLP,
    ContrastiveMemoryBank,
    PersonaContrastiveLoss,
)
from .plugins.curriculum import CurriculumPlugin, CurriculumPluginConfig
from .plugins.uncertainty import UncertaintyPlugin, UncertaintyPluginConfig

__all__ = [
    "DistillMethod",
    "SeqKDMethod",
    "KDMethod",
    "FeatKDMethod",
    "DistillMethodOutput",
    "LossOutput",
    "Plugin",
    "Scope",
    "PluginConfig",
    "ContrastivePluginConfig",
    "CurriculumPluginConfig",
    "ContrastivePlugin",
    "CurriculumPlugin",
    "UncertaintyPlugin",
    "UncertaintyPluginConfig",
    "ProjectionMLP",
    "ContrastiveMemoryBank",
    "PersonaContrastiveLoss",
    "create_layer_mapping",
    "MultiLayerAdaptationLayer",
    "LogitsLoss",
    "CELoss",
    "HiddenLoss",
    "LossOutput",
]
