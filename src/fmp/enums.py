from enum import Enum


class GenerativeModels(Enum):
    VAE = "vae"
    FLOW_MATCHING = "flow_matching"


class FlowMatchingTypes(Enum):
    SIMPLE = "simple"
    UNET = "unet"
    TRANSFORMER = "transformer"
