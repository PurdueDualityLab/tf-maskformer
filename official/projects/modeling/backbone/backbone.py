from typing import Optional

import tensorflow as tf
from official.vision.modeling.backbones import resnet

def build_maskformer_backbone(
      model_id: int = 50,):
    return resnet.ResNet(model_id)