from torch import nn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import retinanet_resnet50_fpn


class Retina_ResNet50_FPB(nn.Module):
    def __init__(self):
        super().__init__()
        model = retinanet_resnet50_fpn(weights_backbone=ResNet50_Weights.DEFAULT,
                                       num_classes=1,
                                       trainable_backbone_layers=1)

