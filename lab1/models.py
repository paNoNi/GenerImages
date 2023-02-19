from torchvision.models import resnet50, ResNet50_Weights, \
    convnext, ConvNeXt_Base_Weights
import torch


class FruitsModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model
        self._out_feature = 131
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x


class ResNet50(FruitsModel):
    def __init__(self):
        super().__init__(model=resnet50(weights=ResNet50_Weights.IMAGENET1K_V1))
        self.backbone.fc = torch.nn.Linear(in_features=self.backbone.fc.in_features,
                                           out_features=self._out_feature,
                                           bias=True)


class ConvNext(FruitsModel):
    def __init__(self):
        super().__init__(model=convnext.convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1))
        self.backbone.classifier[2] = torch.nn.Linear(self.backbone.classifier[2].in_features,
                                                      out_features=self._out_feature,
                                                      bias=True)

