from torchvision.models import resnet50, resnet101
import torch.nn as nn
from copy import deepcopy

class ResNet(nn.Module):
    def __init__(self, model_cfg) :
        super(ResNet, self).__init__()

        pretrained = model_cfg["init_cfg"]["pretrained"]
        if model_cfg['depth'] == 50:
            self.resnet = resnet50(pretrained=pretrained)
        elif model_cfg['depth'] == 101:
            self.resnet = resnet101(pretrained=pretrained)

        self.num_stage = model_cfg["num_stages"]
        self.out_indices = model_cfg["out_indices"]

        assert len(self.out_indices) <= self.num_stage
        # TODO: add freeze_stages later

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        outs = []
        for stage in range(self.num_stage):
            x = getattr(self.resnet, f"layer{stage+1}")(x)
            if stage in self.out_indices:
                outs.append(deepcopy(x))
        return tuple(outs)