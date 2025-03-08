from torchvision.models import resnet50, resnet101, resnet34
import torch.nn as nn
from copy import deepcopy
from ..builder import BACKBONES
from torch.nn import BatchNorm2d
import torch

@BACKBONES.register_module()
class ResNet(nn.Module):
    def __init__(self, depth, num_stages, out_indices, frozen_stages, norm_eval, init_cfg): 
        super(ResNet, self).__init__()

        pretrained = init_cfg['type'] == 'pretrained'
        if depth == 50:
            self.resnet = resnet50(pretrained=pretrained)
        elif depth == 101:
            self.resnet = resnet101(pretrained=pretrained)
        elif depth == 34:
            self.resnet =  resnet34(pretrained=pretrained)

        self.num_stage = num_stages
        self.out_indices = out_indices

        if init_cfg is not None:
            ckpt = init_cfg.get('checkpoint', None)
            if ckpt is not None:
                try:
                    self.resnet.load_state_dict(torch.load(ckpt))
                except:
                    Warning(f"Failed to load checkpoint from {ckpt}")

        assert len(self.out_indices) <= self.num_stage
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.resnet.bn1.eval()
            for m in [self.resnet.conv1, self.resnet.bn1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self.resnet, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.resnet.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, BatchNorm2d):
                    m.eval()


    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        outs = []
        for stage in range(self.num_stage):
            x = getattr(self.resnet, f"layer{stage+1}")(x)
            if stage in self.out_indices:
                outs.append(x)
        return tuple(outs)