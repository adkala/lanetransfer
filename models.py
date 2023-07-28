import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.models import vgg16, VGG16_Weights

def _replaceLayer(x):
    if isinstance(x, nn.MaxPool2d):
        return nn.Upsample(scale_factor=2)
    elif isinstance(x, nn.Conv2d):
        return nn.Conv2d(x.out_channels, x.in_channels, kernel_size=x.kernel_size, stride=x.stride, padding=x.padding)
    else:
        return x

class SymVGG16(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        f = vgg16(weights = VGG16_Weights.DEFAULT if pretrained else None).features[:-1]
        r = [_replaceLayer(x) for x in f[-2:1:-1]]
             
        self.m = nn.Sequential(*f, *r) # symmetrically concatenated vgg
        
        self.fc = nn.Sequential( # per-pixel MLP layers (64 -> 128 -> 3)
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.m(x)
        x = self.fc(x)
        return F.softmax(x, dim=1)