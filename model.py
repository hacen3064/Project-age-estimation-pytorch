import torch.nn as nn
import pretrainedmodels
import pretrainedmodels.utils
from torchvision.models.vgg import vgg16_bn
import torchvision.models as models
from pretrainedmodels.models.senet import  (se_resnext50_32x4d, SENet, 
                                            SEResNeXtBottleneck, pretrained_settings,
                                            initialize_pretrained_model)


def get_model(model_name="se_resnext50_32x4d", num_classes=101, pretrained="imagenet"):
    model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
    dim_feats = model.last_linear.in_features
    model.last_linear = nn.Linear(dim_feats, num_classes)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    return model

def get_vgg():
    
    model = models.resnet18(pretrained=True)
    dim_feats = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(dim_feats, 101)
    return model

def get_resnet():
    
    model = models.resnet18(pretrained=True)
    dim_feats = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(dim_feats, 101)
    return model

class MySeNet(SENet):
    def __init__(self, num_classes=1000, pretrained='imagenet'):
        super(MySeNet, self).__init__(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
        if pretrained is not None:
            settings = pretrained_settings['se_resnext50_32x4d'][pretrained]
            initialize_pretrained_model(self, num_classes, settings)

        dim_feats = self.last_linear.in_features
        self.gender_layer = nn.Linear(dim_feats, 2)
        self.race_layer = nn.Linear(dim_feats, 3)
        self.last_linear = nn.Linear(dim_feats, 101)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
            
    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        age = self.last_linear(x)
        gender = self.gender_layer(x)
        race = self.race_layer(x)
        return age , gender, race


def main():
    model = get_model()
    print(model)


if __name__ == '__main__':
    main()
