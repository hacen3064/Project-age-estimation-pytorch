import torch.nn as nn
import pretrainedmodels
import pretrainedmodels.utils
#from torchvision.models.vgg import vgg16_bn
#import torchvision.models as models


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


def main():
    model = get_model()
    print(model)


if __name__ == '__main__':
    main()
