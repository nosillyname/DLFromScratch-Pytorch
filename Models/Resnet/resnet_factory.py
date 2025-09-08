from resnet import ResNet
from skip_blocks import ResidualBlockSmall,ResidualBlockLarge

def resnet18(num_classes):
    blocks = [2,2,2,2]
    channels = [64,128,256,512]
    in_channels = 3
    out_classes = num_classes
    block = ResidualBlockSmall
    model = ResNet(in_channels,out_classes,block,blocks,channels)
    return model

def resnet34(num_classes):
    blocks = [3,4,6,3]
    channels = [64,128,256,512]
    in_channels = 3
    out_classes = num_classes
    block = ResidualBlockSmall
    model = ResNet(in_channels,out_classes,block,blocks,channels)
    return model

def resnet50(num_classes):
    blocks = [3,4,6,3]
    channels = [64,128,256,512]
    in_channels = 3
    out_classes = num_classes
    block = ResidualBlockLarge
    model = ResNet(in_channels,out_classes,block,blocks,channels)
    return model

def resnet101(num_classes):
    blocks = [3,4,23,3]
    channels = [64,128,256,512]
    in_channels = 3
    out_classes = num_classes
    block = ResidualBlockLarge
    model = ResNet(in_channels,out_classes,block,blocks,channels)
    return model

def resnet152(num_classes):
    blocks = [3,8,36,3]
    channels = [64,128,256,512]
    in_channels = 3
    out_classes = num_classes
    block = ResidualBlockLarge
    model = ResNet(in_channels,out_classes,block,blocks,channels)
    return model


