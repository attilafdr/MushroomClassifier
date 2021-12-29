"""
Torch Model Archiver requires the model to be described in a separate file as a class.
resnet18 imported from torchvision.models is a function that returns a ResNet object instance,
 which is not good enough for the model archiver.
To keep support for pretrained torchvision models, the state dict can be loaded.
"""

from torch.nn import Linear
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet, BasicBlock


class Resnet18Classifier(ResNet):
    """Classifier model class definition"""
    def __init__(self, n_classes, pretrained=False):
        super(Resnet18Classifier, self).__init__(BasicBlock, [2, 2, 2, 2])
        if pretrained:
            """Load the pretrained resnet18 state dict if requested"""
            self.load_state_dict(resnet18(pretrained=pretrained).state_dict())
        self.fc = Linear(self.fc.in_features, n_classes)
