import torch.nn as nn
import torchvision.transforms as transforms

__all__ = ['alexnetvlr']

class AlexNetOWT_BN(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNetOWT_BN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2,
                      bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=5, padding=2, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192),
            nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),
            nn.Linear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),
            nn.Linear(4096, num_classes)
        )

        self.regime = {
            0: {'optimizer': 'SGD', 'lr': 11.3137*1e-2,
                'weight_decay': 0e-4, 'momentum': 0.9},
            20: {'lr': 11.3137*1e-3},
            40: {'lr': 11.3137*1e-4}
        }
        self.quiet_parameters = [
             #'features.0.weight',
             #'features.2.weight',
             #'features.2.bias',
             #'features.4.weight',
             #'features.7.weight',
             #'features.7.bias',
             #'features.8.weight',
             #'features.10.weight',
             #'features.10.bias',
             #'features.11.weight',
             #'features.13.weight',
             #'features.13.bias',
             #'features.14.weight',
             #'features.17.weight',
             #'features.17.bias',
             #'classifier.0.weight',
             #'classifier.1.weight',
             #'classifier.1.bias',
             #'classifier.4.weight',
             #'classifier.5.weight',
             #'classifier.5.bias',
             #'classifier.8.weight',
             #'classifier.8.bias',
             ]

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.input_transform = {
            'train': transforms.Compose([
                transforms.Scale(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]),
            'eval': transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        }

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnetvlr(**kwargs):
    num_classes = getattr(kwargs, 'num_classes', 1000)
    return AlexNetOWT_BN(num_classes)
