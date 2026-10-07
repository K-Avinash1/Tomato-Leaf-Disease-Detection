import torch.nn as nn


def conv_block(cin, cout, n_convs):
    layers = []
    for i in range(n_convs):
        layers += [nn.Conv2d(cin if i == 0 else cout, cout, kernel_size=3, padding=1, bias=False),
                   nn.BatchNorm2d(cout),
                   nn.ReLU(inplace=True)]
    layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class TomatoCNN(nn.Module):
    """Baseline CNN trained from scratch: 5 conv stages -> global average pooling -> linear."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32, 1),
            conv_block(32, 64, 2),
            conv_block(64, 128, 2),
            conv_block(128, 256, 2),
            conv_block(256, 256, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))