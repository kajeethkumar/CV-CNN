# CVCNN: Complex Valued Convolutional Neural Network

This is a sample code of our project
# Python
Supported versions: Python >= 3.10
# Requirements
```
pip install scikit-learn torch scipy pywt
```
# Model Architecture
```
class CVCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.block1 = nn.Sequential(
            ComplexConv2d(1, 16, 3, padding=1),
            ComplexReLU(),
            ComplexMaxPool2d(2),
            ComplexDropout(0.25)
        )

        self.block2 = nn.Sequential(
            ComplexConv2d(16, 32, 3, padding=1),
            ComplexReLU(),
            ComplexMaxPool2d(2),
            ComplexDropout(0.25)
        )

        # We'll infer this dynamically (see below)
        self.fc1 = None
        self.fc2 = None
        self.n_classes = n_classes

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        x = x.view(x.size(0), -1)

        # Lazy initialization (VERY IMPORTANT)
        if self.fc1 is None:
            self.fc1 = ComplexLinear(x.shape[1], 64).to(x.device)
            self.fc2 = ComplexLinear(64, self.n_classes).to(x.device)

        x = self.fc1(x)
        x = self.fc2(x)
        return x
```
**Note: This repository is under development**.
