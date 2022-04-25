import torch.nn as nn

class Convnet(nn.Module):
    def __init__(self, dropout=0.5):
      super(Convnet, self).__init__()
      self.convnet = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3), 
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2), 

        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3), 
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),  

        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3), 
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),  

        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3), 
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3), 
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),  
        nn.Flatten() 
        )
      self.classifier = nn.Sequential(
          nn.Dropout(dropout), 
          nn.Linear(in_features= 4608, out_features=512),
          nn.ReLU(),
          nn.Dropout(dropout),
          nn.Linear(in_features=512, out_features=256),
          nn.ReLU(),
          nn.Linear(in_features=256, out_features=128),
          nn.ReLU(),
          nn.Linear(in_features=128, out_features=1)
        )
    def forward(self, x):
      x = self.convnet(x)
      x = self.classifier(x)
      return x