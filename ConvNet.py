import torch.nn as nn

""" 
A class implementing the structure of the end to end learning CNN developed by NVIDIA
https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

Methods
-------
forward(input)
    Runs the forward pass on the CNN
"""

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
             nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ELU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=5, stride=2),
            nn.ELU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ELU())
        self.dropout = nn.Dropout(0.5) 
        self.fc1 = nn.Linear(64*1*18,100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50,10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, input):
        """Does the forward pass for the CNN
        Parameters
        ----------
        input : tensor
            The tensor representing the image in the batch 

        Returns
        ------
        output: 
            This is the CNN model 
        """
        output = self.layer2(self.layer1(input))
        output = self.layer4(self.layer3(output))
        output = self.dropout(self.layer5(output))
        output = self.fc2(self.fc1(output))
        output = self.fc4(self.fc3(output))

        return output
