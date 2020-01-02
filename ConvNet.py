import torch.nn as nn

N_img_height = 66
N_img_width = 220
N_img_channels = 3
inputShape = (N_img_height, N_img_width, N_img_channels)


nn.Conv2d(N_img_channels, 24, kernel_size=5, stride=2)
# normalization    
 # perform custom normalization before lambda layer in network
#model.add(Lambda(lambda x: x/ 127.5 - 1, input_shape = inputShape))
#nn.Conv2d(24, 36, 5)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ELU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=5, stride=2),
            nn.ELU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1),
            nn.ELU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU())
        self.dropout = nn.Dropout(0.5) 
        self.fc1 = nn.Linear(x,100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50,10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, input):
        output = self.layer1(input)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.layer5(output)
        return output

#self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
