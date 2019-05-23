import torch.nn as nn
import torch.nn.functional as f


class NaimishNet(nn.Module):
    
    def __init__(self):
        super(NaimishNet, self).__init__()

        """
        NaimishNet has layers below:

        Layer Num  | Number of Filters | Filter Shape
         ---------------------------------------------
        1          |        32         |    (4,4)
        2          |        64         |    (3,3)
        3          |        128        |    (2,2)
        4          |        256        |    (1,1)
        ---------------------------------------------

        Activation : ELU(Exponential Linear Unit)
        MaxxPool : 4x(2,2)
        """

        self.max_pool = nn.MaxPool2d(2, 2)
    
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.drop1 = nn.Dropout(0.1)
        
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.drop2 = nn.Dropout(0.2)
        
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.drop3 = nn.Dropout(0.3)
        
        self.conv4 = nn.Conv2d(128, 256, 1)
        self.drop4 = nn.Dropout(0.4)
        
        self.dense1 = nn.Linear(256 * 13 *13, 256 * 13)
        self.drop5 = nn.Dropout(0.5)
        
        self.dense2 = nn.Linear(256 * 13, 256 * 13)
        self.drop6 = nn.Dropout(0.6)
        
        self.dense3 = nn.Linear(256 * 13, 136)


    def forward(self, x):
        x = self.max_pool(f.elu(self.conv1(x)))
        x = self.drop1(x)

        x = self.max_pool(f.elu(self.conv2(x)))
        x = self.drop2(x)

        x = self.max_pool(f.elu(self.conv3(x)))
        x = self.drop3(x)

        x = self.max_pool(f.elu(self.conv4(x)))
        x = self.drop4(x)

        # Flatten layer
        x = x.view(x.size(0), -1)
        
        x = f.elu(self.dense1(x))
        x = self.drop5(x)

        x = f.relu(self.dense2(x))
        x = self.drop6(x)

        x = self.dense3(x)

        return x
        
