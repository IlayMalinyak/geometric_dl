from PIL import Image
import os
from torchvision import transforms, datasets
import torch
from group_conv import Rot90Group, bilinear_interpolation
import matplotlib.pyplot as plt
from utils import *

train_transform = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.1307,), (0.3081,))
                                                  ])

# To demonstrate the generalization capabilities our rotation equivariant layers bring, we apply a random
# rotation between 0 and 360 deg to the test set.
test_transform = transforms.Compose([transforms.ToTensor(),
                                                 RandomRotationFromSet([0, 90, 180, 270]),
                                                 transforms.Normalize((0.1307,), (0.3081,))
                                                 ])
train_ds = datasets.FashionMNIST(root="data", download=True, train=True, transform=train_transform)
test_ds = datasets.FashionMNIST(root="data", download=True, train=False, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)

for i in range(3):
    x,y = train_ds[i]
    print(f"Image shape: {x.shape}, Label: {y}")
    plt.imshow(x.squeeze().numpy())
    plt.title(f"Label: {y}")
    plt.show()
