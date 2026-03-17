import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# dataset path
data_dir = r"D:\PD_LAB\dataset\train\spoilage_detection"

# transforms
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# dataset
dataset = datasets.ImageFolder(data_dir, transform=transform)

# dataloader
loader = DataLoader(dataset, batch_size=9, shuffle=True)

# get one batch
images, labels = next(iter(loader))

class_names = dataset.classes

# plot grid
plt.figure(figsize=(8,8))

for i in range(9):

    plt.subplot(3,3,i+1)

    img = images[i].permute(1,2,0)

    plt.imshow(img)
    plt.title(class_names[labels[i]])
    plt.axis("off")

plt.tight_layout()
plt.show()