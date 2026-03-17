import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# dataset path
data_dir = r"D:\PD_LAB\dataset\train\spoilage_detection"

# transforms (image preprocessing before tensor conversion)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# dataset loader
dataset = datasets.ImageFolder(data_dir, transform=transform)

print("Classes:", dataset.classes)
print("Total images:", len(dataset))

# DataLoader
loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

# get one batch
images, labels = next(iter(loader))

print("Tensor shape:", images.shape)
print("Labels:", labels)

# visualize first image tensor
img = images[0].permute(1,2,0)   # convert CHW → HWC for display

plt.imshow(img)
plt.title("Example tensor image")
plt.show()