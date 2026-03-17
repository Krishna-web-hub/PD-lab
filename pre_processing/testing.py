from torchvision.datasets import ImageFolder
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

dataset = ImageFolder(
    r"D:\PD_LAB\processed_images",
    transform=transform
)

print("Classes:", dataset.classes)
print("Total images:", len(dataset))