from torchvision.datasets import ImageFolder
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

dataset = ImageFolder(
    r"D:\drive-download-20260316T210828Z-3-001\processed_images",
    transform=transform
)

print("Classes:", dataset.classes)
print("Total images:", len(dataset))