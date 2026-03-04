import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from predict import model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataset path
test_dir = "dataset/test"

transform = transforms.Compose([
    transforms.Resize((160,160)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(test_dir, transform=transform)

loader = DataLoader(dataset, batch_size=32, shuffle=False)

y_true = []
y_pred = []

model.eval()

with torch.no_grad():
    for images, labels in loader:

        images = images.to(device)

        outputs = model(images)

        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())


cm = confusion_matrix(y_true, y_pred)


plt.figure(figsize=(12,10))

sns.heatmap(
    cm,
    cmap="Greens",
    xticklabels=dataset.classes,
    yticklabels=dataset.classes
)

plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Confusion Matrix - Plant Disease Classification")

plt.tight_layout()

plt.savefig("confusion_matrix.png", dpi=300)

plt.show() 