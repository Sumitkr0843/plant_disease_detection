import torch
import timm
from PIL import Image
from torchvision import transforms

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load classes
with open("classes.txt") as f:
    classes = [line.strip() for line in f.readlines()]

num_classes = len(classes)

# create model
model = timm.create_model(
    "mobilevit_xs",
    num_classes=num_classes
)

# load checkpoint
checkpoint = torch.load("model/plant_disease_model.pth", map_location=device)

# load weights (ignore small architecture differences)
model.load_state_dict(checkpoint["model_state_dict"], strict=False)

model.to(device)
model.eval()

# image transform
transform = transforms.Compose([
    transforms.Resize((160,160)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])


def predict(image_path):

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return classes[predicted.item()]