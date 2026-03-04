import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from predict import model


# preprocessing
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

img_path = "test_imgs\diseased.jpg"

image = Image.open(img_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)

rgb_img = np.array(image.resize((256,256))) / 255.0


# choose multiple layers
target_layers = [
    model.stages[0][-1],
    model.stages[1][-1],
    model.stages[2][-1],
    model.stages[3][-1]
]


cams = []

for layer in target_layers:

    cam = GradCAM(model=model, target_layers=[layer])

    grayscale_cam = cam(input_tensor=input_tensor)[0]

    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    cams.append(visualization)


# plot grid
fig, ax = plt.subplots(1,5,figsize=(18,4))

ax[0].imshow(rgb_img)
ax[0].set_title("Original")
ax[0].axis("off")

for i in range(4):
    ax[i+1].imshow(cams[i])
    ax[i+1].set_title(f"CAM-{i+1}")
    ax[i+1].axis("off")

plt.tight_layout()

plt.savefig("gradcam02_grid.png",dpi=300)

print("Saved IEEE style GradCAM figure as gradcam_grid.png")