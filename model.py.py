import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
)

state_dict = torch.load("best_unet_fire.pth", map_location=device)
model.load_state_dict(state_dict)

model.to(device)

model.eval()

transform = A.Compose([
    A.Resize(256,256),
    A.Normalize(),
    ToTensorV2()
])

def predict(image):

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    aug = transform(image=image_rgb)

    tensor = aug["image"].unsqueeze(0).to(device)

    with torch.no_grad():

        pred = model(tensor)

        pred = torch.sigmoid(pred)

    pred = pred.squeeze().cpu().numpy()

    pred = (pred > 0.5).astype(np.uint8)

    return pred