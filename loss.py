# Om Lachake, Aug 2024

import torch
import torch.nn as nn
import numpy as np
import cv2
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


def rgb_to_hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return hsv_image

def extract_hue(image):
    hsv_image = rgb_to_hsv(image)
    hue_channel = hsv_image[:, :, 0]
    return hue_channel



class HueLoss(nn.Module):
    def __init__(self):
        super(HueLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def preprocess_for_hue(self,image):
        if isinstance(image, torch.Tensor):
            batch_size, channels, height, width = image.shape
            image_np = image.permute(0, 2, 3, 1).cpu().numpy()  # [B, C, H, W] -> [B, H, W, C]
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image
        return image_np


    def forward(self, real_image, generated_image):
        real_image_np = self.preprocess_for_hue(real_image)
        generated_image_np = self.preprocess_for_hue(generated_image)

        batch_size = real_image.shape[0]
        hue_losses = []

        for i in range(batch_size):
            real_hue = extract_hue(real_image_np[i])
            generated_hue = extract_hue(generated_image_np[i])
            real_hue = torch.from_numpy(real_hue).float()
            generated_hue = torch.from_numpy(generated_hue).float()
            hue_loss = self.l1_loss(real_hue, generated_hue)
            hue_losses.append(hue_loss)

        return torch.mean(torch.stack(hue_losses))

class PerceptualLoss(nn.Module):
    def __init__(self, feature_layer=3):
        super(PerceptualLoss, self).__init__()  
        vgg = models.vgg16(pretrained=True).features
        self.features = nn.Sequential(*list(vgg.children())[:feature_layer]).eval()
        for param in self.features.parameters():
            param.requires_grad = False
        self.l1_loss = nn.L1Loss()
        self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    def preprocess_for_perceptual(self,images):
        return self.transform(images) 

    def forward(self, real_image, generated_image):

        if real_image.shape != generated_image.shape:
            raise ValueError("Input images must have the same dimensions")
        real_image = self.preprocess_for_perceptual(real_image)
        generated_image = self.preprocess_for_perceptual(generated_image)

        real_features = self.features(real_image)
        generated_features = self.features(generated_image)
        return self.l1_loss(real_features, generated_features)




