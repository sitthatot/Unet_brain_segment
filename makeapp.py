import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
from glob import glob
import os
import random
import time


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)
        return down, p


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)

        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)

        middle = self.bottle_neck(p4)

        up_1 = self.up_convolution_1(middle, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = self.out(up_4)
        return torch.sigmoid(out)


def single_image_inference(image_pth, model_pth, device):
    # Initialize and load the model
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_pth, map_location=device))
    model.eval()

    # Calculate model size in MB
    model_size = os.path.getsize(model_pth) / (1024 * 1024)

    # Define image transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Load and preprocess the image and corresponding mask
    img = transform(Image.open(image_pth)).float().to(device)
    real_mask = transform(Image.open(image_pth.replace(".tif", "_mask.tif"))).float().to(device)
    img = img.unsqueeze(0)

    # Perform inference and track time
    start_time = time.time()
    with torch.no_grad():
        pred_mask = model(img)
    inference_time = time.time() - start_time

    # Post-process predicted mask
    pred_mask = pred_mask.squeeze(0).cpu().detach()
    pred_mask = pred_mask.permute(1, 2, 0).numpy()
    pred_mask = (pred_mask > 0.5).astype(float)

    # Prepare image and ground truth mask for display
    img = img.squeeze(0).cpu().detach().permute(1, 2, 0)
    real_mask = real_mask.squeeze(0).cpu().detach().numpy()

    # Plot image, ground truth mask, and predicted mask
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    
    ax[0].imshow(img, cmap="gray")
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    
    ax[1].imshow(real_mask, cmap="gray")
    ax[1].set_title("Ground Truth Mask")
    ax[1].axis('off')
    
    ax[2].imshow(pred_mask, cmap="gray")
    ax[2].set_title("Predicted Mask")
    ax[2].axis('off')
    
    plt.show()

    # Return model size and inference time
    return model_size, inference_time


def browse_image():
    global selected_image_path  # Declare as global to update the variable
    default_image_folder = "TCGA_HT_A616_19991226"
    selected_image_path = filedialog.askopenfilename(initialdir=default_image_folder, title="Select Image",
                                                     filetypes=[("TIF files", "*.tif"), ("All files", "*.*")])
    if selected_image_path:
        image_label.config(text=os.path.basename(selected_image_path))


def browse_model():
    global selected_model_path  # Declare as global to update the variable
    default_model_folder = "models/"
    selected_model_path = filedialog.askopenfilename(initialdir=default_model_folder, title="Select Model",
                                                     filetypes=[("Model files", "*.pth"), ("All files", "*.*")])
    if selected_model_path:
        model_label.config(text=os.path.basename(selected_model_path))


def run_inference():
    try:
        if not selected_image_path or not selected_model_path:
            raise FileNotFoundError("Please select both an image and a model file.")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_size, inference_time = single_image_inference(selected_image_path, selected_model_path, device)
        
        # Display model size and inference time
        messagebox.showinfo("Results", f"Model size: {model_size:.2f} MB\nInference time: {inference_time:.4f} seconds")
    
    except Exception as e:
        messagebox.showerror("Error", str(e))


# Create the GUI window
root = tk.Tk()
root.title("Image Segmentation Inference")

selected_image_path = ""
selected_model_path = ""

# Image selection
tk.Label(root, text="Select Image:").grid(row=0, column=0, padx=10, pady=5, sticky='e')
image_btn = tk.Button(root, text="Browse", command=browse_image)
image_btn.grid(row=0, column=1, padx=10, pady=5)
image_label = tk.Label(root, text="No image selected", width=30)
image_label.grid(row=0, column=2, padx=10, pady=5)

# Model selection
tk.Label(root, text="Select Model:").grid(row=1, column=0, padx=10, pady=5, sticky='e')
model_btn = tk.Button(root, text="Browse", command=browse_model)
model_btn.grid(row=1, column=1, padx=10, pady=5)
model_label = tk.Label(root, text="No model selected", width=30)
model_label.grid(row=1, column=2, padx=10, pady=5)

# Run inference button
run_btn = tk.Button(root, text="Run", command=run_inference)
run_btn.grid(row=2, column=1, padx=10, pady=20)

root.mainloop()
