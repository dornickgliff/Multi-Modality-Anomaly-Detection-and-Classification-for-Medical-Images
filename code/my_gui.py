import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
import torch.nn.functional as F

class ModalityClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(ModalityClassifier, self).__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ModalityClassifier(num_classes=3).to(device)
model.load_state_dict(torch.load("efficient_b0.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


classes = ['MRI', 'OCT', 'Xray']

def predict_modality(image_path, model, transform, classes):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = output.max(1)
        return classes[predicted.item()]

# Encoder, Decoder, and Anomaly Detection
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        self.stage1 = nn.Sequential(*list(resnet.children())[:4])
        self.stage2 = nn.Sequential(*list(resnet.children())[4:6])
        self.stage3 = nn.Sequential(*list(resnet.children())[6:7])
        self.stage4 = nn.Sequential(*list(resnet.children())[7:8])

    def forward(self, x):
        F1 = self.stage1(x)
        F2 = self.stage2(F1)
        F3 = self.stage3(F2)
        F4 = self.stage4(F3)
        return F1, F2, F3, F4

class MSTB(nn.Module):
    def __init__(self, d_model, nhead, reduction_factor=4):
        super(MSTB, self).__init__()
        self.local_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.global_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.reduce_kv = nn.Conv2d(d_model, d_model // reduction_factor, kernel_size=1)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model))
    def forward(self, F_local, F_region):
        B, C, H, W = F_local.shape
        F_local_flat = F_local.contiguous().view(B, C, -1).permute(0, 2, 1)
        F_region_flat = F_region.contiguous().view(B, C, -1).permute(0, 2, 1)
        F_region_reduced = self.reduce_kv(F_region).contiguous().view(B, -1, H * W).permute(0, 2, 1)
        local_out, _ = self.local_attn(F_local_flat, F_local_flat, F_local_flat)
        global_out, _ = self.global_attn(F_local_flat, F_region_flat, F_region_flat)
        combined = local_out + global_out
        combined_flat = combined.reshape(-1, combined.size(-1))
        refined_flat = self.ffn(combined_flat)
        refined = refined_flat.reshape(B, H * W, -1).permute(0, 2, 1).reshape(B, C, H, W)
        return refined
8
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mstb = MSTB(d_model=512, nhead=8, reduction_factor=4)
        self.upsample1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.final_upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False)
        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, F4_E):
        FConv = self.conv_block(F4_E)
        FTrans = self.mstb(F4_E, F4_E)
        F3_D = self.upsample1(FTrans)
        F2_D = self.upsample2(F3_D)
        F1_D = self.upsample3(F2_D)
        reconstructed_image = self.final_upsample(F1_D)
        reconstructed_image = self.final_conv(reconstructed_image)
        return F1_D, F2_D, F3_D, FTrans, reconstructed_image

encoder = Encoder().to(device)
decoder = Decoder().to(device)
def compute_anomaly_map(F_E, F_D, input_size):
    anomaly_maps = []

    for f_e, f_d in zip(F_E, F_D):
        # Resize feature maps to match input image size
        f_e_resized = F.interpolate(f_e, size=input_size, mode='bilinear', align_corners=False)
        f_d_resized = F.interpolate(f_d, size=input_size, mode='bilinear', align_corners=False)

        # Compute cosine similarity between encoder and decoder feature maps
        cos_sim = F.cosine_similarity(f_e_resized, f_d_resized, dim=1)

        # Convert similarity to anomaly score (higher difference → more anomalous)
        anomaly_map_k = 1 - cos_sim  # Higher value indicates higher anomaly

        anomaly_maps.append(anomaly_map_k)

    # Sum the anomaly maps across different stages
    anomaly_map = torch.sum(torch.stack(anomaly_maps, dim=0), dim=0)

    return anomaly_map

def compute_anomaly_score(anomaly_map):
    return torch.max(anomaly_map)

def test_anomaly_detection(encoder, decoder, image, device, threshold=1.44):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        F1_E, F2_E, F3_E, F4_E = encoder(image_tensor)
        F1_D, F2_D, F3_D, F4_D, reconstructed_image = decoder(F4_E)
        anomaly_map = compute_anomaly_map([F1_E, F2_E, F3_E, F4_E], [F1_D, F2_D, F3_D, F4_D], image.size[::-1])
        anomaly_score = compute_anomaly_score(anomaly_map)
        anomaly = anomaly_score.item() > 0.2
        return anomaly, anomaly_map

def generate_heatmap_cv2(image, anomaly_map, alpha=0.6):
    anomaly_map_np = anomaly_map.squeeze().cpu().numpy()
    anomaly_map_np = cv2.normalize(anomaly_map_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(anomaly_map_np, cv2.COLORMAP_JET)
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    heatmap_overlay = cv2.addWeighted(image_bgr, 1 - alpha, heatmap, alpha, 0)
    cv2.imshow("Anomaly Heatmap", heatmap_overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Medical Image Analysis")
        self.geometry("800x600")

        # Upload Image Button
        self.upload_button = tk.Button(self, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        # Image Preview
        self.image_label = tk.Label(self)
        self.image_label.pack(pady=10)

        # Modality Classification
        self.modality_label = tk.Label(self, text="Modality: ")
        self.modality_label.pack(pady=10)

        # Anomaly Detection
        self.anomaly_label = tk.Label(self, text="Anomaly: ")
        self.anomaly_label.pack(pady=10)

        # Heatmap Display
        self.heatmap_label = tk.Label(self)
        self.heatmap_label.pack(pady=10)

        # Disease Classification
        self.disease_label = tk.Label(self, text="Disease: ")
        self.disease_label.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpeg;*.jpg;*.png")])
        if file_path:
            # Display the uploaded image
            image = Image.open(file_path)
            image.thumbnail((300, 300))
            img_tk = ImageTk.PhotoImage(image)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk

            # Classify Modality
            predicted_modality = predict_modality(file_path, model, transform, classes)
            self.modality_label.config(text=f"Modality: {predicted_modality}")

            # Load Encoder and Decoder based on Modality
            if(predicted_modality=="OCT"):
                encoder.load_state_dict(torch.load("encoder_model_retinal.pth"))
                decoder.load_state_dict(torch.load("decoder_model_retinal.pth"))
            elif(predicted_modality=="MRI"):
                encoder.load_state_dict(torch.load("encoder_model_brain.pth"))
                decoder.load_state_dict(torch.load("decoder_model_brain.pth"))
            elif(predicted_modality=="Xray"): 
                encoder.load_state_dict(torch.load("encoder_model_chest.pth"))
                decoder.load_state_dict(torch.load("decoder_model_chest.pth"))

            encoder.to(device)
            decoder.to(device)
            encoder.eval()
            decoder.eval()

            # Anomaly Detection
            anomaly, anomaly_map = test_anomaly_detection(encoder, decoder, Image.open(file_path).convert("RGB"), device)
            self.anomaly_label.config(text=f"Anomaly: {'Detected' if anomaly else 'Not Detected'}")
            
            if anomaly:
                # Generate Heatmap
                generate_heatmap_cv2(Image.open(file_path).convert("RGB"), anomaly_map)
                if(predicted_modality=='Xray'):
                    vgg16 = models.vgg16(pretrained=True)
                    vgg16.classifier[6] = nn.Linear(4096, 3)
                    vgg16 = vgg16.to(device)

                    # DenseNet-201 Model
                    densenet201 = models.densenet201(pretrained=True)
                    densenet201.classifier = nn.Linear(1920, 3)
                    densenet201 = densenet201.to(device)

                    # EfficientNet-B0 Model
                    efficientnet_b0 = models.efficientnet_b0(pretrained=True)
                    efficientnet_b0.classifier[1] = nn.Linear(1280, 3)
                    efficientnet_b0 = efficientnet_b0.to(device)


                    vgg16.load_state_dict(torch.load("vgg16_xray_model.pth"))
                    vgg16.to(device).eval()

                    densenet201.load_state_dict(torch.load("densenet201_xray_model.pth"))
                    densenet201.to(device).eval()

                    efficientnet_b0.load_state_dict(torch.load("efficientnet_b0_xray_model.pth"))
                    efficientnet_b0.to(device).eval()

                    ensemble_models = [vgg16, densenet201, efficientnet_b0]
                    
                    class_labels = ["Covid", "Pneumonia", "Normal"]

                    def predict_image(image_path, models):
                        image = Image.open(image_path).convert("RGB")
                        image = transform(image).unsqueeze(0)  # Add batch dimension
                        image = image.to(device)

                        models = [model.eval() for model in models]

                        with torch.no_grad():
                            avg_output = sum(model(image) for model in models) / len(models)
                            _, predicted_class = avg_output.max(1)

                        return class_labels[predicted_class.item()]

                    # Test with a single image
                    image_path = file_path
                    prediction = predict_image(image_path, ensemble_models)
                    self.disease_label.config(text=f"Disease: {prediction}")

                        
                elif(predicted_modality=='OCT'):
                    # VGG-16 Model
                    vgg16 = models.vgg16(pretrained=True)
                    vgg16.classifier[6] = nn.Linear(4096, 4)
                    vgg16 = vgg16.to(device)

                    # DenseNet-201 Model
                    densenet201 = models.densenet201(pretrained=True)
                    densenet201.classifier = nn.Linear(1920, 4)
                    densenet201 = densenet201.to(device)

                    # EfficientNet-B0 Model
                    efficientnet_b0 = models.efficientnet_b0(pretrained=True)
                    efficientnet_b0.classifier[1] = nn.Linear(1280, 4)
                    efficientnet_b0 = efficientnet_b0.to(device)

                            
                    vgg16.load_state_dict(torch.load("vgg16_ret_model.pth"))
                    vgg16.to(device).eval()

                    densenet201.load_state_dict(torch.load("densenet201_ret_model.pth"))
                    densenet201.to(device).eval()

                    efficientnet_b0.load_state_dict(torch.load("efficientnet_b0_ret_model.pth"))
                    efficientnet_b0.to(device).eval()

                    ensemble_models = [vgg16, densenet201, efficientnet_b0]
                    
                    class_labels = ["CNV", "DME", "DRUSSEN", "NORMAL"]

                    def predict_image(image_path, models):
                        image = Image.open(image_path).convert("RGB")
                        image = transform(image).unsqueeze(0).to(device)
                        models = [model.eval() for model in models]
                        with torch.no_grad():
                            avg_output = sum(model(image) for model in models) / len(models)
                            _, predicted_class = avg_output.max(1)
                        return class_labels[predicted_class.item()]

                    # Test with a single image
                    image_path = file_path
                    prediction = predict_image(image_path, ensemble_models)
                    self.disease_label.config(text=f"Disease: {prediction}")


                elif(predicted_modality=='MRI'):
                    # VGG-16 Model
                    vgg16 = models.vgg16(pretrained=True)
                    vgg16.classifier[6] = nn.Linear(4096, 4)
                    vgg16 = vgg16.to(device)

                    # DenseNet-201 Model
                    densenet201 = models.densenet201(pretrained=True)
                    densenet201.classifier = nn.Linear(1920, 4)
                    densenet201 = densenet201.to(device)

                    # EfficientNet-B0 Model
                    efficientnet_b0 = models.efficientnet_b0(pretrained=True)
                    efficientnet_b0.classifier[1] = nn.Linear(1280, 4)
                    efficientnet_b0 = efficientnet_b0.to(device)

                        
                    vgg16.load_state_dict(torch.load("vgg16_brain_model.pth"))
                    vgg16.to(device).eval()

                    densenet201.load_state_dict(torch.load("densenet201_brain_model.pth"))
                    densenet201.to(device).eval()

                    efficientnet_b0.load_state_dict(torch.load("efficientnet_b0_brain_model.pth"))
                    efficientnet_b0.to(device).eval()

                    ensemble_models = [vgg16, densenet201, efficientnet_b0]
                    class_labels = ["Glioma", "Meningioma", "Normal", "Pituitary"]

                    def predict_image(image_path, models):
                        image = Image.open(image_path).convert("RGB")
                        image = transform(image).unsqueeze(0)  # Add batch dimension
                        image = image.to(device)

                        models = [model.eval() for model in models]

                        with torch.no_grad():
                            avg_output = sum(model(image) for model in models) / len(models)
                            _, predicted_class = avg_output.max(1)

                        return class_labels[predicted_class.item()]
                    
                    image_path = file_path
                    prediction = predict_image(image_path, ensemble_models)
                    self.disease_label.config(text=f"Disease: {prediction}")

if __name__ == "__main__":
    app = Application()
    app.mainloop()