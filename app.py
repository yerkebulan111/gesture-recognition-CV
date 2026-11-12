import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import time

# Page configuration
st.set_page_config(page_title="Gesture Recognition & XAI Dashboard", layout="wide")

# Constants
MODEL_PATH = "models/resnet18_gesture.pth"
CLASSES = ["peace", "stop", "thumbs_up"]
DEVICE = torch.device("cpu") # Change to cuda if available

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

model = load_model()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Simple Grad-CAM implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
            
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, target_class]
        score.backward()

        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = torch.clamp(cam, min=0)
        cam = cam / (torch.max(cam) + 1e-8)
        return cam.cpu().detach().numpy()

grad_cam = GradCAM(model, model.layer4[1].conv2)

# UI Layout
st.title("✋ Hand Gesture Recognition & Explainability (XAI) Dashboard")
st.write("Assignment 2: Robust Gesture Classification with ResNet18 & Grad-CAM Visualizations")

uploaded_file = st.file_uploader("Upload a gesture test image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")
    input_tensor = transform(original_image).unsqueeze(0).to(DEVICE)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Image")
        st.image(original_image, caption="Uploaded Gesture", use_container_width=True)
        
    # Inference & Latency Tracking
    start_time = time.time()
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
    inference_time = (time.time() - start_time) * 1000 # in ms
    
    predicted_class = CLASSES[predicted_idx.item()]
    
    # Generate Grad-CAM
    input_tensor.requires_grad = True
    cam_map = grad_cam.generate(input_tensor, predicted_idx.item())
    
    # Resize cam to original image size
    cam_resized = cv2.resize(cam_map, (original_image.size[0], original_image.size[1]))
    heat_map = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heat_map = cv2.cvtColor(heat_map, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap on original image
    superimposed = np.uint8(0.5 * np.array(original_image) + 0.5 * heat_map)
    
    with col2:
        st.subheader("Model Prediction & XAI")
        st.success(f"**Predicted Gesture:** {predicted_class.upper()}")
        st.metric(label="Confidence Score", value=f"{confidence.item() * 100:.2f}%")
        st.metric(label="Inference Latency", value=f"{inference_time:.2f} ms")
        
        st.subheader("Grad-CAM Heatmap")
        st.image(superimposed, caption="Where the model looked (Red/Yellow = High Attention)", use_container_width=True)

    # Class Probabilities Breakdown
    st.subheader("Class Probability Distribution")
    prob_dict = {CLASSES[i]: float(probabilities[i].item()) for i in range(len(CLASSES))}
    st.bar_chart(prob_dict)
