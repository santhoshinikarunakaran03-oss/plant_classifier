import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import json
import os
import urllib.request

st.set_page_config(page_title="Plant Variety Classifier", page_icon="🌿", layout="centered")

st.title("🌿 Plant Variety Classification App")
st.write("Upload a leaf image to identify the plant species using your trained ResNet-18 model.")

MODEL_PATH = "plant_model.pth"
CLASS_NAMES_PATH = "class_names.json"

# Paste your copied release link here
MODEL_URL = "https://github.com/santhoshinikarunakaran03-oss/plant_classifier/releases/download/v1.0/plant_model.pth"

@st.cache_resource
def load_model_and_classes():
    # Automatically download weights if not present on Streamlit server
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model weights on initial startup... Please wait.")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        
    if not os.path.exists(CLASS_NAMES_PATH):
        return None, None
    
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
        
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    
    return model, class_names

model, class_names = load_model_and_classes()

if model is None:
    st.error("class_names.json not found! Please upload it to your GitHub repo.")
else:
    uploaded_file = st.file_uploader("Choose a plant leaf image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)
            
        predicted_class = class_names[predicted_idx.item()]
        
        st.success(f"**Prediction:** {predicted_class}")
        st.info(f"**Confidence Score:** {confidence.item() * 100:.2f}%")
