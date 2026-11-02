import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os
import urllib.request
import math

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Plant Crop Classifier v2.1",
    page_icon="🌱",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "plant_model.pth")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names_v2.json")
MODEL_URL = "https://github.com/santhoshinikarunakaran03-oss/plant_classifier/releases/download/v2.1/plant_model.pth"

# --- 2. CROP METADATA DATABASE (Applications & Descriptions) ---
CROP_METADATA = {
    "Barnyard_millet": {
        "category": "Millet / Cereal",
        "uses": "Gluten-free cereal grain, animal fodder, rapid-growth cover crop.",
        "key_traits": "Erect grass, dense spikelets, high dietary fiber and iron."
    },
    "co51": {
        "category": "Paddy / Rice Variety",
        "uses": "High-yielding staple rice variety, straw for livestock feed.",
        "key_traits": "Medium slim grains, drought-tolerant short duration crop."
    },
    "co66": {
        "category": "Paddy / Rice Variety",
        "uses": "Commercial rice production, bran oil extraction.",
        "key_traits": "Broad erect green leaves, lodging resistant."
    },
    "finger_millet": {
        "category": "Millet (Ragi)",
        "uses": "Nutritional flour (Ragi malt/dosa), infant food, baby cereal.",
        "key_traits": "Finger-like terminal spikes, exceptionally high calcium content."
    },
    "foxtail_millet": {
        "category": "Millet",
        "uses": "Health food grain, bird seed, traditional brewing.",
        "key_traits": "Bristly cylindrical spike head resembling a foxtail."
    },
    "greengram": {
        "category": "Pulse / Legume",
        "uses": "Sprouted beans, dhal preparation, soil nitrogen fixation.",
        "key_traits": "Trifoliate leaves, small green seed pods, dark green canopy."
    },
    "groundnut": {
        "category": "Oilseed / Legume",
        "uses": "Peanut oil extraction, roasted snacks, protein cake feed.",
        "key_traits": "Tetrafoliate compound leaves, subterranean pod formation."
    },
    "kodo_millet": {
        "category": "Millet",
        "uses": "Diabetes-friendly grain substitute, bio-fuel biomass.",
        "key_traits": "Tough stem grass, flat linear leaves, highly drought-resistant."
    },
    "little_millet": {
        "category": "Millet",
        "uses": "Dietary porridge, livestock fodder in arid regions.",
        "key_traits": "Small panicles, slender linear leaf blades."
    },
    "maize_blackgram": {
        "category": "Intercrop System",
        "uses": "Dual cereal-pulse cropping, soil fertility improvement.",
        "key_traits": "Broad maize leaves interplanted with low-growing blackgram legumes."
    },
    "pillipesara": {
        "category": "Forage Legume",
        "uses": "Green manure, soil reclamation, high-protein cattle grazing.",
        "key_traits": "Deep taproot, trailing vines, pinnate compound leaves."
    },
    "proso_millet": {
        "category": "Millet",
        "uses": "Birdseed mixtures, gluten-free baking, emergency catch crop.",
        "key_traits": "Drooping open panicles, hairy leaf sheaths."
    },
    "sunflower": {
        "category": "Commercial Oilseed",
        "uses": "Edible cooking oil, sunflower seeds, bio-diesel feedstock.",
        "key_traits": "Large heart-shaped coarse leaves, yellow composite flowers."
    },
    "tobaco_godavari": {
        "category": "Cash Crop",
        "uses": "Commercial tobacco processing, nicotine extract production.",
        "key_traits": "Broad ovate leaves with prominent sticky glandular hairs."
    },
    "wheat": {
        "category": "Staple Cereal",
        "uses": "Flour production (bread/flatbreads), bran livestock feed.",
        "key_traits": "Parallel veined long narrow leaf blades, terminal head spikes."
    },
    "white_ponni": {
        "category": "Paddy / Rice Variety",
        "uses": "Premium cooked rice, rice flour products.",
        "key_traits": "Slender white grains, medium tall plant canopy with narrow leaf blades."
    }
}

# --- 3. MODEL & METADATA INITIALIZATION ---
def download_model_if_missing():
    if not os.path.exists(MODEL_PATH):
        st.info("⬇️ Downloading trained model weights (v2.1)... Please wait.")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download model weights: {e}")
            st.stop()

download_model_if_missing()

if not os.path.exists(CLASS_NAMES_PATH):
    st.error(f"Missing metadata file: `{CLASS_NAMES_PATH}`.")
    st.stop()

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

@st.cache_resource
def load_trained_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device

model, device = load_trained_model()

# --- 4. PREPROCESSING & INFERENCE ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def calculate_entropy(probabilities):
    return -sum(p * math.log(p) for p in probabilities if p > 0)

def predict_top_k(image, k=3):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0).tolist()
    
    # Sort top-k predictions
    top_indices = sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)[:k]
    top_predictions = [
        (class_names[idx], probabilities[idx] * 100) for idx in top_indices
    ]
    
    top_confidence = top_predictions[0][1]
    entropy = calculate_entropy(probabilities)
    
    return top_predictions, top_confidence, entropy

# --- 5. DASHBOARD LAYOUT ---
st.title("🌱 Crop Classifier & Agricultural Knowledge Base")
st.caption("Version 2.1 | PyTorch ResNet-18 Transfer Learning")

col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.subheader("Upload Sample")
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Input Leaf Image", use_container_width=True)

with col_right:
    st.subheader("Inference Analysis")
    if uploaded_file:
        with st.spinner("Calculating feature vectors..."):
            top_3_preds, top_conf, entropy = predict_top_k(image, k=3)
        
        CONFIDENCE_THRESHOLD = 70.0
        ENTROPY_THRESHOLD = 1.25
        
        # --- CASE A: CONFIDENT CLASSIFICATION ---
        if top_conf >= CONFIDENCE_THRESHOLD and entropy <= ENTROPY_THRESHOLD:
            top_class, confidence = top_3_preds[0]
            st.success(f"### Predicted Crop: **{top_class}**")
            st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
            
            meta = CROP_METADATA.get(top_class, {})
            st.markdown("---")
            st.markdown(f"**Category:** {meta.get('category', 'N/A')}")
            st.markdown(f"**Primary Applications:** {meta.get('uses', 'N/A')}")
            st.markdown(f"**Morphological Features:** {meta.get('key_traits', 'N/A')}")
            
        # --- CASE B: UNRECOGNIZED / LOW CONFIDENCE (TOP 3 FALLBACK) ---
        else:
            st.warning("⚠️ **Unrecognized / Low Confidence Leaf Sample**")
            st.caption(f"Top confidence ({top_conf:.2f}%) or entropy ({entropy:.4f}) indicates high uncertainty.")
            st.info("### Closest Candidate Matching Crops")
            
            for idx, (c_name, c_prob) in enumerate(top_3_preds, 1):
                meta = CROP_METADATA.get(c_name, {})
                with st.expander(f"**#{idx} Candidate: {c_name} ({c_prob:.2f}% Match)**", expanded=(idx==1)):
                    st.write(f"**Classification Category:** {meta.get('category', 'N/A')}")
                    st.write(f"**Agricultural Uses:** {meta.get('uses', 'N/A')}")
                    st.write(f"**Key Visual Traits:** {meta.get('key_traits', 'N/A')}")
    else:
        st.info("Upload an image on the left panel to execute model inference.")
