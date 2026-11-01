import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import json
import os
import urllib.request
import torch.nn.functional as F
import datetime
from supabase import create_client

st.set_page_config(page_title="Crop Classifier V2 & Active Learner", page_icon="🌿", layout="centered")

# --- SUPABASE CONFIGURATION ---
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")

@st.cache_resource
def get_supabase_client():
    if SUPABASE_URL and SUPABASE_KEY:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    return None

supabase = get_supabase_client()

# --- CROP KNOWLEDGE BASE ---
CROP_INFO = {
    "Barnyard_millet": {
        "common_name": "Barnyard Millet (Sanwa / Kuthiraivali)",
        "regions": "India (Uttarakhand, Tamil Nadu, Andhra Pradesh), East Asia",
        "thriving_env": "Drought-resistant, heat-tolerant. Thrives in drylands & poor soils with minimal rainfall (300-500mm)."
    },
    "co51": {
        "common_name": "TNAU Rice CO 51 (Short-Duration Paddy)",
        "regions": "Southern India (Tamil Nadu, Andhra Pradesh, Puducherry)",
        "thriving_env": "Warm tropical climates (25°C-35°C), irrigated lowland rice ecosystems, fertile clay loams."
    },
    "co66": {
        "common_name": "TNAU Crop Variety CO 66",
        "regions": "Peninsular India (Tamil Nadu, Karnataka, Andhra Pradesh)",
        "thriving_env": "Tropical drylands & rainfed agro-ecosystems; prefers well-drained sandy loams (24°C-30°C)."
    },
    "finger_millet": {
        "common_name": "Finger Millet (Ragi / Mandua)",
        "regions": "India (Karnataka, Tamil Nadu, Uttarakhand), East Africa",
        "thriving_env": "Warm semi-arid climates, elevation up to 2000m, moderate rainfall, acidic to neutral loamy soils."
    },
    "foxtail_millet": {
        "common_name": "Foxtail Millet (Kangni / Thinai)",
        "regions": "India (Andhra Pradesh, Karnataka, Tamil Nadu), China",
        "thriving_env": "Highly drought-tolerant, short-duration crop. Thrives in semi-arid regions with well-drained loams."
    },
    "greengram": {
        "common_name": "Greengram (Mung Bean)",
        "regions": "India, Southeast Asia, Australia",
        "thriving_env": "Warm tropical/subtropical climates (25°C-35°C), well-drained fertile soils; sensitive to waterlogging."
    },
    "groundnut": {
        "common_name": "Groundnut (Peanut)",
        "regions": "India (Gujarat, Andhra Pradesh, Tamil Nadu), China, USA",
        "thriving_env": "Tropical/subtropical zones, light sandy-loam soils (25°C-30°C) with abundant sunshine."
    },
    "kodo_millet": {
        "common_name": "Kodo Millet (Varagu)",
        "regions": "Deccan Plateau of India (Madhya Pradesh, Chhattisgarh, Tamil Nadu)",
        "thriving_env": "Extremely hardy; thrives in stony, shallow, poor-fertility soils in hot semi-arid drylands."
    },
    "little_millet": {
        "common_name": "Little Millet (Kutki / Samai)",
        "regions": "India (Madhya Pradesh, Odisha, Andhra Pradesh, Tamil Nadu)",
        "thriving_env": "Resilient to drought & temporary waterlogging; grows well in light sandy loams in warm climates."
    },
    "maize_blackgram": {
        "common_name": "Maize - Blackgram Intercropping System",
        "regions": "India (Andhra Pradesh, Telangana, Madhya Pradesh)",
        "thriving_env": "Warm sunny weather (20°C-32°C), well-drained fertile soils; blackgram fixes nitrogen for maize."
    },
    "pillipesara": {
        "common_name": "Pillipesara (Wild Mung / Forage Legume)",
        "regions": "Southern & Eastern India (Andhra Pradesh, Telangana, Tamil Nadu)",
        "thriving_env": "Heavy clay loams, rice fallows, humid tropical environments; tolerates damp conditions & mild drought."
    },
    "proso_millet": {
        "common_name": "Proso Millet (Chetna / Panivaragu)",
        "regions": "India, China, Russia, USA",
        "thriving_env": "Short duration (60-75 days), highly water-efficient; grows in cool-to-warm temperate & dry tropical plains."
    },
    "sunflower": {
        "common_name": "Sunflower",
        "regions": "India (Karnataka, Maharashtra), Ukraine, Russia, Argentina",
        "thriving_env": "Warm sunny climates (20°C-28°C), deep well-drained neutral-to-slightly-alkaline loams or black soils."
    },
    "tobaco_godavari": {
        "common_name": "Tobacco (Godavari River Belt Cultivar)",
        "regions": "Andhra Pradesh (East & West Godavari River districts)",
        "thriving_env": "Deep black soils & light red soils; warm humid weather (20°C-32°C) with controlled leaf curing."
    },
    "wheat": {
        "common_name": "Wheat (Gehun)",
        "regions": "Northern/Central India (Punjab, Haryana, UP, MP), Global",
        "thriving_env": "Cool winter vegetative phase (10°C-15°C), warm sunny ripening phase (20°C-25°C), well-drained loams."
    },
    "white_ponni": {
        "common_name": "White Ponni Rice",
        "regions": "Southern India (Cauvery Delta Region - Tamil Nadu)",
        "thriving_env": "Rich river delta alluvial soils, high humidity (25°C-35°C), abundant irrigation/floodwater."
    }
}

# --- MODEL V2 CONFIGURATION ---
MODEL_PATH = "plant_model_v2.pth"
CLASS_NAMES_PATH = "class_names_v2.json"
MODEL_URL = "https://github.com/santhoshinikarunakaran03-oss/plant_classifier/releases/download/v2.0/plant_model_v2.pth"

@st.cache_resource
def load_model_and_classes():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading parallel model weights (v2)..."):
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

# --- UI INTERFACE ---
st.title("🌿 Precision Crop Variety Classifier (V2)")
st.write("Upload a leaf/crop image. Non-target plants will trigger an **Unrecognized** alert.")

if model is None:
    st.error("Missing class_names_v2.json file in repository.")
else:
    uploaded_file = st.file_uploader("Upload leaf photo...", type=["jpg", "jpeg", "png"])

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
            probs = F.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probs, 0)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
            
        conf_percentage = confidence.item() * 100
        predicted_class = class_names[predicted_idx.item()]
        
        # Dual-threshold Rejection Gate
        is_recognized = confidence.item() >= 0.70 and entropy <= 1.25
        
        if not is_recognized:
            st.warning("⚠️ **Plant Unrecognized / Out of Scope**")
            st.info("The uploaded image does not match any of the 16 target crop classes with high confidence.")
        else:
            st.success(f"🌱 **Predicted Variety:** {predicted_class}")
            st.info(f"📊 **Confidence Score:** {conf_percentage:.2f}%")
            
            # --- CROP KNOWLEDGE DISPLAY ---
            if predicted_class in CROP_INFO:
                info = CROP_INFO[predicted_class]
                st.write("---")
                st.subheader(f"📖 Agricultural Profile: {info['common_name']}")
                st.markdown(f"**📍 Primary Cultivation Regions:** {info['regions']}")
                st.markdown(f"**🌤️ Thriving Environment & Soil:** {info['thriving_env']}")
            
        # --- ACTIVE LEARNING DATASET COLLECTION ---
        st.write("---")
        st.subheader("🤖 Help Train Model V2")
        st.write("Confirming prediction accuracy uploads this image directly to our active learning dataset.")
        
        feedback = st.radio("Was the identification accurate?", ["Yes", "No"], index=0 if is_recognized else 1)
        
        corrected_label = predicted_class
        if feedback == "No":
            corrected_label = st.selectbox(
                "Select true crop variety (or 'Grass_Turf'):", 
                class_names + ["Grass_Turf", "Unknown_Plant"]
            )
            
        if st.button("Submit Image to Cloud Dataset"):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            remote_file_path = f"{corrected_label}/{timestamp}.jpg"
            
            temp_filename = "temp_upload_v2.jpg"
            image.save(temp_filename)
            
            if supabase:
                with open(temp_filename, "rb") as f:
                    supabase.storage.from_("plant-user-uploads").upload(file=f, path=remote_file_path)
                st.success(f"Successfully uploaded image under class folder **'{corrected_label}'** in Supabase!")
            else:
                st.error("Supabase connection not active. Please check Streamlit Secrets configuration.")
                
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
