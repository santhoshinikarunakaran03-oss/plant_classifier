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

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AgriVision AI | Precision Crop Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS INJECTION ---
st.markdown("""
    <style>
    /* Main Background & Fonts */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header Container Styling */
    .header-box {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .header-box h1 {
        color: white !important;
        margin: 0;
        font-weight: 700;
    }
    .header-box p {
        color: #e0e0e0;
        margin-top: 6px;
        font-size: 1.05rem;
    }

    /* Metric & Prediction Cards */
    .custom-card {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    
    /* Active Learning Container */
    .feedback-box {
        background-color: #f0f7ff;
        border-left: 5px solid #0066cc;
        padding: 20px;
        border-radius: 6px;
        margin-top: 15px;
    }
    
    /* Hide Default Streamlit Menu Padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    </style>
""", unsafe_allow_html=True)

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

# --- MODEL CONFIGURATION ---
LOCAL_MODEL_FILE = "plant_model_v2_1.pth"
CLASS_NAMES_PATH = "class_names_v2.json"
MODEL_URL = "https://github.com/santhoshinikarunakaran03-oss/plant_classifier/releases/download/v2.1/plant_model.pth"

@st.cache_resource
def load_model_and_classes():
    # Force fresh download if v2.1 file does not exist locally on the cloud container
    if not os.path.exists(LOCAL_MODEL_FILE):
        with st.spinner("📦 Synchronizing neural network weights (v2.1)..."):
            try:
                opener = urllib.request.build_opener()
                opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
                urllib.request.install_opener(opener)
                urllib.request.urlretrieve(MODEL_URL, LOCAL_MODEL_FILE)
            except Exception as e:
                st.error(f"Failed to fetch model weights from v2.1 release: {e}")
                return None, None

    if not os.path.exists(CLASS_NAMES_PATH):
        st.error("Missing target class metadata file (class_names_v2.json).")
        return None, None
        
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
        
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(LOCAL_MODEL_FILE, map_location=torch.device('cpu')))
    model.eval()
    return model, class_names

model, class_names = load_model_and_classes()

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/leaf.png", width=64)
    st.title("AgriVision Control Hub")
    st.markdown("---")
    st.markdown("### 📊 System Status")
    st.success("🟢 Model Engine: **ResNet-18 (v2.0)**")
    if supabase:
        st.success("🟢 Cloud Sync: **Active (Supabase)**")
    else:
        st.warning("🟠 Cloud Sync: **Offline**")
    
    st.markdown("---")
    st.markdown("### 🌾 Target Scope (16 Classes)")
    if class_names:
        st.caption(", ".join(class_names[:8]) + ", and 8 more...")
    
    st.markdown("---")
    st.caption("Developed for Precision Agriculture & Active Learning")

# --- HEADER SECTION ---
st.markdown("""
    <div class="header-box">
        <h1>🌿 Precision Crop Classifier & Field Guide</h1>
        <p>AI-powered crop identification with automated dual-threshold rejection gating and real-time active learning feedback.</p>
    </div>
""", unsafe_allow_html=True)

if model is None:
    st.error("Model runtime initialization failed. Check GitHub releases and logs.")
else:
    # --- TWO COLUMN LAYOUT ---
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📤 Upload Leaf Sample")
        uploaded_file = st.file_uploader("Select crop image (JPG, PNG)...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Field Sample", use_container_width=True)

    with col2:
        if uploaded_file is not None:
            st.subheader("🔍 Neural Inference Result")
            
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
                st.warning("⚠️ **Out of Scope / Unrecognized Plant**")
                st.info("The sample does not match any of the 16 target crop profiles with sufficient statistical confidence.")
            else:
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.metric(label="Predicted Variety", value=predicted_class)
                with m_col2:
                    st.metric(label="Confidence Score", value=f"{conf_percentage:.1f}%")
                
                # --- CROP KNOWLEDGE PROFILE ---
                if predicted_class in CROP_INFO:
                    info = CROP_INFO[predicted_class]
                    st.markdown("### 📖 Agricultural Profile")
                    st.markdown(f"**Variety Name:** {info['common_name']}")
                    st.markdown(f"**📍 Cultivation Zones:** {info['regions']}")
                    st.markdown(f"**🌤️ Optimal Environment:** {info['thriving_env']}")

    # --- FULL-WIDTH ACTIVE LEARNING PANEL ---
    if uploaded_file is not None:
        st.write("---")
        st.subheader("🤖 Active Learning Field Data Collection")
        
        with st.container():
            st.write("Confirming prediction accuracy directly uploads this image to our cloud dataset for automatic model retraining.")
            
            fb_col1, fb_col2 = st.columns([1, 2])
            with fb_col1:
                feedback = st.radio("Is this identification correct?", ["Yes", "No"], index=0 if is_recognized else 1)
            
            corrected_label = predicted_class
            with fb_col2:
                if feedback == "No":
                    corrected_label = st.selectbox(
                        "Select actual plant variety:", 
                        class_names + ["Grass_Turf", "Unknown_Plant"]
                    )
                
            if st.button("🚀 Upload Sample to Cloud Dataset", type="primary"):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                remote_file_path = f"{corrected_label}/{timestamp}.jpg"
                
                temp_filename = "temp_upload_v2.jpg"
                image.save(temp_filename)
                
                if supabase:
                    try:
                        with open(temp_filename, "rb") as f:
                            supabase.storage.from_("plant-user-uploads").upload(
                                file=f, 
                                path=remote_file_path,
                                file_options={"cache-control": "3600", "upsert": "true"}
                            )
                        st.success(f"Sample registered under class **'{corrected_label}'** in Supabase Storage!")
                    except Exception as upload_err:
                        st.error(f"Failed to upload to Supabase: {upload_err}")
                else:
                    st.error("Supabase credentials missing. Check Streamlit Secrets.")
                    
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
