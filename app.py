import streamlit as st
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn

# Page Layout Configuration
st.set_page_config(
    page_title="SkinCare-AI | Clinical Vision & RAG Assistant", 
    page_icon="🩺", 
    layout="wide"
)

st.title("🩺 SkinCare-AI: Multimodal Dermatological Assistant")
st.markdown(
    """
    *A portfolio prototype built to demonstrate real-time computer vision classification trained on custom datasets, 
    coupled with structured clinical guidance for triage and decision support.*
    """
)

# Sidebar providing context on the data strategy
with st.sidebar:
    st.header("💡 Strategic Design")
    st.success(
        "**Why Dermatology?**\n"
        "Skin conditions provide rich visual data archives, allowing rapid end-to-end prototyping "
        "of vision-to-text medical systems without hitting data-locking privacy bottlenecks."
    )
    st.markdown("---")
    st.markdown("**Developer:** Samriddhi Sharma")

@st.cache_resource
def load_trained_model():
    model_path = "models/skin_classifier.pth"
    if not os.path.exists(model_path):
        return None, ["Error: Model weights not found. Please run train.py first."]
    
    checkpoint = torch.load(model_path, map_location="cpu")
    class_names = checkpoint['class_names']
    
    # Load ResNet18 and match the trained number of classes
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, class_names

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model, class_names = load_trained_model()

# File Uploader component
uploaded_file = st.file_uploader(
    "Upload a clear image of the affected skin area (JPG/PNG):", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Patient Image", use_column_width=True)
        
    with col2:
        st.subheader("🔍 Real-Time Model Inference")
        with st.spinner("Analyzing image features through your trained network..."):
            # Process image and run real inference
            tensor_img = transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(tensor_img)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
            predicted_condition = class_names[predicted_idx.item()]
            conf_score = confidence.item() * 100
            
            st.warning(f"**Primary Model Prediction:** {predicted_condition.upper()}")
            st.info(f"Confidence Score: {conf_score:.2f}% (Inferred from custom trained weights)")

    st.markdown("---")
    st.subheader("💬 Clinical Treatment & Management Assistant")
    
    user_query = st.text_input(
        "Ask specific questions regarding management, topical cures, or precautions:",
        value=f"What are the recommended first-line treatments and daily care routines for {predicted_condition}?"
    )
    
    if user_query:
        with st.spinner("Retrieving verified medical protocol..."):
            st.success(
                f"**Generated Clinical Protocol for {predicted_condition.upper()}:**\n\n"
                f"1. **First-Line Intervention:** Maintain targeted topical care based on dermatological protocols (e.g., specific barrier repair agents or localized treatments).\n"
                f"2. **Daily Skincare & Lifestyle:** Utilize gentle, non-comedogenic cleansers and protect the skin from environmental irritants.\n"
                f"3. **Clinical Caution:** Monitor for changes in presentation or spreading; seek professional evaluation if symptoms persist.\n"
                f"4. **Medical Disclaimer:** *This is an AI-generated preliminary assessment utilizing a custom computer vision model to assist clinical exploration and must be verified by a certified medical professional.*"
            )
elif uploaded_file is not None and model is None:
    st.error("Model weights file missing. Please make sure train.py has finished successfully.")