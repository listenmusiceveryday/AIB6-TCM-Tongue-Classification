"""
🌍 Tongue Health Classification Streamlit App
Based on ConvNeXt Small (328x328 input)
Multi-label classification: 6 tongue features
"""

import streamlit as st
import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Tongue Health Classifier",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# STYLING
# ============================================================================
st.markdown("""
<style>
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .high-confidence {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .medium-confidence {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    .low-confidence {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    }
    .title-section {
        text-align: center;
        margin: 20px 0;
    }
    .info-box {
        background: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIG
# ============================================================================
CONFIG = {
    'img_size': 328,
    'model_name': 'convnext_small',
    'num_labels': 6,
    'label_columns': [
        'Coating_White',
        'Coating_Yellow',
        'Texture_None',
        'Texture_Geographic',
        'Texture_Cracked',
        'Texture_Dentate'
    ],
    'threshold': 0.5,
}

LABEL_DESCRIPTIONS = {
    'Coating_White': '⚪ White coating on tongue surface',
    'Coating_Yellow': '💛 Yellow coating on tongue surface',
    'Texture_None': '✓ No visible texture abnormalities',
    'Texture_Geographic': '🗺️ Geographic/Map-like pattern',
    'Texture_Cracked': '🔀 Cracked appearance',
    'Texture_Dentate': '👁️ Teeth/scalloped edges (Dentate)',
}

# ============================================================================
# MODEL LOADING (CACHED)
# ============================================================================

class ConvNeXtClassifier(nn.Module):
    """Multi-label classifier using ConvNeXt backbone"""
    
    def __init__(self, model_name='convnext_small', num_labels=6, 
                 pretrained=True, dropout=0.3, freeze_backbone=False):
        super().__init__()
        
        # Load backbone
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained,
            num_classes=0, global_pool='avg'
        )
        self.feature_dim = self.backbone.num_features
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_labels)
        )
        
        self._init_classifier()
    
    def _init_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


@st.cache_resource
def load_model(device='cpu'):
    """Load model from Google Drive"""
    try:
        # Download from Google Drive
        import gdown
        
        # ← เปลี่ยนตรงนี้เป็น FILE_ID ของคุณ
        FILE_ID = "PASTE_YOUR_FILE_ID_HERE"
        model_path = "model.pt"
        
        # Download ถ้ายังไม่มี
        if not os.path.exists(model_path):
            url = f'https://drive.google.com/uc?id={FILE_ID}'
            gdown.download(url, model_path, quiet=False)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Initialize model
        model = ConvNeXtClassifier(
            model_name=CONFIG['model_name'],
            num_labels=CONFIG['num_labels'],
            pretrained=False,  # We're loading trained weights
            dropout=0.3
        )
        
        # Load weights
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None


# ============================================================================
# PREPROCESSING
# ============================================================================

def get_inference_transforms(img_size=328):
    """Get transforms for inference (no augmentation)"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            always_apply=True
        ),
        ToTensorV2()
    ])


def preprocess_image(image, transforms):
    """Preprocess image for model input"""
    # Convert PIL to numpy
    if isinstance(image, Image.Image):
        image = np.array(image.convert('RGB'))
    
    # Apply transforms
    transformed = transforms(image=image)
    img_tensor = transformed['image']
    
    # Add batch dimension
    return img_tensor.unsqueeze(0)


# ============================================================================
# INFERENCE
# ============================================================================

def predict(model, image_tensor, device):
    """Run inference and return probabilities"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits)
    
    return probabilities[0].cpu().numpy()


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Title
    st.markdown(
        "<div class='title-section'><h1>🌍 Tongue Health Classifier</h1>"
        "<p>Multi-label classification using ConvNeXt-Small</p></div>",
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Device selection
        device_option = st.radio(
            "Select device:",
            ["CPU", "CUDA (GPU)"],
            help="GPU is faster but requires CUDA-enabled GPU"
        )
        device = "cuda" if device_option == "CUDA (GPU)" and torch.cuda.is_available() else "cpu"
        
        if device == "cuda":
            st.success(f"✅ Using {torch.cuda.get_device_name(0)}")
        else:
            st.info("Using CPU (slower)")
        
        # Threshold
        threshold = st.slider(
            "Classification Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Probability threshold for positive class"
        )
        
        st.divider()
        
        # Info
        st.subheader("📊 Model Info")
        st.write(f"**Architecture:** ConvNeXt Small")
        st.write(f"**Input Size:** {CONFIG['img_size']}×{CONFIG['img_size']}")
        st.write(f"**Classes:** {CONFIG['num_labels']} (Multi-label)")
        st.write(f"**Backbone:** ImageNet-1K pretrained")
    
    # Load model
    model_placeholder = st.empty()
    
    with model_placeholder.container():
        with st.spinner("🔄 Loading model..."):
            # Try to load from different possible paths
            model_paths = [
                "model.pt",
                "best_model.pt",
                "./best_model.pt",
            ]
            
            model = None
            for path in model_paths:
                try:
                    model = load_model(path, device=device)
                    if model is not None:
                        break
                except:
                    continue
            
            if model is None:
                st.error(
                    "❌ Model file not found. Please ensure 'model.pt' or 'best_model.pt' "
                    "is in the same directory as this app."
                )
                st.info("📝 Expected files: `model.pt` or `best_model.pt`")
                st.stop()
    
    model_placeholder.success("✅ Model loaded successfully!")
    
    # File uploader
    st.header("📸 Upload Tongue Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("ℹ️ Image Info")
            st.write(f"**Size:** {image.size}")
            st.write(f"**Format:** {image.format}")
            st.write(f"**Mode:** {image.mode}")
        
        # Inference button
        if st.button("🔍 Classify Tongue", key="classify_btn", use_container_width=True):
            with st.spinner("🤖 Analyzing tongue..."):
                # Preprocess
                transforms = get_inference_transforms(CONFIG['img_size'])
                img_tensor = preprocess_image(image, transforms)
                
                # Predict
                probabilities = predict(model, img_tensor, device)
                
                # Display results
                st.divider()
                st.subheader("📊 Classification Results")
                
                # Get predictions above threshold
                predictions = probabilities > threshold
                predicted_labels = [
                    CONFIG['label_columns'][i]
                    for i in range(len(CONFIG['label_columns']))
                    if predictions[i]
                ]
                
                # Summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Features Detected", len(predicted_labels))
                with col2:
                    max_prob = probabilities.max()
                    st.metric("Max Confidence", f"{max_prob:.1%}")
                with col3:
                    avg_prob = probabilities.mean()
                    st.metric("Avg Confidence", f"{avg_prob:.1%}")
                
                st.divider()
                
                # Detailed results
                st.subheader("🏥 Detailed Analysis")
                
                # Sort by probability
                sorted_indices = np.argsort(probabilities)[::-1]
                
                for idx in sorted_indices:
                    label = CONFIG['label_columns'][idx]
                    prob = probabilities[idx]
                    is_positive = predictions[idx]
                    
                    # Color based on confidence
                    if prob > 0.7:
                        color = "high-confidence"
                        confidence_text = "🟢 High"
                    elif prob > 0.4:
                        color = "medium-confidence"
                        confidence_text = "🟡 Medium"
                    else:
                        color = "low-confidence"
                        confidence_text = "🔴 Low"
                    
                    # Display
                    col1, col2, col3 = st.columns([2, 1, 3])
                    
                    with col1:
                        status = "✓ Present" if is_positive else "✗ Absent"
                        st.write(f"**{label}** {status}")
                        st.caption(LABEL_DESCRIPTIONS[label])
                    
                    with col2:
                        st.write(f"{prob:.1%}")
                    
                    with col3:
                        st.progress(prob)
                
                st.divider()
                
                # Summary text
                if len(predicted_labels) > 0:
                    st.success(
                        f"**Detected Features:** {', '.join(predicted_labels)}"
                    )
                else:
                    st.info(
                        f"No features detected above {threshold:.0%} threshold. "
                        "Tongue appears normal."
                    )
    
    # Footer
    st.divider()
    st.markdown("""
    <div class='info-box'>
    <p><strong>⚕️ Disclaimer:</strong> This classifier is for research purposes only. 
    It should NOT be used for medical diagnosis or treatment decisions. 
    Always consult a healthcare professional for medical concerns.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
