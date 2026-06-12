# 🌍 Tongue Health Classifier - Streamlit Deployment Guide

## 📋 Overview

Streamlit web app สำหรับ Tongue Health Classification ใช้ ConvNeXt Small model
- **Input Size:** 328×328 px
- **Classes:** 6 multi-label features
- **Model Size:** ~571 MB
- **Framework:** PyTorch + Streamlit

---

## 🚀 Quick Start (Local Testing)

### 1. **Install Dependencies**

```bash
pip install -r requirements.txt
```

### 2. **Prepare Files**

โครงสร้างไฟล์ต้องเป็นแบบนี้:

```
project/
├── tongue_classifier_app.py    # Streamlit app
├── requirements.txt             # Dependencies
├── model.pt  (หรือ best_model.pt)  # 571 MB trained model ⭐
└── README.md
```

### 3. **Run Locally**

```bash
streamlit run tongue_classifier_app.py
```

จากนั้นเปิด browser ไปที่: `http://localhost:8501`

---

## 📦 Model File Preparation

### 💾 Save Model from Colab

```python
# ในไฟล์ Colab notebook ของคุณ
import torch

# Save model (2 options)

# Option 1: Save only state_dict (เล็กกว่า, ต้อง define model architecture)
torch.save(model.state_dict(), 'model_weights.pt')

# Option 2: Save full checkpoint (มี metadata)
torch.save({
    'model_state_dict': model.state_dict(),
    'epoch': best_epoch,
    'best_val_f1': best_f1_score,
}, 'best_model.pt')

# ✅ ดาวน์โหลดจาก Colab Files
# Files > model.pt > right-click > Download
```

### ⚠️ Large File Tips

เนื่องจากไฟล์ 571 MB ใหญ่มาก:

**Option A: Git LFS (แนะนำ)**
```bash
# ติดตั้ง git-lfs
brew install git-lfs  # macOS
# หรือ apt install git-lfs  # Linux

# ใน repo
git lfs install
git lfs track "*.pt"
git add .gitattributes model.pt
git commit -m "Add large model file"
git push
```

**Option B: Compress Model**
```python
# ลด precision จาก float32 → float16
import torch

model.half()  # Convert to float16
torch.save(model.state_dict(), 'model_fp16.pt')

# ขนาดประมาณ 285 MB (ลดลงครึ่ง แต่อาจลดความแม่นยำเล็กน้อย)
```

---

## 🌍 Deployment Options

### **Option 1: Streamlit Cloud (ฟรี, ง่ายที่สุด)**

#### Setup

1. **สร้าง GitHub Repository**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/tongue-classifier
   git push -u origin main
   ```

2. **Deploy to Streamlit Cloud**
   - ไป https://share.streamlit.io
   - Click "New app"
   - เลือก Repository, Branch, Main file
   - App จะ deploy ใน ~2 นาที

**ข้อดี:**
- ✅ ฟรี
- ✅ ง่าย (no server config)
- ✅ Auto-deploy จากที่เปลี่ยน GitHub

**ข้อเสีย:**
- ❌ ไฟล์ 571 MB อาจ timeout (5 นาที limit)
- ❌ Cold start slow
- ❌ compute resource limited

#### Solution: Use `.gitignore` + Download from URL

```bash
# .gitignore
*.pt
model/

# แทน push model ไป GitHub
# ดาวน์โหลด model จาก Google Drive/Hugging Face ตอน runtime
```

ปรับ app code:

```python
@st.cache_resource
def load_model(device='cpu'):
    import gdown
    
    # Download from Google Drive (share link)
    model_url = 'https://drive.google.com/uc?id=YOUR_FILE_ID'
    output_path = 'model.pt'
    
    if not Path(output_path).exists():
        with st.spinner('Downloading model (~600MB)...'):
            gdown.download(model_url, output_path, quiet=False)
    
    # Load model
    return load_checkpoint(output_path, device)
```

---

### **Option 2: Hugging Face Spaces (แนะนำ)**

ดีที่สุดสำหรับโมเดลใหญ่เพราะมี Git LFS support

#### Setup

1. **สร้าง Space**
   - ไป https://huggingface.co/spaces
   - Click "Create new Space"
   - Owner: คุณ, Name: `tongue-classifier`
   - Space type: `Streamlit`
   - Click Create

2. **Clone & Push**
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/tongue-classifier
   cd tongue-classifier
   
   # Copy files
   cp ../tongue_classifier_app.py .
   cp ../requirements.txt .
   cp ../model.pt .
   
   # Push
   git add .
   git commit -m "Add tongue classifier"
   git push
   ```

3. **Done!** App จะ deploy อัตโนมัติ

**ข้อดี:**
- ✅ Support Git LFS (ไฟล์ใหญ่ได้)
- ✅ Free tier ดี
- ✅ HuggingFace community

**ข้อเสีย:**
- ❌ เริ่มตัวใหม่ทีละครั้งถ้า inactive (ฟรี tier)

---

### **Option 3: Railway / Render (Self-Hosted)**

สำหรับ control เต็มที่

#### Railway

```bash
# 1. Login
railway login

# 2. Init
railway init

# 3. Deploy
railway up

# 4. Set environment
railway variables set TORCH_HOME=/app/cache
```

**Cost:** ~$5/month for basic tier

#### Render

```bash
# 1. Connect GitHub repo
# ไป render.com > Create > Web Service

# 2. Settings
Build Command: pip install -r requirements.txt
Start Command: streamlit run tongue_classifier_app.py --server.port 10000

# 3. Deploy
```

---

## 🔧 Configuration Files

### `streamlit/config.toml` (optional)

```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[client]
maxUploadSize = 100  # 100 MB limit

[server]
maxUploadSize = 100
enableCORS = true
```

### `.streamlit/secrets.toml` (for API keys, etc)

```toml
# For Hugging Face remote model loading
hf_api_key = "hf_xxxxxxxxxxxx"

# For Google Drive access
google_drive_folder_id = "xxxxxx"
```

---

## 🎯 Performance Tips

### 1. **Model Caching**
```python
@st.cache_resource
def load_model(path, device='cpu'):
    # Model loads once, reused for all users
    return model
```

### 2. **Image Caching**
```python
@st.cache_data
def preprocess_image(uploaded_file):
    # Cache preprocessed images
    return img_tensor
```

### 3. **Lazy Loading**
```python
if uploaded_file is not None:
    # Only load model when needed
    model = load_model('model.pt')
```

### 4. **Reduce Model Size**
```python
# Quantize model (PyTorch dynamic quantization)
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
torch.save(quantized_model, 'model_quantized.pt')
# ขนาดลดลง ~75%
```

---

## 🐛 Troubleshooting

### ❌ "ModuleNotFoundError: No module named 'timm'"

```bash
pip install timm==0.9.12
```

### ❌ "CUDA out of memory"

```python
# Use CPU instead
device = "cpu"

# Or reduce batch size in preprocessing
img_size = 224  # instead of 328
```

### ❌ "Model file not found"

ตรวจสอบ:
1. ชื่อไฟล์ถูกต้อง (`model.pt` หรือ `best_model.pt`)
2. ไฟล์อยู่ใน directory เดียวกับ app.py
3. File size ถูกต้อง (~571 MB)

```python
# Debug
import os
print("Files in current directory:")
print(os.listdir('.'))
```

### ❌ Streamlit Cloud timeout

**Fix:** Don't upload model to GitHub, download at runtime

```python
import gdown

@st.cache_resource
def load_model():
    if not os.path.exists('model.pt'):
        gdown.download(
            'https://drive.google.com/uc?id=YOUR_ID',
            'model.pt',
            quiet=False
        )
    return load_checkpoint('model.pt', 'cpu')
```

---

## 📊 Monitor & Update

### Check Deployment

```bash
# Streamlit Cloud
streamlit logs YOUR_APP_NAME

# Railway
railway logs

# Render
# View in Render dashboard
```

### Update Model

```bash
# Git approach
git add model.pt
git commit -m "Update model with better accuracy"
git push

# If using remote download, just update Google Drive link
```

---

## 📝 Example Deployment Checklist

- [ ] Model file ready (571 MB .pt)
- [ ] `tongue_classifier_app.py` ready
- [ ] `requirements.txt` updated
- [ ] `.gitignore` configured (if needed)
- [ ] `.streamlit/config.toml` created
- [ ] GitHub repo created
- [ ] Deployment platform selected
- [ ] App deployed & tested
- [ ] Share link with team

---

## 🔗 Useful Links

- Streamlit Docs: https://docs.streamlit.io
- PyTorch Model Saving: https://pytorch.org/docs/stable/notes/serialization.html
- Timm Models: https://github.com/huggingface/pytorch-image-models
- Streamlit Cloud: https://share.streamlit.io
- Hugging Face Spaces: https://huggingface.co/spaces

---

## ❓ Questions?

1. **Model loading slow?** → Use remote download, cache aggressively
2. **GPU needed?** → Add `if st.sidebar.checkbox('Use GPU'):`
3. **Multiple models?** → Use `st.selectbox()` to switch models
4. **Want CI/CD?** → Set up GitHub Actions for auto-testing

Good luck! 🚀
