#!/bin/bash

# 🌍 Tongue Classifier - Quick Setup Script

echo "=================================================="
echo "🌍 Tongue Health Classifier - Setup"
echo "=================================================="

# Check Python
python_version=$(python3 --version)
echo "✓ Python: $python_version"

# Create virtual environment (optional)
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
echo "🔌 Activating virtual environment..."
source venv/bin/activate  # Linux/macOS
# For Windows: venv\Scripts\activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=================================================="
echo "✅ Setup Complete!"
echo "=================================================="
echo ""
echo "📝 Next steps:"
echo "1. Place your model file (model.pt or best_model.pt) in the current directory"
echo "2. Run the app: streamlit run tongue_classifier_app.py"
echo "3. Open http://localhost:8501 in your browser"
echo ""
echo "📦 Files needed:"
echo "   ✓ tongue_classifier_app.py"
echo "   ✓ requirements.txt"
echo "   ✓ model.pt (571 MB - download from Colab)"
echo ""
echo "💡 Tips:"
echo "   - First load will download PyTorch (1.5 GB) if not cached"
echo "   - Model loads only once (cached)"
echo "   - Try small images first (~328x328)"
echo ""
