# 🥔 Potato Skin Disease Detection Using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> 🔬 An AI-powered computer vision system for detecting and classifying potato skin diseases using deep learning techniques.

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [🌟 Features](#-features)
- [📊 Dataset](#-dataset)
- [🚀 Getting Started](#-getting-started)
- [💻 Usage](#-usage)
- [🏗️ Model Architecture](#-model-architecture)
- [📈 Results](#-results)
- [🚀 Next Steps](#-Next-Steps)
- [📄 License](#-license)

## 🎯 Project Overview

This project implements a **Convolutional Neural Network (CNN)** using TensorFlow/Keras to automatically detect and classify potato skin diseases from digital images. The system can identify three main categories:

- 🍃 **Healthy Potatoes**
- 🦠 **Early Blight Disease**
- 🍄 **Late Blight Disease**

### 🎥 Demo

<details>
<summary>Click to see sample predictions</summary>

```
Input: potato_image.jpg
Output: "Early Blight Disease" (Confidence: 94.2%)
```

</details>

## 🌟 Features

- ✅ **Multi-class Classification**: Detects 3 types of potato conditions
- ✅ **Data Augmentation**: Improves model robustness with image transformations
- ✅ **Interactive Visualization**: Displays sample images with predictions
- ✅ **Optimized Performance**: Uses caching and prefetching for faster training
- ✅ **Scalable Architecture**: Easy to extend to more disease types
- ✅ **Real-time Inference**: Fast prediction on new images

## 📊 Dataset

### 📈 Dataset Statistics

- **Total Images**: 2,152
- **Classes**: 3 (Early Blight, Late Blight, Healthy)
- **Image Size**: 256×256 pixels
- **Color Channels**: RGB (3 channels)
- **Data Split**: 80% Train, 10% Validation, 10% Test

## 📁 Project Structure

```
potato-disease-detection/
├── 📓 POTATO_Skin_Diseases_Detection_Using_Deep_Learning.ipynb
├── 📄 README.md
├── 📋 requirements.txt
├── 📁 PlantVillage/
│   ├── 📁 Potato___Early_blight/
│   ├── 📁 Potato___Late_blight/
│   └── 📁 Potato___healthy/
├── 📁 models/
│   └── 💾 trained_model.h5
└── 📁 results/
    ├── 📊 training_plots.png
    └── 📈 confusion_matrix.png
```

📂 Root Directory/
├── 🐍 app.py # Main Flask application
├── 📦 requirements.txt # Dependencies
├── 🚀 run_flask_app.bat # Easy startup script
├── 📚 README_Flask.md # Complete documentation
├── 📂 templates/
│ └── 🌐 index.html # Web interface
└── 📂 static/
├── 📂 css/
│ └── 💄 style.css # Beautiful styling
└── 📂 js/
└── ⚡ script.js # Interactive functionality

## 🚀 Getting Started

### 📋 Prerequisites

```bash
Python 3.8+
TensorFlow 2.x
Matplotlib
NumPy
```

### ⚡ Quick Start and Installation

### 🐍 Environment Setup

```bash
# Create virtual environment
python -m venv potato_env

# Activate environment
# Windows:
potato_env\Scripts\activate
# macOS/Linux:
source potato_env/bin/activate

# Install packages
pip install -r requirements.txt
```

# Run Application

#### **Step 1: Install Dependencies**

```cmd
pip install -r requirements.txt
```

#### **Step 2: Run the Application**

```cmd
python app.py
```

#### **Step 3: Open Your Browser**

- **Main App**: http://localhost:5000
- **Health Check**: http://localhost:5000/health

## 💻 Usage

### 🔧 Training the Model

The notebook includes the complete pipeline:

1. **Data Loading & Preprocessing**

   ```python
   # Load dataset
   dataset = tf.keras.preprocessing.image_dataset_from_directory(
       "PlantVillage",
       image_size=(256, 256),
       batch_size=32
   )
   ```

2. **Data Augmentation**

   ```python
   # Apply data augmentation
   data_augmentation = tf.keras.Sequential([
       tf.keras.layers.RandomFlip("horizontal_and_vertical"),
       tf.keras.layers.RandomRotation(0.2)
   ])
   ```

3. **Model Configuration**
   ```python
   IMAGE_SIZE = 256
   BATCH_SIZE = 32
   CHANNELS = 3
   EPOCHS = 50
   ```

### 🎯 Making Predictions

```python
# Load your trained model
model = tf.keras.models.load_model('potato_disease_model.h5')

# Make prediction
prediction = model.predict(new_image)
predicted_class = class_names[np.argmax(prediction)]
```

## 🏗️ Model Architecture

### 🧠 Network Components

1. **Input Layer**: 256×256×3 RGB images
2. **Preprocessing**:
   - Image resizing and rescaling (1.0/255)
   - Data augmentation (RandomFlip, RandomRotation)
3. **Feature Extraction**: CNN layers for pattern recognition
4. **Classification**: Dense layers for final prediction

### ⚙️ Training Configuration

- **Optimizer**: Adam (recommended)
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 50
- **Batch Size**: 32

## 📈 Results

### 📊 Performance Metrics

| Metric              | Score |
| ------------------- | ----- |
| Training Accuracy   | XX.X% |
| Validation Accuracy | XX.X% |
| Test Accuracy       | XX.X% |
| F1-Score            | XX.X% |

### 🎨 Visualization

The notebook includes:

- ✅ Sample image visualization
- ✅ Training/validation loss curves
- ✅ Confusion matrix
- ✅ Class-wise accuracy

# 🥔 Potato Disease Detection - Flask Web Application

A modern Flask web application for detecting potato diseases using deep learning. Upload images or use your camera to get instant disease predictions with confidence scores and treatment recommendations.

## ✨ Features

### 🖼️ **Dual Input Methods**

- **📁 File Upload**: Drag & drop or browse to select images
- **📸 Camera Capture**: Take photos directly from your device camera

### 🧠 **AI-Powered Detection**

- **🎯 Accurate Predictions**: Uses trained CNN model for disease detection
- **📊 Confidence Scores**: Shows prediction confidence with color-coded badges
- **📈 Probability Breakdown**: Displays probabilities for all disease classes

### 💡 **Smart Recommendations**

- **🏥 Treatment Advice**: Provides specific recommendations for each condition
- **🚨 Urgency Levels**: Different advice based on disease severity
- **📋 Downloadable Reports**: Generate and download analysis reports

### 🎨 **Modern Interface**

- **📱 Responsive Design**: Works perfectly on mobile and desktop
- **🌟 Beautiful UI**: Modern design with smooth animations
- **🔄 Real-time Analysis**: Instant predictions with loading indicators

## 🦠 Detected Diseases

1. **🍂 Early Blight** - Common fungal disease affecting potato leaves
2. **💀 Late Blight** - Serious disease that can destroy entire crops
3. **✅ Healthy** - No disease detected

## 🎯 How to Use

### **📁 Upload Method**

1. **Select Upload** tab (default)
2. **Drag & drop** an image or **click to browse**
3. **Click "Analyze Disease"** button
4. **View results** with predictions and recommendations

### **📸 Camera Method**

1. **Click Camera** tab
2. **Click "Start Camera"** (allow permissions)
3. **Click "Capture Photo"** when ready
4. **Click "Analyze Disease"** button
5. **View results** with predictions and recommendations

### **📊 Understanding Results**

- **🎯 Primary Diagnosis**: Main prediction with confidence score
- **📈 Probability Breakdown**: All disease probabilities
- **💡 Recommendations**: Treatment and care advice
- **📋 Download Report**: Save results as text file

## 🔧 Technical Details

- **🐍 Backend**: Flask 2.3+ with Python
- **🧠 AI Model**: TensorFlow/Keras CNN
- **🖼️ Image Processing**: PIL/Pillow for preprocessing
- **🎨 Frontend**: HTML5, CSS3, Vanilla JavaScript
- **📱 Camera**: WebRTC getUserMedia API
- **💾 Storage**: Local file system for uploads

## 📋 Requirements

- **🐍 Python**: 3.8+ (Recommended: 3.10+)
- **💻 OS**: Windows, macOS, or Linux
- **🧠 Memory**: 4GB+ RAM (8GB recommended)
- **💾 Storage**: ~2GB for dependencies and models
- **🌐 Browser**: Chrome, Firefox, Safari, Edge (latest versions)

## 🛠️ Troubleshooting

### ❌ **Model Not Loading**

```
Error: Model not loaded! Please check the model file path.
```

**Solution:**

- Ensure `models/1.h5` exists
- Check TensorFlow installation: `pip install tensorflow>=2.13.0`

### ❌ **Camera Not Working**

```
Could not access camera. Please check permissions.
```

**Solution:**

- Allow camera permissions in your browser
- Use HTTPS for camera access (or localhost)
- Check if another app is using the camera

### ❌ **Port Already in Use**

```
Address already in use
```

**Solution:**

- Close other Flask applications
- Change port in `app.py`: `app.run(port=5001)`
- Kill process: `taskkill /f /im python.exe` (Windows)

### ❌ **File Upload Issues**

```
Invalid file type or File too large
```

**Solution:**

- Use supported formats: PNG, JPG, JPEG
- Keep file size under 16MB
- Check image isn't corrupted

## 🎨 Customization

### **🎯 Add New Disease Classes**

1. Update `CLASS_NAMES` in `app.py`
2. Add descriptions in `CLASS_DESCRIPTIONS`
3. Update recommendations in `get_recommendations()`
4. Retrain model with new classes

## 📱 Mobile Responsiveness

The application is now **fully responsive** and optimized for mobile devices:

### 📲 Mobile Features:

- ✅ **Touch-friendly interface** with larger touch targets (44px minimum)
- ✅ **Responsive design** that adapts to screen sizes from 320px to desktop
- ✅ **Mobile camera support** with environment (back) camera preference
- ✅ **Optimized image display** for mobile viewports
- ✅ **Landscape/Portrait orientation** support
- ✅ **iOS Safari compatibility** with viewport fixes
- ✅ **Prevent accidental zoom** on form inputs
- ✅ **Touch-optimized drag & drop** for file uploads

### **🎨 Modify UI**

- **Colors**: Edit CSS variables in `style.css`
- **Layout**: Modify templates in `templates/`
- **Functionality**: Update JavaScript in `static/js/`

### **⚙️ Configuration**

- **Upload size**: Change `MAX_CONTENT_LENGTH` in `app.py`
- **Image size**: Modify `IMAGE_SIZE` parameter
- **Port**: Update `app.run(port=5000)` line

## 🔒 Security Notes

- **🚫 Production Use**: This is for development/research only
- **🔐 Secret Key**: Change `app.secret_key` for production
- **📁 File Validation**: Only accepts image files
- **💾 File Cleanup**: Consider automatic cleanup of old uploads

## 📈 Performance Tips

- **📸 Image Quality**: Use clear, well-lit potato leaf images
- **🎯 Focus**: Ensure leaves fill most of the frame
- **📏 Size**: Optimal size is 256x256 pixels or larger
- **🌟 Lighting**: Good natural lighting gives best results

## 🌐 Browser Compatibility

- ✅ **Chrome**: 90+
- ✅ **Firefox**: 88+
- ✅ **Safari**: 14+
- ✅ **Edge**: 90+
- ⚠️ **Mobile**: Camera features may vary

## 📄 API Endpoints

- `GET /` - Main web interface
- `POST /predict` - Upload image prediction
- `POST /predict_camera` - Camera image prediction
- `GET /health` - Application health check

## 🤝 Support

For issues or questions:

1. Check the troubleshooting section above
2. Verify your Python and dependencies versions
3. Ensure model files are in the correct location
4. Test with the provided sample images

---

## 🚀 Next Steps

### 🔮 Future Enhancements

- [ ] **Model Optimization**: Implement transfer learning with pre-trained models
- [ ] **Web Application**: Create a Flask/Streamlit web interface
- [ ] **Mobile App**: Develop a mobile application for field use
- [ ] **More Diseases**: Expand to detect additional potato diseases
- [ ] **Real-time Detection**: Implement live camera feed processing
- [ ] **API Development**: Create REST API for integration

### 🎯 Improvement Ideas

- [ ] **Hyperparameter Tuning**: Optimize model parameters
- [ ] **Cross-validation**: Implement k-fold cross-validation
- [ ] **Ensemble Methods**: Combine multiple models
- [ ] **Data Balancing**: Handle class imbalance if present

### 🐛 Bug Reports

If you find a bug, please create an issue with:

- Description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information

### 💡 Feature Requests

For new features, please provide:

- Clear description of the feature
- Use case and benefits
- Implementation suggestions```

# ==================DEBUGGING AND TROUBLESHOOTING GUIDE:===========================

# 🥔 Potato Disease Detection - Upload Functionality Guide

## 🚀 Quick Start

1. **Run the Application**:

   ```bash
   python app.py
   ```

   Or double-click `run_and_test.bat`

2. **Access the App**:
   - Main app: http://localhost:5000
   - Debug upload page: http://localhost:5000/debug
   - Health check: http://localhost:5000/health

## 📋 Testing Upload Functionality

### Step 1: Check System Health

1. Go to http://localhost:5000/debug
2. Click "🔍 Check System Health"
3. Verify all items show ✅:
   - Status: healthy
   - Model Loaded: Yes
   - Upload Dir Exists: Yes
   - Upload Dir Writable: Yes

### Step 2: Test Upload Directory

1. Click "📂 Test Upload Directory"
2. Should show "Upload directory is working correctly"

### Step 3: Test Image Upload

1. Click "📁 Click here to select an image" or drag an image
2. Select a potato leaf image (JPG, PNG, JPEG)
3. Preview should appear
4. Click "🔬 Analyze Disease"
5. Results should show:
   - Disease name and confidence
   - Recommendations
   - The analyzed image displayed

## 🔧 Troubleshooting Upload Issues

### Issue: "No file uploaded" Error

**Solutions:**

1. Ensure you're clicking the upload area or browse link
2. Check browser console for JavaScript errors (F12)
3. Try the debug page: http://localhost:5000/debug
4. **Mobile**: Tap firmly on upload area, wait for file picker

### Issue: File Not Saving

**Solutions:**

1. Check upload directory permissions:
   ```bash
   mkdir static/uploads
   ```
2. Run as administrator if on Windows
3. Check disk space
4. **Mobile**: Ensure stable network connection

### Issue: Camera Not Working (Mobile)

**Solutions:**

1. **Grant camera permissions** when prompted
2. **Use HTTPS** for camera access on mobile (required by browsers)
3. **Check camera availability** - some devices block camera access
4. **Try different browsers** (Chrome/Safari work best)
5. **Close other camera apps** that might be using the camera

### Issue: Touch/Tap Not Working (Mobile)

**Solutions:**

1. **Clear browser cache** and reload
2. **Disable browser zoom** if enabled
3. **Try two-finger tap** if single tap doesn't work
4. **Check touch targets** - buttons should be at least 44px
5. **Restart browser app** on mobile device

### Issue: Image Too Small/Large on Mobile

**Solutions:**

1. **Portrait orientation** usually works better
2. **Pinch to zoom** on images if needed
3. **Landscape mode** available for wider screens
4. **Image auto-resizes** based on screen size

### Issue: Slow Performance on Mobile

**Solutions:**

1. **Close other browser tabs** to free memory
2. **Use smaller image files** (under 5MB recommended)
3. **Ensure good network connection** for uploads
4. **Clear browser cache** regularly
5. **Restart browser** if app becomes unresponsive

### Issue: Model Not Loading

**Solutions:**

1. Verify model file exists: `models/1.h5`
2. Install required packages:
   ```bash
   pip install tensorflow pillow flask
   ```

### Issue: JavaScript Errors

**Solutions:**

1. Clear browser cache (Ctrl+F5)
2. Check browser console (F12)
3. Try a different browser
4. Disable browser extensions

### Issue: Image Not Displaying in Results

**Solutions:**

1. Check browser network tab (F12) for failed requests
2. Verify uploaded file in `static/uploads/` folder
3. Check Flask console for file save errors

## 🧪 Debug Features

### Console Logging

The JavaScript includes extensive console logging. Open browser developer tools (F12) to see:

- File selection events
- Upload progress
- Server responses
- Error details

### Debug Endpoints

- `/health` - System status
- `/debug/upload-test` - Upload directory test
- `/debug` - Interactive upload test page

### Manual Testing

1. **File Input Test**:

   ```javascript
   document.getElementById("fileInput").click();
   ```

2. **Check Selected File**:

   ```javascript
   console.log(selectedFile);
   ```

3. **Test FormData**:
   ```javascript
   const formData = new FormData();
   formData.append("file", selectedFile);
   console.log([...formData.entries()]);
   ```

## 💡 Tips for Success

1. **Use supported image formats**: JPG, PNG, JPEG, GIF
2. **Keep file size under 16MB**
3. **Use clear potato leaf images**
4. **Check browser compatibility** (modern browsers work best)
5. **Enable JavaScript**
6. **Allow camera permissions** (for camera capture feature)

## 🆘 Getting Help

If upload functionality still doesn't work:

1. **Check Flask console output** for error messages
2. **Check browser console** (F12 → Console tab)
3. **Try the debug page** at `/debug`
4. **Test with different image files**
5. **Restart the Flask app**
6. **Check file permissions** on the upload directory

## 🎯 Expected Results

After successful upload and analysis:

- ✅ Disease classification (Early Blight, Late Blight, or Healthy)
- ✅ Confidence percentage
- ✅ Treatment recommendations
- ✅ Analyzed image displayed in results
- ✅ Timestamp of analysis

# PDF Report Download Upgrade Guide

## 🎉 New Features Added

### ✨ **PDF Format**

- Professional PDF reports instead of simple text files
- Includes header, footer, tables, and proper formatting
- Company branding and professional layout

### 📁 **Folder Selection**

- Choose where to save your PDF reports
- Modern file picker dialog (supported browsers)
- Automatic fallback to default downloads folder

### 🎨 **Enhanced Report Content**

- **Report Header**: Timestamp, analysis method, model version
- **Analyzed Image**: Embedded image (if available)
- **Diagnosis Section**: Disease name, confidence, risk assessment
- **Probability Breakdown**: Table showing all class probabilities
- **Treatment Recommendations**: Numbered list of actionable advice
- **Professional Footer**: Branding and copyright information

## 🚀 Installation Requirements

Add to your `requirements.txt`:

```
reportlab>=4.0.0
```

Install the new dependency:

```bash
pip install reportlab>=4.0.0
```

# PDF Generation Troubleshooting Guide

## 🔧 If PDF Generation is Failing

### Quick Fix Steps

1. **Install ReportLab Library**

   ```bash
   pip install reportlab>=4.0.0
   ```

2. **Run Installation Script**

   - **Windows**: Double-click `install_pdf_deps.bat`
   - **Linux/Mac**: Run `bash install_pdf_deps.sh`

3. **Restart the Application**
   ```bash
   python app.py
   ```

### Common Issues and Solutions

#### ❌ **"ReportLab not available" Error**

**Problem**: ReportLab library is not installed.

**Solution**:

```bash
pip install reportlab
# or
pip install reportlab>=4.0.0
```

**Alternative**: Use virtual environment

```bash
python -m venv pdf_env
source pdf_env/bin/activate  # Linux/Mac
# or
pdf_env\Scripts\activate     # Windows
pip install reportlab
```

#### ❌ **"Permission denied" or "Access denied" Errors**

**Problem**: Insufficient permissions to install packages.

**Solutions**:

1. **Use --user flag**:

   ```bash
   pip install --user reportlab
   ```

2. **Run as administrator** (Windows):

   - Right-click Command Prompt → "Run as administrator"
   - Then run: `pip install reportlab`

3. **Use sudo** (Linux/Mac):
   ```bash
   sudo pip install reportlab
   ```

#### ❌ **"Module not found" Error Despite Installation**

**Problem**: ReportLab installed in different Python environment.

**Solutions**:

1. **Check Python version**:

   ```bash
   python --version
   which python  # Linux/Mac
   where python  # Windows
   ```

2. **Install for specific Python version**:

   ```bash
   python3 -m pip install reportlab
   # or
   python3.9 -m pip install reportlab
   ```

3. **Verify installation**:
   ```bash
   python -c "import reportlab; print('ReportLab available')"
   ```

#### ❌ **PDF Generation Works but Images Missing**

**Problem**: Image files not accessible or corrupted.

**Solutions**:

1. **Check upload folder permissions**:

   ```bash
   ls -la static/uploads/  # Linux/Mac
   dir static\uploads\     # Windows
   ```

2. **Verify image exists**:

   - Check browser developer tools for 404 errors
   - Ensure images are properly saved during upload

3. **Check image format**:
   - Ensure images are JPG, PNG, or supported formats
   - ReportLab may have issues with some image formats

#### ❌ **Client-side PDF Generation Fails**

**Problem**: jsPDF library not loading.

**Solutions**:

1. **Check internet connection** (jsPDF loads from CDN)

2. **Check browser console** for JavaScript errors

#### ❌ **Folder Selection Not Working**

**Problem**: File System Access API not supported.

**Solutions**:

1. **Update browser**:

   - Chrome 86+ or Edge 86+ required for folder selection
   - Firefox and Safari will use default download folder

2. **Enable experimental features** (Chrome):

   - Go to `chrome://flags`
   - Enable "Experimental Web Platform features"

3. **Accept automatic download** to default folder

The system should work with any clear image of a potato plant leaf!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PlantVillage Dataset**: For providing the potato disease dataset
- **TensorFlow Team**: For the amazing deep learning framework
- **Open Source Community**: For inspiration and resources

## 📞 Contact

- **Author**: Lucky Sharma
- **Email**: panditluckysharma42646@gmail.com
- **LinkedIn**: https://www.linkedin.com/in/lucky-sharma918894599977
- **GitHub**: https://github.com/itsluckysharma01

---

<div align="center">
  <p>⭐ Star this repository if you found it helpful!</p>
  <p>🍀 Happy coding and may your potatoes be healthy!</p>
</div>
"
