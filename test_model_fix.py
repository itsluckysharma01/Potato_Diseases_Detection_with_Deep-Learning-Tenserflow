#!/usr/bin/env python3
"""
Test script to verify the model prediction fixes
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image

# Configuration
MODEL_PATH = "models/1.h5"
IMAGE_SIZE = 256

# Updated class names that match the training order
CLASS_NAMES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
CLASS_DISPLAY_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def load_model():
    """Load the trained model"""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
        return model, True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, False

def preprocess_image(image):
    """Preprocess image for prediction - fixed version"""
    try:
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize image
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        
        # Convert to numpy array (NO NORMALIZATION - model has built-in rescaling)
        img_array = np.array(image)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        print(f"Image preprocessed - Shape: {img_array.shape}, Range: [{img_array.min()}, {img_array.max()}]")
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def test_prediction(model, image_path):
    """Test prediction on a sample image"""
    try:
        # Load and preprocess image
        image = Image.open(image_path)
        processed_image = preprocess_image(image)
        
        if processed_image is None:
            return None
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Get results
        predicted_class = CLASS_NAMES[predicted_class_index]
        predicted_display_name = CLASS_DISPLAY_NAMES[predicted_class_index]
        
        print(f"\n🔍 Prediction Results for {image_path}:")
        print(f"Raw predictions: {predictions[0]}")
        print(f"Predicted class: {predicted_display_name} (index: {predicted_class_index})")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        
        print(f"\n📊 All probabilities:")
        for i, (class_name, display_name) in enumerate(zip(CLASS_NAMES, CLASS_DISPLAY_NAMES)):
            prob = predictions[0][i] * 100
            print(f"  {display_name}: {prob:.2f}%")
        
        return {
            "predicted_class": predicted_display_name,
            "confidence": confidence * 100,
            "all_predictions": predictions[0]
        }
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

def main():
    """Main test function"""
    print("🧪 Testing Model Prediction Fixes\n")
    
    # Load model
    model, model_loaded = load_model()
    if not model_loaded:
        print("❌ Cannot proceed without model")
        return
    
    # Test with sample images from uploads folder
    upload_folder = "static/uploads"
    if os.path.exists(upload_folder):
        image_files = [f for f in os.listdir(upload_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if image_files:
            print(f"📁 Found {len(image_files)} test images in uploads folder\n")
            
            # Test first few images
            for i, image_file in enumerate(image_files[:3]):  # Test first 3 images
                image_path = os.path.join(upload_folder, image_file)
                print(f"Testing image {i+1}: {image_file}")
                result = test_prediction(model, image_path)
                print("-" * 50)
        else:
            print("📁 No test images found in uploads folder")
    else:
        print("📁 Uploads folder not found")
    
    # Test with dummy data
    print("\n🎲 Testing with random dummy data:")
    dummy_image = np.random.randint(0, 255, (1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    predictions = model.predict(dummy_image)
    
    print(f"Dummy predictions: {predictions[0]}")
    print(f"Sum of predictions: {np.sum(predictions[0]):.4f}")
    print(f"Predicted class: {CLASS_DISPLAY_NAMES[np.argmax(predictions[0])]}")

if __name__ == "__main__":
    main()
