import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_from_directory, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64
from datetime import datetime
import tempfile

# PDF generation dependencies with error handling
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
    print("✅ ReportLab library loaded successfully!")
except ImportError as e:
    REPORTLAB_AVAILABLE = False
    print(f"⚠️ ReportLab not available: {e}")
    print("📝 PDF generation will use client-side fallback only")

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)
os.makedirs('templates', exist_ok=True)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Load the trained model
MODEL_PATH = "models/1.h5"  # Update this path if needed
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
    MODEL_LOADED = True
except Exception as e:
    print(f"❌ Error loading model: {e}")
    MODEL_LOADED = False
    model = None

# Class names from your training (must match the exact order from training)
# Training order: ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
CLASS_NAMES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
CLASS_DISPLAY_NAMES = ["Early Blight", "Late Blight", "Healthy"]
CLASS_DESCRIPTIONS = {
    "Potato___Early_blight": "A common fungal disease that causes dark spots on potato leaves. Treatment with copper-based fungicides is recommended.",
    "Potato___Late_blight": "A serious disease caused by Phytophthora infestans. Immediate action required - remove infected plants and apply appropriate fungicides.",
    "Potato___healthy": "The potato plant appears healthy with no signs of disease detected. Continue good agricultural practices."
}

# Image preprocessing parameters
IMAGE_SIZE = 256

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    """Preprocess the uploaded image for prediction"""
    try:
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize image
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        
        # Convert to numpy array 
        img_array = np.array(image)
        
        # DO NOT normalize here - the model has built-in rescaling layer
        # The model expects pixel values in range [0, 255]
        # img_array = img_array / 255.0  # Removed this line
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Debug: Print image statistics
        print(f"Image shape: {img_array.shape}")
        print(f"Image pixel range: [{img_array.min():.2f}, {img_array.max():.2f}]")
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_disease(image):
    """Make prediction on the preprocessed image"""
    if not MODEL_LOADED or model is None:
        return {"error": "Model not loaded"}
    
    try:
        # Preprocess image
        processed_image = preprocess_image(image)
        if processed_image is None:
            return {"error": "Failed to preprocess image"}
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Debug: Print prediction details
        print(f"Raw predictions: {predictions[0]}")
        print(f"Predicted class index: {predicted_class_index}")
        print(f"Confidence: {confidence:.4f}")
        
        # Get class name
        predicted_class = CLASS_NAMES[predicted_class_index]
        predicted_display_name = CLASS_DISPLAY_NAMES[predicted_class_index]
        
        # Create detailed results
        all_predictions = {}
        for i, (class_name, display_name) in enumerate(zip(CLASS_NAMES, CLASS_DISPLAY_NAMES)):
            all_predictions[display_name] = {
                'probability': round(float(predictions[0][i]) * 100, 2),
                'description': CLASS_DESCRIPTIONS[class_name]
            }
        
        return {
            "predicted_class": predicted_display_name,
            "confidence": round(confidence * 100, 2),
            "description": CLASS_DESCRIPTIONS[predicted_class],
            "all_predictions": all_predictions,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return {"error": f"Prediction failed: {str(e)}"}

def get_recommendations(disease_name, confidence):
    """Get treatment recommendations based on prediction"""
    recommendations = {
        'Early Blight': [
            "Remove affected leaves immediately and dispose properly",
            "Apply copper-based fungicide spray",
            "Improve air circulation around plants",
            "Avoid overhead watering",
            "Consider crop rotation for next season"
        ],
        'Late Blight': [
            "URGENT: Remove and destroy infected plants immediately",
            "Apply systemic fungicides (metalaxyl-based)",
            "Monitor weather conditions closely",
            "Increase plant spacing for better air circulation",
            "Harvest healthy tubers as soon as possible"
        ],
        'Healthy': [
            "Continue current care practices",
            "Maintain proper watering schedule",
            "Monitor plants regularly for early signs of disease",
            "Ensure good soil drainage",
            "Apply balanced fertilizer as needed"
        ]
    }
    return recommendations.get(disease_name, ["Consult agricultural expert for specific advice"])

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', model_loaded=MODEL_LOADED)

@app.route('/test')
def test_upload():
    """Simple upload test page"""
    return render_template('test_upload.html')

@app.route('/debug')
def debug_upload():
    """Debug upload test page"""
    return render_template('debug_upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        print(f"Received prediction request. Files: {list(request.files.keys())}")
        
        if 'file' not in request.files:
            print("No file in request")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        print(f"File received: {file.filename}, size: {file.content_length if hasattr(file, 'content_length') else 'unknown'}")
        
        if file.filename == '':
            print("Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            print(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG files.'}), 400
        
        # Ensure upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        print(f"Saving file to: {filepath}")
        file.save(filepath)
        print(f"File saved successfully")
        
        # Verify file exists
        if not os.path.exists(filepath):
            print(f"File was not saved properly: {filepath}")
            return jsonify({'error': 'Failed to save uploaded file'}), 500
        
        print(f"File size on disk: {os.path.getsize(filepath)} bytes")
        
        # Open and predict
        try:
            image = Image.open(filepath)
            print(f"Image opened successfully: {image.size}, mode: {image.mode}")
        except Exception as e:
            print(f"Failed to open image: {e}")
            return jsonify({'error': f'Invalid image file: {str(e)}'}), 400
        
        result = predict_disease(image)
        print(f"Prediction result: {result}")
        
        if 'error' in result:
            return jsonify(result), 500
        
        # Add recommendations and image URL for upload method
        result['recommendations'] = get_recommendations(result['predicted_class'], result['confidence'])
        result['image_url'] = url_for('uploaded_file', filename=filename)
        
        print(f"Final result with image URL: {result['image_url']}")
        return jsonify(result)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/predict_camera', methods=['POST'])
def predict_camera():
    """Handle camera image prediction"""
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]  # Remove data:image/png;base64, prefix
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Save camera image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"camera_{timestamp}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(filepath)
        
        # Make prediction
        result = predict_disease(image)
        
        if 'error' in result:
            return jsonify(result), 500
        
        # Add recommendations and image URL for camera method
        result['recommendations'] = get_recommendations(result['predicted_class'], result['confidence'])
        result['image_url'] = url_for('uploaded_file', filename=filename)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Camera prediction failed: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    upload_dir_exists = os.path.exists(app.config['UPLOAD_FOLDER'])
    upload_dir_writable = os.access(app.config['UPLOAD_FOLDER'], os.W_OK) if upload_dir_exists else False
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL_LOADED,
        'upload_dir_exists': upload_dir_exists,
        'upload_dir_writable': upload_dir_writable,
        'upload_path': app.config['UPLOAD_FOLDER'],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/debug/upload-test')
def debug_upload_test():
    """Debug endpoint to test upload directory"""
    try:
        # Ensure upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Test file creation
        test_file = os.path.join(app.config['UPLOAD_FOLDER'], 'test.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        
        # Clean up test file
        os.remove(test_file)
        
        return jsonify({
            'status': 'success',
            'message': 'Upload directory is working correctly',
            'path': app.config['UPLOAD_FOLDER']
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Upload directory test failed: {str(e)}',
            'path': app.config['UPLOAD_FOLDER']
        }), 500

def test_model_predictions():
    """Test the model with some dummy data to verify it's working correctly"""
    if not MODEL_LOADED or model is None:
        return {"error": "Model not loaded"}
    
    try:
        # Create dummy test data - same shape as expected input
        dummy_image = np.random.randint(0, 255, (1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        
        # Make prediction
        predictions = model.predict(dummy_image)
        
        print(f"Model test - Input shape: {dummy_image.shape}")
        print(f"Model test - Output shape: {predictions.shape}")
        print(f"Model test - Predictions: {predictions[0]}")
        print(f"Model test - Sum of predictions: {np.sum(predictions[0])}")
        print(f"Model test - Class names order: {CLASS_NAMES}")
        
        return {
            "status": "success",
            "input_shape": str(dummy_image.shape),
            "output_shape": str(predictions.shape),
            "predictions": predictions[0].tolist(),
            "prediction_sum": float(np.sum(predictions[0])),
            "class_names": CLASS_NAMES
        }
    except Exception as e:
        print(f"Model test error: {e}")
        return {"error": f"Model test failed: {str(e)}"}

@app.route('/debug/model-test')
def debug_model_test():
    """Debug endpoint to test model functionality"""
    result = test_model_predictions()
    return jsonify(result)

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return render_template('index.html', model_loaded=MODEL_LOADED)

def generate_pdf_report(prediction_data, image_path=None):
    """Generate a professional PDF report for the disease prediction"""
    if not REPORTLAB_AVAILABLE:
        print("❌ ReportLab not available - cannot generate server-side PDF")
        return None
        
    try:
        # Create a temporary file for the PDF
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        
        # Create the PDF document
        doc = SimpleDocTemplate(temp_pdf.name, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkgreen
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.darkblue
        )
        
        # Title
        story.append(Paragraph("🥔 POTATO DISEASE DETECTION REPORT", title_style))
        story.append(Spacer(1, 20))
        
        # Header info table
        header_data = [
            ['Report Generated:', prediction_data.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))],
            ['Analysis Method:', 'Deep Learning AI Classification'],
            ['Model Version:', 'TensorFlow/Keras CNN v1.0']
        ]
        
        header_table = Table(header_data, colWidths=[2*inch, 4*inch])
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(header_table)
        story.append(Spacer(1, 30))
        
        # Add image if provided
        if image_path and os.path.exists(image_path):
            story.append(Paragraph("📸 ANALYZED IMAGE", heading_style))
            try:
                # Resize image to fit in PDF
                img = RLImage(image_path)
                img.drawHeight = 3 * inch
                img.drawWidth = 3 * inch
                story.append(img)
                story.append(Spacer(1, 20))
            except Exception as img_error:
                print(f"Warning: Could not add image to PDF: {img_error}")
                story.append(Paragraph("Image could not be embedded in PDF", styles['Normal']))
                story.append(Spacer(1, 20))
        
        # Diagnosis Section
        story.append(Paragraph("🎯 DIAGNOSIS RESULTS", heading_style))
        
        # Main diagnosis
        diagnosis_data = [
            ['Predicted Disease:', prediction_data.get('predicted_class', 'Unknown')],
            ['Confidence Level:', f"{prediction_data.get('confidence', 0):.2f}%"],
            ['Risk Assessment:', get_risk_level(prediction_data.get('confidence', 0))]
        ]
        
        diagnosis_table = Table(diagnosis_data, colWidths=[2*inch, 4*inch])
        diagnosis_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(diagnosis_table)
        story.append(Spacer(1, 20))
        
        # Description
        story.append(Paragraph("📋 DESCRIPTION", heading_style))
        description = Paragraph(prediction_data.get('description', 'No description available.'), styles['Normal'])
        story.append(description)
        story.append(Spacer(1, 20))
        
        # Probability breakdown
        story.append(Paragraph("📊 PROBABILITY BREAKDOWN", heading_style))
        
        prob_data = [['Disease Type', 'Probability']]
        all_predictions = prediction_data.get('all_predictions', {})
        for disease, info in all_predictions.items():
            prob_data.append([disease, f"{info.get('probability', 0):.2f}%"])
        
        prob_table = Table(prob_data, colWidths=[3*inch, 2*inch])
        prob_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(prob_table)
        story.append(Spacer(1, 20))
        
        # Recommendations
        story.append(Paragraph("💡 TREATMENT RECOMMENDATIONS", heading_style))
        
        recommendations = prediction_data.get('recommendations', [])
        for i, rec in enumerate(recommendations, 1):
            rec_text = f"{i}. {rec}"
            story.append(Paragraph(rec_text, styles['Normal']))
            story.append(Spacer(1, 8))
        
        story.append(Spacer(1, 30))
        
        # Footer
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_CENTER,
            textColor=colors.grey
        )
        
        story.append(Paragraph("_______________________________________________", footer_style))
        story.append(Spacer(1, 10))
        story.append(Paragraph("Generated by Potato Disease Detection System", footer_style))
        story.append(Paragraph("Powered by Flask & TensorFlow | Lucky Sharma", footer_style))
        story.append(Paragraph("© 2025 All Rights Reserved", footer_style))
        
        # Build PDF
        doc.build(story)
        
        print(f"✅ PDF report generated successfully: {temp_pdf.name}")
        return temp_pdf.name
        
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_risk_level(confidence):
    """Determine risk level based on confidence"""
    if confidence >= 80:
        return "High Confidence"
    elif confidence >= 60:
        return "Medium Confidence"
    else:
        return "Low Confidence - Manual Verification Recommended"

@app.route('/generate-pdf-report', methods=['POST'])
def generate_pdf_report_route():
    """Generate and download PDF report"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Check if ReportLab is available
        if not REPORTLAB_AVAILABLE:
            print("⚠️ ReportLab not available, suggesting client-side fallback")
            return jsonify({
                'error': 'Server-side PDF generation not available',
                'fallback': 'client',
                'message': 'ReportLab library not installed. Using client-side fallback.'
            }), 503
        
        # Get image path if provided
        image_path = None
        if 'image_url' in data:
            # Extract filename from URL and construct full path
            image_filename = data['image_url'].split('/')[-1]
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
            
            # Verify image exists
            if not os.path.exists(image_path):
                image_path = None
        
        # Generate PDF
        pdf_path = generate_pdf_report(data, image_path)
        
        if not pdf_path:
            print("❌ PDF generation failed, suggesting client-side fallback")
            return jsonify({
                'error': 'Server-side PDF generation failed',
                'fallback': 'client',
                'message': 'Could not generate PDF on server. Using client-side fallback.'
            }), 503
        
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        disease_name = data.get('predicted_class', 'unknown').replace(' ', '_')
        pdf_filename = f"potato_disease_report_{disease_name}_{timestamp}.pdf"
        
        print(f"✅ PDF generated successfully: {pdf_filename}")
        return send_file(
            pdf_path,
            as_attachment=True,
            download_name=pdf_filename,
            mimetype='application/pdf'
        )
        
    except Exception as e:
        print(f"❌ PDF generation error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'PDF generation failed: {str(e)}',
            'fallback': 'client',
            'message': 'Server error occurred. Using client-side fallback.'
        }), 503

if __name__ == '__main__':
    print("🚀 Starting Potato Disease Detection Flask App...")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"🤖 Model loaded: {MODEL_LOADED}")
    print("🌐 Access the app at: http://localhost:5000")
    print("💡 Press Ctrl+C to stop the server")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
