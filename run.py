# from flask import Flask, render_template, request, jsonify, Response, send_from_directory
# from flask_cors import CORS
# import os
# import cv2
# import numpy as np
# from werkzeug.utils import secure_filename
# import base64
# import time
# import threading
# import shutil

# # ============ FIX PROTOBUF ISSUE ============
# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# # ============ GPU OPTIMIZATION ============
# import tensorflow as tf

# # Enable GPU memory growth
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         print(f"✅ GPU detected: {len(gpus)} GPU(s)")
#         if gpus:
#             print(f"   GPU: {gpus[0].name}")
#     except RuntimeError as e:
#         print(f"⚠️ GPU config error: {e}")
# else:
#     print("⚠️ No GPU found - using CPU")

# # ============ IMPORT SEGMENTATION ============
# from utils.segmentation import get_model, CLASS_NAMES

# # Initialize Flask app
# app = Flask(__name__)

# # Enable CORS
# CORS(app, resources={
#     r"/*": {
#         "origins": "*",
#         "methods": ["GET", "POST", "OPTIONS"],
#         "allow_headers": ["Content-Type", "Authorization", "Accept"]
#     }
# })

# # Configuration
# app.config['UPLOAD_FOLDER'] = 'static/uploads'
# app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max
# app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

# # Create upload folder
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# # ============ HELPER FUNCTIONS ============

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# def image_to_base64(image):
#     """Convert numpy image to base64"""
#     if image.dtype != np.uint8:
#         image = image.astype(np.uint8)
#     _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
#     return base64.b64encode(buffer).decode('utf-8')

# def get_class_distribution(pred):
#     unique, counts = np.unique(pred, return_counts=True)
#     print(f"{unique}------------{counts}")
#     total = pred.size
#     distribution = {}
#     for cls, count in zip(unique, counts):
#         percentage = (count / total) * 100
#         if cls < len(CLASS_NAMES):
#             distribution[CLASS_NAMES[cls]] = {
#                 'pixels': int(count),
#                 'percentage': round(percentage, 2)
#             }
#             print("---------------------------2-----------------------------------")
#             print(f"{distribution[CLASS_NAMES[cls]]}")
#         else:
#             distribution[f'class_{cls}'] = {
#                 'pixels': int(count),
#                 'percentage': round(percentage, 2)
#             }
#     print("---------------------------------3-----------------------------")
#     print(f"{distribution}")
#     return distribution

# # ============ ROUTES ============

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/camera')
# def camera():
#     return render_template('camera.html')

# @app.route('/upload', methods=['POST', 'OPTIONS'])
# def upload_image():
#     """Handle image upload and segmentation"""
#     if request.method == 'OPTIONS':
#         response = jsonify({'status': 'ok'})
#         response.headers.add('Access-Control-Allow-Origin', '*')
#         response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
#         response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Accept')
#         return response
    
#     try:
#         print("📸 Received upload request")
        
#         # Check if file is present
#         if 'image' not in request.files:
#             print("❌ No file in request")
#             return jsonify({'error': 'No file uploaded'}), 400
        
#         file = request.files['image']
#         print(f"📁 File: {file.filename}")
        
#         if file.filename == '':
#             print("❌ Empty filename")
#             return jsonify({'error': 'No file selected'}), 400
        
#         if not allowed_file(file.filename):
#             print(f"❌ File type not allowed: {file.filename}")
#             return jsonify({'error': 'File type not allowed. Use: png, jpg, jpeg, bmp, tiff'}), 400
        
#         # Save the uploaded file
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
#         print(f"✅ File saved: {filepath}")
        
#         # Read image from saved file
#         img = cv2.imread(filepath)
#         if img is None:
#             print("❌ Failed to read image")
#             return jsonify({'error': 'Could not read image'}), 400
        
#         print(f"✅ Image loaded: {img.shape}")
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
#         # Get model
#         model = get_model()
#         if model is None:
#             print("❌ Model not loaded")
#             return jsonify({'error': 'Model not loaded'}), 500
        
#         print("🔮 Running prediction...")
#         # Run prediction
#         pred, original_img, inference_time = model.predict(img_rgb)
#         print(f"++++++++++++++++++++++++++++++++++++++++++++++++{pred}++++++++++++++++++++++++++++++")

#         if pred is None:
#             print("❌ Prediction failed")
#             return jsonify({'error': 'Prediction failed'}), 500
        
#         print(f"✅ Prediction complete in {inference_time:.2f}ms")
        
#         # Generate outputs
#         colorized = model.colorize_prediction(pred)
#         overlay = model.create_overlay(img_rgb, pred)
        
        



#         # here we want - to which class that pericular pixel belongs





#         # Save output images
#         colorized_path = os.path.join(app.config['UPLOAD_FOLDER'], f'colorized_{filename}')
#         overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], f'overlay_{filename}')
#         cv2.imwrite(colorized_path, cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR))
#         cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
#         # Convert to base64
#         with open(filepath, 'rb') as f:
#             original_b64 = base64.b64encode(f.read()).decode('utf-8')
#         with open(colorized_path, 'rb') as f:
#             colorized_b64 = base64.b64encode(f.read()).decode('utf-8')
#         with open(overlay_path, 'rb') as f:
#             overlay_b64 = base64.b64encode(f.read()).decode('utf-8')
        
#         distribution = get_class_distribution(pred)
        
#         print("✅ Results ready")
#         return jsonify({
#             'success': True,
#             'original': original_b64,
#             'colorized': colorized_b64,
#             'overlay': overlay_b64,
#             'inference_time': round(inference_time, 2),
#             'distribution': distribution,
#             'filename': filename
#         })
        
#     except Exception as e:
#         print(f"❌ Error: {e}")
#         import traceback
#         traceback.print_exc()
#         return jsonify({'error': str(e)}), 500

# @app.route('/predict_frame', methods=['POST'])
# def predict_frame():
#     """Predict on single frame from camera"""
#     try:
#         data = request.get_json()
#         if not data or 'image' not in data:
#             return jsonify({'error': 'No image data'}), 400
        
#         # Decode base64 image
#         image_data = base64.b64decode(data['image'].split(',')[1])
#         np_arr = np.frombuffer(image_data, np.uint8)
#         img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
#         if img is None:
#             return jsonify({'error': 'Invalid image'}), 400
        
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
#         model = get_model()
#         if model is None:
#             return jsonify({'error': 'Model not loaded'}), 500
        
#         # Predict - unpack 3 values
#         pred, _, inference_time = model.predict(img_rgb)
        
#         if pred is None:
#             return jsonify({'error': 'Prediction failed'}), 500
        
#         # Generate overlay
#         overlay = model.create_overlay(img_rgb, pred)
#         overlay_b64 = image_to_base64(overlay)
        
#         distribution = get_class_distribution(pred)
        
#         return jsonify({
#             'success': True,
#             'overlay': overlay_b64,
#             'inference_time': round(inference_time, 2),
#             'classes_detected': len(distribution),
#             'distribution': distribution
#         })
        
#     except Exception as e:
#         print(f"❌ Error: {e}")
#         return jsonify({'error': str(e)}), 500

# @app.route('/health', methods=['GET'])
# def health_check():
#     model = get_model()
#     gpu_info = {
#         'available': len(gpus) > 0,
#         'name': gpus[0].name if gpus else 'None'
#     }
#     return jsonify({
#         'status': 'healthy',
#         'model_loaded': model is not None,
#         'model_path': model.model_path if model else None,
#         'gpu': gpu_info
#     })

# @app.route('/benchmark', methods=['GET'])
# def benchmark():
#     model = get_model()
#     if model is None:
#         return jsonify({'error': 'Model not loaded'}), 500
    
#     test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
#     # Warm-up
#     for _ in range(10):
#         pred, _, _ = model.predict(test_img)
#         if pred is None:
#             return jsonify({'error': 'Model prediction failed'}), 500
    
#     times = []
#     for _ in range(50):
#         start = time.perf_counter()
#         pred, _, _ = model.predict(test_img)
#         end = time.perf_counter()
#         if pred is None:
#             continue
#         times.append((end - start) * 1000)
    
#     if not times:
#         return jsonify({'error': 'Benchmark failed'}), 500
    
#     avg_time = np.mean(times)
#     fps = 1000 / avg_time if avg_time > 0 else 0
    
#     return jsonify({
#         'avg_inference_ms': round(avg_time, 2),
#         'fps': round(fps, 1),
#         'gpu': len(gpus) > 0,
#         'gpu_name': gpus[0].name if gpus else 'None',
#         'model': model.model_path
#     })

# @app.route('/download/<filename>')
# def download_file(filename):
#     """Download segmented images"""
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

# if __name__ == '__main__':
#     print("="*60)
#     print("🚀 Starting Segmentation Flask App")
#     print("="*60)
    
#     model = get_model()
#     if model:
#         print(f"✅ Model loaded: {model.model_path}")
#     else:
#         print("⚠️ No model loaded!")
    
#     print("="*60)
#     print("🌐 Server running at:")
#     print("   http://127.0.0.1:5000")
#     print("   http://localhost:5000")
#     print("="*60)
    
#     app.run(
#         host='127.0.0.1',
#         port=5000,
#         debug=True,
#         threaded=True,
#         use_reloader=False
#     )


from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from flask_cors import CORS
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import base64
import time
import threading
import json

# ============ FIX PROTOBUF ISSUE ============
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# ============ GPU OPTIMIZATION ============
import tensorflow as tf

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU detected: {len(gpus)} GPU(s)")
        if gpus:
            print(f"   GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(f"⚠️ GPU config error: {e}")
else:
    print("⚠️ No GPU found - using CPU")

# ============ IMPORT SEGMENTATION ============
from utils.segmentation import get_model, CLASS_NAMES

# Initialize Flask app
app = Flask(__name__)

# Enable CORS
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"]
    }
})

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ============ CLASS COLORS ============
CLASS_COLORS = [
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128], [128, 64, 128], [0, 192, 128]
]

# ============ HELPER FUNCTIONS ============

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def image_to_base64(image):
    """Convert numpy image to base64"""
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')

def get_class_distribution(pred):
    """Get distribution of classes in prediction"""
    unique, counts = np.unique(pred, return_counts=True)
    total = pred.size
    distribution = {}
    for cls, count in zip(unique, counts):
        percentage = (count / total) * 100
        if cls < len(CLASS_NAMES):
            distribution[CLASS_NAMES[cls]] = {
                'pixels': int(count),
                'percentage': round(percentage, 2)
            }
        else:
            distribution[f'class_{cls}'] = {
                'pixels': int(count),
                'percentage': round(percentage, 2)
            }
    return distribution

def get_pixel_class_mapping(pred):
    """
    Get pixel-wise class mapping
    Returns a dictionary mapping pixel values to class information
    """
    unique_pixels = np.unique(pred)
    
    pixel_class_info = {}
    for pixel_value in unique_pixels:
        # Get class name
        if pixel_value < len(CLASS_NAMES):
            class_name = CLASS_NAMES[pixel_value]
        else:
            class_name = f'class_{pixel_value}'
        
        # Get color
        if pixel_value < len(CLASS_COLORS):
            color = CLASS_COLORS[pixel_value]
        else:
            color = [255, 255, 255]
        
        # Count pixels
        count = np.sum(pred == pixel_value)
        percentage = (count / pred.size) * 100
        
        pixel_class_info[int(pixel_value)] = {
            'class_name': class_name,
            'color': color,
            'color_hex': '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2]),
            'pixel_value': int(pixel_value),
            'count': int(count),
            'percentage': round(percentage, 2)
        }
    
    return pixel_class_info

def get_pixel_at_position(pred, x, y):
    """
    Get class information for a specific pixel at position (x, y)
    """
    if x < 0 or x >= pred.shape[1] or y < 0 or y >= pred.shape[0]:
        return None
    
    pixel_value = int(pred[y, x])
    
    if pixel_value < len(CLASS_NAMES):
        class_name = CLASS_NAMES[pixel_value]
    else:
        class_name = f'class_{pixel_value}'
    
    if pixel_value < len(CLASS_COLORS):
        color = CLASS_COLORS[pixel_value]
    else:
        color = [255, 255, 255]
    
    return {
        'x': x,
        'y': y,
        'pixel_value': pixel_value,
        'class_name': class_name,
        'color': color,
        'color_hex': '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
    }

def get_all_pixel_classes(pred):
    """
    Get class for every pixel in the image
    Returns a 2D array with class names for each pixel
    """
    height, width = pred.shape
    pixel_classes = []
    
    for y in range(height):
        row = []
        for x in range(width):
            pixel_value = int(pred[y, x])
            if pixel_value < len(CLASS_NAMES):
                class_name = CLASS_NAMES[pixel_value]
            else:
                class_name = f'class_{pixel_value}'
            row.append(class_name)
        pixel_classes.append(row)
    
    return pixel_classes

# ============ ROUTES ============

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_image():
    """Handle image upload and segmentation with pixel-wise class mapping"""
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Accept')
        return response
    
    try:
        print("📸 Received upload request")
        
        # Check if file is present
        if 'image' not in request.files:
            print("❌ No file in request")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['image']
        print(f"📁 File: {file.filename}")
        
        if file.filename == '':
            print("❌ Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            print(f"❌ File type not allowed: {file.filename}")
            return jsonify({'error': 'File type not allowed. Use: png, jpg, jpeg, bmp, tiff'}), 400
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"✅ File saved: {filepath}")
        
        # Read image from saved file
        img = cv2.imread(filepath)
        if img is None:
            print("❌ Failed to read image")
            return jsonify({'error': 'Could not read image'}), 400
        
        print(f"✅ Image loaded: {img.shape}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get model
        model = get_model()
        if model is None:
            print("❌ Model not loaded")
            return jsonify({'error': 'Model not loaded'}), 500
        
        print("🔮 Running prediction...")
        # Run prediction
        pred, original_img, inference_time = model.predict(img_rgb)
        
        if pred is None:
            print("❌ Prediction failed")
            return jsonify({'error': 'Prediction failed'}), 500
        
        print(f"✅ Prediction complete in {inference_time:.2f}ms")
        
        # Generate outputs
        colorized = model.colorize_prediction(pred)
        overlay = model.create_overlay(img_rgb, pred)
        
        # ============ PIXEL-WISE CLASS MAPPING ============
        print("\n" + "="*60)
        print("📊 PIXEL-WISE CLASS MAPPING")
        print("="*60)
        print(f"Total pixels in image: {pred.size}")
        
        # Get pixel class mapping
        pixel_class_info = get_pixel_class_mapping(pred)
        all_pixel_classes = get_all_pixel_classes(pred)
        
        print(f"Unique classes detected: {len(pixel_class_info)}")
        print("\nPixel Value → Class Name | Color | Pixel Count")
        print("-"*60)
        
        for pixel_value, info in pixel_class_info.items():
            print(f"Pixel {pixel_value:3d} → {info['class_name']:20s} | {info['color_hex']} | {info['count']:6d} pixels ({info['percentage']:5.2f}%)")
        print("="*60)
        
        # Get sample pixels from different positions
        sample_pixels = []
        height, width = pred.shape
        positions = [
            (0, 0), (width//2, height//2), (width-1, height-1),
            (width//4, height//4), (3*width//4, 3*height//4)
        ]
        for x, y in positions:
            if x < width and y < height:
                pixel_info = get_pixel_at_position(pred, x, y)
                if pixel_info:
                    sample_pixels.append(pixel_info)
        # =================================================
        
        # Save output images
        colorized_path = os.path.join(app.config['UPLOAD_FOLDER'], f'colorized_{filename}')
        overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], f'overlay_{filename}')
        cv2.imwrite(colorized_path, cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR))
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        # Save pixel class mapping as JSON
        pixel_mapping_path = os.path.join(app.config['UPLOAD_FOLDER'], f'pixel_mapping_{filename}.json')
        with open(pixel_mapping_path, 'w') as f:
            json.dump({
                'pixel_class_info': pixel_class_info,
                'image_shape': pred.shape,
                'total_pixels': int(pred.size),
                'unique_classes': len(pixel_class_info)
            }, f, indent=2)
        
        # Convert to base64
        with open(filepath, 'rb') as f:
            original_b64 = base64.b64encode(f.read()).decode('utf-8')
        with open(colorized_path, 'rb') as f:
            colorized_b64 = base64.b64encode(f.read()).decode('utf-8')
        with open(overlay_path, 'rb') as f:
            overlay_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        distribution = get_class_distribution(pred)
        
        print("✅ Results ready")
        return jsonify({
            'success': True,
            'original': original_b64,
            'colorized': colorized_b64,
            'overlay': overlay_b64,
            'inference_time': round(inference_time, 2),
            'distribution': distribution,
            'pixel_class_info': pixel_class_info,
            'all_pixel_classes': all_pixel_classes,
            'sample_pixels': sample_pixels,
            'image_shape': pred.shape,
            'filename': filename
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    """Predict on single frame from camera with pixel-wise class mapping"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data'}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'].split(',')[1])
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        model = get_model()
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Predict - unpack 3 values
        pred, _, inference_time = model.predict(img_rgb)
        
        if pred is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Generate overlay
        overlay = model.create_overlay(img_rgb, pred)
        overlay_b64 = image_to_base64(overlay)
        
        # Get pixel-wise class mapping
        distribution = get_class_distribution(pred)
        pixel_class_info = get_pixel_class_mapping(pred)
        
        # Get center pixel class
        height, width = pred.shape
        center_pixel = get_pixel_at_position(pred, width//2, height//2)
        
        return jsonify({
            'success': True,
            'overlay': overlay_b64,
            'inference_time': round(inference_time, 2),
            'classes_detected': len(distribution),
            'distribution': distribution,
            'pixel_class_info': pixel_class_info,
            'center_pixel': center_pixel,
            'image_shape': pred.shape
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/pixel_info', methods=['POST'])
def get_pixel_info():
    """
    Get class information for specific pixels
    Expects JSON with 'x' and 'y' coordinates
    """
    try:
        data = request.get_json()
        if not data or 'x' not in data or 'y' not in data:
            return jsonify({'error': 'No pixel coordinates provided'}), 400
        
        x = data['x']
        y = data['y']
        
        # For demo, return class info (in production, you'd have stored prediction)
        return jsonify({
            'error': 'Please upload an image first',
            'suggestion': 'Upload an image to get pixel class information'
        }), 400
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    model = get_model()
    gpu_info = {
        'available': len(gpus) > 0,
        'name': gpus[0].name if gpus else 'None'
    }
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_path': model.model_path if model else None,
        'gpu': gpu_info
    })

@app.route('/benchmark', methods=['GET'])
def benchmark():
    model = get_model()
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Warm-up
    for _ in range(10):
        pred, _, _ = model.predict(test_img)
        if pred is None:
            return jsonify({'error': 'Model prediction failed'}), 500
    
    times = []
    for _ in range(50):
        start = time.perf_counter()
        pred, _, _ = model.predict(test_img)
        end = time.perf_counter()
        if pred is None:
            continue
        times.append((end - start) * 1000)
    
    if not times:
        return jsonify({'error': 'Benchmark failed'}), 500
    
    avg_time = np.mean(times)
    fps = 1000 / avg_time if avg_time > 0 else 0
    
    return jsonify({
        'avg_inference_ms': round(avg_time, 2),
        'fps': round(fps, 1),
        'gpu': len(gpus) > 0,
        'gpu_name': gpus[0].name if gpus else 'None',
        'model': model.model_path
    })

@app.route('/download/<filename>')
def download_file(filename):
    """Download segmented images"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/class_info', methods=['GET'])
def get_class_info():
    """Get all class information"""
    class_info = []
    for i in range(len(CLASS_NAMES)):
        color = CLASS_COLORS[i] if i < len(CLASS_COLORS) else [255, 255, 255]
        class_info.append({
            'id': i,
            'name': CLASS_NAMES[i],
            'color': color,
            'color_hex': '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
        })
    return jsonify({
        'classes': class_info,
        'total_classes': len(CLASS_NAMES)
    })

if __name__ == '__main__':
    print("="*60)
    print("🚀 Starting Segmentation Flask App")
    print("="*60)
    
    model = get_model()
    if model:
        print(f"✅ Model loaded: {model.model_path}")
    else:
        print("⚠️ No model loaded!")
    
    print("="*60)
    print("🌐 Server running at:")
    print("   http://127.0.0.1:5000")
    print("   http://localhost:5000")
    print("="*60)
    print("📊 Class Info: http://127.0.0.1:5000/class_info")
    print("="*60)
    
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=True,
        threaded=True,
        use_reloader=False
    )