"""
Segmentation Utilities for Flask App
Supports: TFLite, ONNX, and PyTorch models
"""

import tensorflow as tf
import numpy as np
import cv2
import os
import time
import matplotlib.pyplot as plt

# ============ TRY IMPORT ONNX ============
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️ ONNX Runtime not installed. Install: pip install onnxruntime")

# ============ SEGMENTATION MODEL CLASS ============
class SegmentationModel:
    """Semantic segmentation model wrapper for inference"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model_type = self._detect_model_type(model_path)
        self.interpreter = None
        self.onnx_session = None
        self.input_details = None
        self.output_details = None
        self.n_classes = 23
        self.input_height = 256
        self.input_width = 256
        self.is_channel_first = False  # Track ONNX format
        self.load_model()
        
    def _detect_model_type(self, model_path):
        """Detect model type from file extension"""
        if model_path.endswith('.tflite'):
            return 'tflite'
        elif model_path.endswith('.onnx'):
            return 'onnx'
        elif model_path.endswith('.pth') or model_path.endswith('.pt'):
            return 'pytorch'
        elif model_path.endswith('.h5') or model_path.endswith('.keras'):
            return 'keras'
        else:
            return 'unknown'
    
    def load_model(self):
        """Load model based on type"""
        if self.model_type == 'tflite':
            return self._load_tflite()
        elif self.model_type == 'onnx':
            return self._load_onnx()
        elif self.model_type == 'keras':
            return self._load_keras()
        elif self.model_type == 'pytorch':
            return self._load_pytorch()
        else:
            print(f"❌ Unknown model type: {self.model_path}")
            return False
    
    def _load_tflite(self):
        """Load TFLite model"""
        try:
            if not os.path.exists(self.model_path):
                print(f"❌ Model file not found: {self.model_path}")
                return False
            
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Auto-detect input size
            input_shape = self.input_details[0]['shape']
            if len(input_shape) == 4:
                self.input_height = input_shape[1]
                self.input_width = input_shape[2]
            
            print(f"✅ TFLite model loaded: {os.path.basename(self.model_path)}")
            print(f"   Input shape: {self.input_details[0]['shape']}")
            print(f"   Output shape: {self.output_details[0]['shape']}")
            
            return True
        except Exception as e:
            print(f"❌ Error loading TFLite model: {e}")
            return False
    
    def _load_onnx(self):
        """Load ONNX model"""
        try:
            if not ONNX_AVAILABLE:
                print("❌ ONNX Runtime not installed")
                return False
            
            if not os.path.exists(self.model_path):
                print(f"❌ Model file not found: {self.model_path}")
                return False
            
            # Create ONNX Runtime session
            self.onnx_session = ort.InferenceSession(
                self.model_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            
            # Get input/output details
            input_info = self.onnx_session.get_inputs()[0]
            output_info = self.onnx_session.get_outputs()[0]
            
            input_shape = input_info.shape
            input_name = input_info.name
            output_name = output_info.name
            
            # Detect channel ordering
            # If shape is (batch, channels, height, width) -> channel first
            # If shape is (batch, height, width, channels) -> channel last
            if len(input_shape) == 4:
                # Check if second dimension is 3 (channels) -> channel first
                if input_shape[1] == 3 or input_shape[1] == 1:
                    self.is_channel_first = True
                    self.input_height = input_shape[2] if input_shape[2] else 256
                    self.input_width = input_shape[3] if input_shape[3] else 256
                else:
                    self.is_channel_first = False
                    self.input_height = input_shape[1] if input_shape[1] else 256
                    self.input_width = input_shape[2] if input_shape[2] else 256
            
            print(f"✅ ONNX model loaded: {os.path.basename(self.model_path)}")
            print(f"   Input name: {input_name}")
            print(f"   Input shape: {input_shape}")
            print(f"   Output name: {output_name}")
            print(f"   Channel format: {'Channel First (NCHW)' if self.is_channel_first else 'Channel Last (NHWC)'}")
            
            return True
        except Exception as e:
            print(f"❌ Error loading ONNX model: {e}")
            return False
    
    def _load_keras(self):
        """Load Keras model"""
        try:
            if not os.path.exists(self.model_path):
                print(f"❌ Model file not found: {self.model_path}")
                return False
            
            self.keras_model = tf.keras.models.load_model(self.model_path)
            
            # Auto-detect input size
            input_shape = self.keras_model.input_shape
            if len(input_shape) == 4:
                self.input_height = input_shape[1]
                self.input_width = input_shape[2]
            
            print(f"✅ Keras model loaded: {os.path.basename(self.model_path)}")
            print(f"   Input shape: {input_shape}")
            
            return True
        except Exception as e:
            print(f"❌ Error loading Keras model: {e}")
            return False
    
    def _load_pytorch(self):
        """Load PyTorch model (requires custom model class)"""
        print("⚠️ PyTorch models require custom loading. Please use TFLite or ONNX.")
        return False
    
    def preprocess_image(self, image):
        """Preprocess image for inference"""
        try:
            # If image is path, load it
            if isinstance(image, str):
                img = cv2.imread(image)
                if img is None:
                    raise ValueError(f"Could not read image: {image}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = image
            
            # Store original size
            original_size = img.shape[:2]
            
            # Resize to model input size
            img_resized = cv2.resize(img, (self.input_width, self.input_height))
            
            # Normalize
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            # Add batch dimension
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            # For ONNX channel-first models, transpose from NHWC to NCHW
            if self.model_type == 'onnx' and self.is_channel_first:
                # NHWC (batch, height, width, channels) -> NCHW (batch, channels, height, width)
                img_batch = np.transpose(img_batch, (0, 3, 1, 2))
            
            return img_batch, original_size, img
            
        except Exception as e:
            print(f"❌ Preprocessing error: {e}")
            return None, None, None
    
    def predict(self, image):
        """Run inference on image"""
        try:
            # Preprocess
            img_batch, original_size, img_rgb = self.preprocess_image(image)
            if img_batch is None:
                return None, None, None
            
            # Run inference based on model type
            start_time = time.perf_counter()
            
            if self.model_type == 'tflite':
                pred = self._predict_tflite(img_batch)
            elif self.model_type == 'onnx':
                pred = self._predict_onnx(img_batch)
            elif self.model_type == 'keras':
                pred = self._predict_keras(img_batch)
            else:
                return None, None, None
            
            inference_time = (time.perf_counter() - start_time) * 1000  # ms
            
            if pred is None:
                return None, None, None
            
            # Resize back to original size
            pred_resized = cv2.resize(
                pred.astype(np.uint8),
                (original_size[1], original_size[0]),
                interpolation=cv2.INTER_NEAREST
            )
            
            return pred_resized, img_rgb, inference_time
            
        except Exception as e:
            print(f"❌ Inference error: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def _predict_tflite(self, img_batch):
        """TFLite inference"""
        self.interpreter.set_tensor(self.input_details[0]['index'], img_batch)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        return np.argmax(output, axis=-1)[0]
    
    def _predict_onnx(self, img_batch):
        """ONNX inference"""
        input_name = self.onnx_session.get_inputs()[0].name
        output_name = self.onnx_session.get_outputs()[0].name
        
        # Convert to numpy if needed
        if isinstance(img_batch, np.ndarray):
            input_data = img_batch
        else:
            input_data = img_batch.numpy() if hasattr(img_batch, 'numpy') else np.array(img_batch)
        
        # Run inference
        outputs = self.onnx_session.run([output_name], {input_name: input_data})
        
        # Handle output shape
        output = outputs[0]
        
        # If output is channel-first (NCHW), transpose to NHWC
        if self.is_channel_first and len(output.shape) == 4:
            # NCHW -> NHWC
            output = np.transpose(output, (0, 2, 3, 1))
        
        return np.argmax(output, axis=-1)[0]
    
    def _predict_keras(self, img_batch):
        """Keras inference"""
        output = self.keras_model.predict(img_batch, verbose=0)
        return np.argmax(output, axis=-1)[0]
    
    def colorize_prediction(self, pred):
        """Colorize segmentation prediction"""
        colors = [
            [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
            [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
            [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
            [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
            [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
            [0, 64, 128], [128, 64, 128], [0, 192, 128]
        ]
        colors = np.array(colors, dtype=np.uint8)
        
        color_image = np.zeros((*pred.shape, 3), dtype=np.uint8)
        for c in range(self.n_classes):
            color_image[pred == c] = colors[c % len(colors)]
        
        return color_image
    
    def create_overlay(self, original_img, pred, alpha=0.6):
        """Create overlay of prediction on original image"""
        colorized = self.colorize_prediction(pred)
        overlay = cv2.addWeighted(original_img, alpha, colorized, 1 - alpha, 0)
        return overlay

# ============ GLOBAL MODEL INSTANCE ============
model = None

def get_model():
    """Get or create model instance"""
    global model
    if model is None:
        # Try TFLite first (most compatible)
        model_paths = [
            'models/mobilenet_unet.tflite',      # TFLite - Recommended
            'models/mobilenet_unet.onnx',        # ONNX
            'models/best_vgg_unet.tflite',
            'models/vgg_unet_final.tflite',
            'models/best_vgg_unet.h5',
            'models/vgg_unet_final.h5',
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                print(f"🔍 Testing: {path}")
                temp_model = SegmentationModel(path)
                if temp_model.model_type != 'unknown':
                    model = temp_model
                    print(f"✅ Using model: {path}")
                    return model
        
        print("❌ No model found!")
        print("Available models in 'models' folder:")
        if os.path.exists('models'):
            for f in os.listdir('models'):
                print(f"   - {f}")
        return None
    
    return model

# ============ CLASS NAMES ============
CLASS_NAMES = [
    'tree', 'grass', 'other_vegetation', 'dirt', 'gravel', 'rocks',
    'water', 'paved_area', 'pool', 'person', 'dog', 'car',
    'bicycle', 'roof', 'wall', 'fence', 'fence_pole',
    'window', 'door', 'obstacle', 'background'
]