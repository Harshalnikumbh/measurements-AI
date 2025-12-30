import os
import cv2
import math
import time
import torch
import joblib
import trimesh
import logging
import hashlib
import requests
import numpy as np
from pathlib import Path
from datetime import datetime
from flask_caching import Cache
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template, send_file


# Define a central logger for the application
logger = logging.getLogger('BodyApp')
logger.setLevel(logging.INFO) # Set default logging level

# Create a console handler and set the level to INFO
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create a formatter and add it to the handler
formatter = logging.Formatter(
    '[%(asctime)s] [%(levelname)s] [%(name)s.%(funcName)s] - %(message)s'
)
ch.setFormatter(formatter)

# Add the handler to the logger (avoid duplicate handlers if re-running in interactive env)
if not logger.handlers:
    logger.addHandler(ch)

# MediaPipe for pose detection
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    print("MediaPipe not available. Pose validation will be skipped.")
    MEDIAPIPE_AVAILABLE = False

if 'PYOPENGL_PLATFORM' in os.environ:
    del os.environ['PYOPENGL_PLATFORM']

# Try to import HMR2 dependencies
HMR2_AVAILABLE = False
try:
    from hmr2.configs import CACHE_DIR_4DHUMANS
    from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT, download_models
    from hmr2.utils import recursive_to
    from hmr2.datasets.vitdet_dataset import ViTDetDataset
    from hmr2.utils.renderer import Renderer, cam_crop_to_full
    from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    import hmr2
    HMR2_AVAILABLE = True
    logger.info("HMR2 and Detectron2 successfully imported.")
except ImportError as e:
    logger.warning(f"HMR2 or Detectron2 not available: {e}. HMR2 functionalities will be disabled.")

# Force CPU loading for PyTorch models
_original_load = torch.load
def cpu_load(*args, **kwargs):
    kwargs['map_location'] = torch.device('cpu')
    return _original_load(*args, **kwargs)
torch.load = cpu_load
device = torch.device('cpu')
logger.info(f"Using device: {device}")

# --- Flask App Configuration ---
app = Flask(__name__)
ML_MODEL_AVAILABLE = True
try:
    SIZE_MODEL = joblib.load("size_model.joblib")
    SCALER = joblib.load("scaler.joblib")
    GENDER_ENCODER = joblib.load("gender_encoder.joblib")
    logger.info("ML Size models (joblib) loaded successfully.")
except Exception as e:
    logger.warning(f"ML Size models could not be loaded: {e}. ML size prediction will be disabled.")    
    ML_MODEL_AVAILABLE = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['TRYON_FOLDER'] = 'tryon_results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['TRYON_FOLDER'], exist_ok=True)
logger.info("Upload, output, and try-on result folders are set up.")

#-- Cache Configuration ---
cache_config = {
    'CACHE_TYPE': 'FileSystemCache',  # Use 'RedisCache' for production with Redis
    'CACHE_DIR': 'cache',  # Directory for cache files
    'CACHE_DEFAULT_TIMEOUT': 300,  # Default timeout in seconds
    'CACHE_THRESHOLD': 500  # Maximum number of items in cache
}
app.config.from_mapping(cache_config)
cache = Cache(app)

# Create cache directory
os.makedirs(app.config['CACHE_DIR'], exist_ok=True)
logger.info("Cache system initialized.")

# --- Cache Helper Functions ---
def generate_image_hash(image_path):
    """Generate SHA256 hash of an image file for cache key."""
    try:
        with open(image_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]  # Use first 16 chars
    except Exception as e:
        logger.warning(f"Could not generate hash for {image_path}: {e}")
        return None

def generate_cache_key(*args):
    """Generate a cache key from multiple arguments."""
    key_string = '_'.join(str(arg) for arg in args)
    return hashlib.md5(key_string.encode()).hexdigest()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Virtual Try-On API Configuration
LIGHTX_API_KEY = "ae0ddbab09454d599116b0ec308dec7c_a43a8874d35d4cc88b85d224711d1d07_andoraitools"  
LIGHTX_BASE_URL = "https://api.lightxeditor.com/external/api/v2"
CONTENT_TYPE = "image/jpeg"

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Pose Validation Class ---

class PoseValidator:
    """Validate that the person is standing straight using MediaPipe Pose."""
    
    WAIST_ANGLE_THRESHOLD = 160  # Minimum angle to be considered straight
    
    def __init__(self):
        if not MEDIAPIPE_AVAILABLE:
            logger.critical("Attempted to instantiate PoseValidator but MediaPipe is not available.")
            raise RuntimeError("MediaPipe is not available for pose validation")
        self.mp_pose = mp.solutions.pose

    # Add this method to the PoseValidator class (around line ~195)
    def validate_images_cached(self, front_image_path, side_image_path):
        """
        Validate both images with caching.
        Returns: (overall_success, detailed_results)
        """
        # Generate cache keys based on image hashes
        front_hash = generate_image_hash(front_image_path)
        side_hash = generate_image_hash(side_image_path)
        
        if front_hash is None or side_hash is None:
            logger.warning("Could not generate image hashes for pose validation, proceeding without cache")
            return self.validate_images(front_image_path, side_image_path)
    
        cache_key = f"pose_validation_{front_hash}_{side_hash}"
    
        # Check cache
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            logger.info(f"✓ Cache hit for pose validation: {cache_key}")
            return cached_result
        
        # Cache miss - perform validation
        logger.info(f"Cache miss for pose validation: {cache_key}, validating...")
        result = self.validate_images(front_image_path, side_image_path)
        
        # Cache the result (900 seconds = 15 minutes)
        cache.set(cache_key, result, timeout=900)
        logger.info(f"✓ Cached pose validation result: {cache_key}")
        
        return result
    
    @staticmethod
    def calculate_angle(a, b, c):
        """Calculate angle between three points (a-b-c) where b is the vertex."""
        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])
        
        dot = ba[0] * bc[0] + ba[1] * bc[1]
        mag_ba = math.hypot(*ba)
        mag_bc = math.hypot(*bc)
        
        if mag_ba * mag_bc == 0:
            return 180.0
        
        cos_angle = max(min(dot / (mag_ba * mag_bc), 1.0), -1.0)
        return math.degrees(math.acos(cos_angle))
    
    def load_pose_landmarks(self, image_path):
        """Load and detect pose landmarks from an image."""
        logger.debug(f"Loading image for pose detection: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Image load failed for: {image_path}")
            raise ValueError(f"Could not load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        with self.mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
            result = pose.process(image_rgb)
        
        if not result.pose_landmarks:
            logger.error(f"No person detected in image: {image_path}")
            raise ValueError("No person detected in the image")
        
        return result.pose_landmarks.landmark
    
    def check_front_view(self, image_path):
        """
        Check if person is standing straight in front view.
        Returns: (is_accepted, angle, message)
        """
        try:
            landmarks = self.load_pose_landmarks(image_path)
            
            def get_point(idx):
                return (landmarks[idx].x, landmarks[idx].y)
            
            # Get key landmarks
            LS = self.mp_pose.PoseLandmark.LEFT_SHOULDER.value
            RS = self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            LH = self.mp_pose.PoseLandmark.LEFT_HIP.value
            RH = self.mp_pose.PoseLandmark.RIGHT_HIP.value
            LK = self.mp_pose.PoseLandmark.LEFT_KNEE.value
            RK = self.mp_pose.PoseLandmark.RIGHT_KNEE.value
            
            # Calculate center points
            shoulder = (
                (get_point(LS)[0] + get_point(RS)[0]) / 2,
                (get_point(LS)[1] + get_point(RS)[1]) / 2
            )
            hip = (
                (get_point(LH)[0] + get_point(RH)[0]) / 2,
                (get_point(LH)[1] + get_point(RH)[1]) / 2
            )
            knee = (
                (get_point(LK)[0] + get_point(RK)[0]) / 2,
                (get_point(LK)[1] + get_point(RK)[1]) / 2
            )
            
            # Calculate waist angle
            angle = self.calculate_angle(shoulder, hip, knee)
            is_accepted = angle >= self.WAIST_ANGLE_THRESHOLD
            
            message = f"Front view: waist angle {angle:.1f}° - {'ACCEPTED' if is_accepted else 'REJECTED'}"
            
            return is_accepted, angle, message
            
        except ValueError as e:
            logger.error(f"Front view error: {str(e)}")
            raise ValueError(f"Front view error: {str(e)}")
        except Exception as e:
            logger.critical(f"Unexpected error in front view validation: {str(e)}")
            raise RuntimeError(f"Unexpected error in front view validation: {str(e)}")
    
    def check_side_view(self, image_path):
        """
        Check if person is standing straight in side view.
        Returns: (is_accepted, angle, message)
        """
        try:
            landmarks = self.load_pose_landmarks(image_path)
            
            def get_point(idx):
                return (landmarks[idx].x, landmarks[idx].y)
            
            def get_visibility(idx):
                return landmarks[idx].visibility
            
            # Get key landmarks
            LS = self.mp_pose.PoseLandmark.LEFT_SHOULDER.value
            RS = self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            LH = self.mp_pose.PoseLandmark.LEFT_HIP.value
            RH = self.mp_pose.PoseLandmark.RIGHT_HIP.value
            LA = self.mp_pose.PoseLandmark.LEFT_ANKLE.value
            RA = self.mp_pose.PoseLandmark.RIGHT_ANKLE.value
            
            # Choose the more visible side
            use_left = get_visibility(LS) > get_visibility(RS)
            
            shoulder = get_point(LS if use_left else RS)
            hip = get_point(LH if use_left else RH)
            ankle = get_point(LA if use_left else RA)
            
            # Calculate waist angle
            angle = self.calculate_angle(shoulder, hip, ankle)
            is_accepted = angle >= self.WAIST_ANGLE_THRESHOLD
            
            side_name = "left" if use_left else "right"
            message = f"Side view ({side_name}): waist angle {angle:.1f}° - {'ACCEPTED' if is_accepted else 'REJECTED'}"
            
            return is_accepted, angle, message
            
        except ValueError as e:
            logger.error(f"Side view error: {str(e)}")
            raise ValueError(f"Side view error: {str(e)}")
        except Exception as e:
            logger.critical(f"Unexpected error in side view validation: {str(e)}")
            raise RuntimeError(f"Unexpected error in side view validation: {str(e)}")
    
    def validate_images(self, front_image_path, side_image_path):
        """
        Validate both front and side images.
        Returns: (overall_success, detailed_results)
        """
        results = {
            'front_accepted': False,
            'side_accepted': False,
            'front_angle': None,
            'side_angle': None,
            'front_message': '',
            'side_message': '',
            'errors': []
        }
        
        # Check front view
        try:
            front_ok, front_angle, front_msg = self.check_front_view(front_image_path)
            results['front_accepted'] = front_ok
            results['front_angle'] = front_angle
            results['front_message'] = front_msg
        except Exception as e:
            results['errors'].append(f"Front image: {str(e)}")
            results['front_message'] = f"Front image error: {str(e)}"
            logger.error(f"Front view validation failed: {str(e)}")
        
        # Check side view
        try:
            side_ok, side_angle, side_msg = self.check_side_view(side_image_path)
            results['side_accepted'] = side_ok
            results['side_angle'] = side_angle
            results['side_message'] = side_msg
        except Exception as e:
            results['errors'].append(f"Side image: {str(e)}")
            results['side_message'] = f"Side image error: {str(e)}"
            logger.error(f"Side view validation failed: {str(e)}")
        
        # Overall success requires both images to be accepted
        overall_success = results['front_accepted'] and results['side_accepted'] and len(results['errors']) == 0
        
        return overall_success, results

# --- Clothing Size Recommendation Class ---

class ClothingSizeRecommender:
    """Recommend clothing size based on body measurements."""
    
    # Standard size chart (measurements in inches)
    SIZE_CHART = {
        'XS': {'chest': (30, 32), 'waist': (24, 26), 'hips': (32, 34)},
        'S': {'chest': (34, 36), 'waist': (26, 28), 'hips': (36, 38)},
        'M': {'chest': (36, 38), 'waist': (28, 30), 'hips': (38, 40)},
        'L': {'chest': (38, 40), 'waist': (30, 32), 'hips': (40, 42)},
        'XL': {'chest': (40, 42), 'waist': (32, 34), 'hips': (42, 44)},
        'XXL': {'chest': (42, 46), 'waist': (34, 38), 'hips': (44, 48)},
        'XXXL': {'chest': (46, 50), 'waist': (38, 42), 'hips': (48, 52)}
    }
    
    @staticmethod
    def recommend_size(chest_in, waist_in, hip_in):
        """
        Recommend clothing size based on the average of measurements in inches.
        Always returns a size - never None or empty.
        """
        logger.info(f"Recommending size for measurements (inches) - Chest: {chest_in}, Waist: {waist_in}, Hips: {hip_in}")
        # Input validation - ensure we have valid measurements
        if not all([chest_in > 0, waist_in > 0, hip_in > 0]):
            logger.warning("Invalid measurements provided for size recommendation.")
            return 'M'  # Default size for invalid inputs
        
        # Calculate the average of all three measurements
        avg_measurement = (chest_in + waist_in + hip_in) / 3
        
        # Additional check: if average is below XS range, recommend XS
        if avg_measurement < 29:
            return 'XS'
        # If average is above XXXL range, recommend XXXL
        elif avg_measurement > 47:
            return 'XXXL'
        
        # Find the size where the average falls within or closest to the ranges
        best_size = 'M'  # Default fallback
        min_distance = float('inf')
        
        for size, ranges in ClothingSizeRecommender.SIZE_CHART.items():
            # Calculate the midpoint of each range
            chest_mid = (ranges['chest'][0] + ranges['chest'][1]) / 2
            waist_mid = (ranges['waist'][0] + ranges['waist'][1]) / 2
            hip_mid = (ranges['hips'][0] + ranges['hips'][1]) / 2
            
            # Calculate the average of the midpoints for this size
            size_avg_mid = (chest_mid + waist_mid + hip_mid) / 3
            
            # Calculate distance from the user's average to this size's average
            distance = abs(avg_measurement - size_avg_mid)
            
            if distance < min_distance:
                min_distance = distance
                best_size = size
        
        return best_size

# --- Virtual Try-On Functions ---

class VirtualTryOnService:
    """Service class for handling virtual try-on operations."""
    
    @staticmethod
    def get_upload_url(image_path):
        """Get upload URL from LightX API."""
        logger.debug(f"Requesting upload URL for image: {image_path}")
        try:
            size = os.path.getsize(image_path)
            
            payload = {
                "uploadType": "imageUrl",
                "size": size,
                "contentType": CONTENT_TYPE
            }
            
            headers = {
                "Content-Type": "application/json",
                "x-api-key": LIGHTX_API_KEY
            }
            
            res = requests.post(
                f"{LIGHTX_BASE_URL}/uploadImageUrl",
                json=payload,
                headers=headers,
                timeout=30
            )
            
            res.raise_for_status()
            
            # Parse response
            response_data = res.json()
            
            # Check if response has expected structure
            if not response_data or "body" not in response_data:
                logger.error(f"Invalid API response structure: {response_data}")
                raise ValueError(f"Invalid API response structure: {response_data}")
            
            body = response_data["body"]
            
            if "uploadImage" not in body or "imageUrl" not in body:
                logger.error(f"Missing required fields in response body: {body}")
                raise ValueError(f"Missing required fields in response body: {body}")
            
            return body["uploadImage"], body["imageUrl"]
            
        except requests.exceptions.Timeout:
            logger.error("Request to LightX API timed out")
            raise TimeoutError("Request to LightX API timed out")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get upload URL: {str(e)}")
            raise RuntimeError(f"Failed to get upload URL: {str(e)}")
        except (KeyError, ValueError) as e:
            logger.error(f"Invalid response from API: {str(e)}")
            raise RuntimeError(f"Invalid response from API: {str(e)}")
    
    @staticmethod
    def upload_image(upload_url, image_path):
        logger.debug(f"Uploading image to URL: {upload_url}")
        """Upload image to the provided URL."""
        try:
            with open(image_path, "rb") as f:
                res = requests.put(
                    upload_url,
                    data=f,
                    headers={"Content-Type": CONTENT_TYPE},
                    timeout=60
                )
            res.raise_for_status()
            logger.debug("Image uploaded successfully.")
        except requests.exceptions.Timeout:
            logger.error("Image upload timed out")
            raise TimeoutError("Image upload timed out")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to upload image: {str(e)}")
            raise RuntimeError(f"Failed to upload image: {str(e)}")
    
    @staticmethod
    def start_virtual_tryon(person_url, outfit_url, segmentation_type=0):
        """Start virtual try-on process."""
        logger.debug("Starting virtual try-on process.")
        try:
            payload = {
                "imageUrl": person_url,
                "outfitImageUrl": outfit_url,
                "segmentationType": segmentation_type
            }
            
            headers = {
                "Content-Type": "application/json",
                "x-api-key": LIGHTX_API_KEY
            }
            
            res = requests.post(
                f"{LIGHTX_BASE_URL}/aivirtualtryon",
                json=payload,
                headers=headers,
                timeout=30
            )
            
            res.raise_for_status()
            
            # Parse response
            response_data = res.json()
            
            if not response_data or "body" not in response_data:
                logger.error(f"Invalid API response structure: {response_data}")
                raise ValueError(f"Invalid API response structure: {response_data}")
            
            body = response_data["body"]
            
            if "orderId" not in body:
                logger.error(f"Missing orderId in response: {body}")
                raise ValueError(f"Missing orderId in response: {body}")
            
            return body["orderId"]
            
        except requests.exceptions.Timeout:

            logger.error("Request to start virtual try-on timed out")
            raise TimeoutError("Request to start virtual try-on timed out")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to start virtual try-on: {str(e)}")
            raise RuntimeError(f"Failed to start virtual try-on: {str(e)}")
        except (KeyError, ValueError) as e:
            logger.error(f"Invalid response from API: {str(e)}")
            raise RuntimeError(f"Invalid response from API: {str(e)}")
    
    @staticmethod
    def check_status(order_id, max_attempts=60):
        logger.debug(f"Checking status for order ID: {order_id}")
        """Check the status of virtual try-on order."""
        headers = {
            "Content-Type": "application/json",
            "x-api-key": LIGHTX_API_KEY
        }
        
        payload = {"orderId": order_id}
        
        for i in range(max_attempts):
            try:
                time.sleep(3)
                
                res = requests.post(
                    f"{LIGHTX_BASE_URL}/order-status",
                    json=payload,
                    headers=headers,
                    timeout=30
                )
                
                res.raise_for_status()
                
                # Parse response
                response_data = res.json()
                
                if not response_data or "body" not in response_data:
                    logger.error(f"Invalid API response structure: {response_data}")
                    continue
                
                body = response_data["body"]
                
                if "status" not in body:
                    logger.error(f"Missing status in response body: {body}")
                    continue
                
                status = body["status"]
                logger.debug(f"Attempt {i+1}/{max_attempts}: Order status is '{status}'")
                
                if status == "active":
                    if "output" not in body:
                        logger.error("Status is active but output is missing")
                        raise ValueError("Status is active but output is missing")
                    
                    output = body["output"]
                    
                    # Handle different output formats
                    if isinstance(output, dict):
                        # If output is a dict, look for URL in common keys
                        result_url = output.get("url") or output.get("imageUrl") or output.get("resultUrl")
                        if not result_url:
                            logger.error(f"Output dict does not contain a URL: {output}")
                            raise ValueError(f"Output is a dict but no URL found: {output}")
                        return result_url
                    elif isinstance(output, str):
                        logger.info("Output is a string URL.")
                        # If output is a string, assume it's the URL
                        return output
                    else:
                        logger.error(f"Unexpected output format: {type(output)}")   
                        raise ValueError(f"Unexpected output format: {type(output)}")
                
                elif status == "failed":
                    logger.error("Virtual try-on process failed.")
                    error_msg = body.get("error", body.get("message", "Unknown error"))
                    raise RuntimeError(f"Virtual try-on failed: {error_msg}")
                
                elif status in ["pending", "processing", "queued"]:
                    logger.info(f"Order is still processing (status: {status}), continuing to poll...")
                    # Status is still processing, continue polling
                    continue
                else:
                    logger.warning(f"Unknown status '{status}' on attempt {i+1}")
                    continue
                    
            except requests.exceptions.Timeout:
                logger.error(f"Status check timed out on attempt {i+1}/{max_attempts}")
                if i == max_attempts - 1:
                    logger.error(f"Status check timed out after {max_attempts} attempts")
                    raise TimeoutError(f"Status check timed out after {max_attempts} attempts")
                continue
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error on attempt {i+1}/{max_attempts}: {str(e)}")
                if i == max_attempts - 1:
                    raise RuntimeError(f"Failed to check status after {max_attempts} attempts: {str(e)}")
                continue
            except (KeyError, ValueError) as e:
                logger.error(f"Parse error on attempt {i+1}/{max_attempts}: {str(e)}")
                if i == max_attempts - 1:
                    logger.error(f"Invalid response format after {max_attempts} attempts: {str(e)}")
                    raise RuntimeError(f"Invalid response format: {str(e)}")
                continue
        
        raise TimeoutError(f"Virtual try-on timed out after {max_attempts * 3} seconds")
    
    @staticmethod
    def download_result_image(image_url, save_path):
        logger.debug(f"Downloading result image from URL: {image_url}")
        """Download the result image from URL."""
        try:
            logger.info(f"Downloading result image to: {save_path}")
            response = requests.get(image_url, stream=True, timeout=60)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                logger.debug("Writing image data to file...")
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return save_path
            
        except requests.exceptions.Timeout:
            logger.error("Download of result image timed out")
            raise TimeoutError("Download of result image timed out")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download result image: {str(e)}")
            raise RuntimeError(f"Failed to download result image: {str(e)}")

# --- BMI and Body Type Classes ---

class BodyTypeClassifier:
    """Classify body type based on measurements and characteristics."""
    
    @staticmethod
    def classify_body_type(gender, chest, waist, hip, shoulder_width=None):
        logger.debug(f"Classifying body type for gender: {gender}, chest: {chest}, waist: {waist}, hip: {hip}, shoulder_width: {shoulder_width}")
        gender = gender.lower()
        
        if gender == 'male':
            logger.debug("Using male body type classification.")
            return BodyTypeClassifier._classify_male(chest, waist, hip, shoulder_width)
        else:
            logger.debug("Using female body type classification.")
            return BodyTypeClassifier._classify_female(chest, waist, hip)
    
    @staticmethod
    def _classify_male(chest, waist, hip, shoulder_width):
        logger.debug("Classifying male body type.")
        """Classify male body type."""
        chest_waist_ratio = chest / waist if waist > 0 else 1
        shoulder_waist_ratio = shoulder_width / waist if waist > 0 and shoulder_width else chest_waist_ratio
        
        if shoulder_waist_ratio > 1.25:
            return "inverted_triangle"
        elif chest_waist_ratio > 1.15 and hip / waist < 1.05:
            return "triangle"
        elif abs(chest - waist) < 5 and abs(hip - waist) < 5:
            return "rectangle"
        else:
            return "trapezoid"
    
    @staticmethod
    def _classify_female(chest, waist, hip):
        logger.debug("Classifying female body type.")
        """Classify female body type."""
        bust_hip_diff = abs(chest - hip)
        if bust_hip_diff < 5 and (hip - waist) > 10:
            return "hourglass"
        elif hip > chest + 5:
            return "pear"
        elif chest > hip + 5 and (chest - waist) < 10:
            return "apple"
        elif chest > hip + 8:
            return "inverted_triangle"
        else:
            return "rectangle"

class BMICalculator:
    logger.debug("Calculating BMI and categorizing.")
    """Calculate BMI and categorize."""
    
    @staticmethod
    def calculate_bmi(weight_kg, height_cm):
        logger.debug("Calculating BMI.")
        """Calculate BMI: weight(kg) / (height(m))^2"""
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        return round(bmi, 2)
    
    @staticmethod
    def categorize_bmi(bmi):
        logger.debug("Categorizing BMI.")
        """Categorize BMI into standard ranges."""
        if bmi < 18.5:
            return "underweight"
        elif 18.5 <= bmi < 25:
            return "normal"
        elif 25 <= bmi < 30:
            return "overweight"
        else:
            return "obese"

class MeasurementCorrector:
    logger.debug("Applying measurement corrections based on BMI and body type.")
    """Apply correction factors based on BMI and body type."""
    
    MALE_CORRECTIONS = {
        "inverted_triangle": {
            "underweight": {"chest": -2, "waist": -1, "hip": 0},
            "normal": {"chest": 0, "waist": 0, "hip": 0},
            "overweight": {"chest": 2, "waist": 3, "hip": 1},
            "obese": {"chest": 4, "waist": 6, "hip": 2}
        },
        "triangle": {
            "underweight": {"chest": -1, "waist": -2, "hip": -1},
            "normal": {"chest": 0, "waist": 0, "hip": 0},
            "overweight": {"chest": 1, "waist": 2, "hip": 1},
            "obese": {"chest": 3, "waist": 5, "hip": 2}
        },
        "rectangle": {
            "underweight": {"chest": -1, "waist": -1, "hip": -1},
            "normal": {"chest": 0, "waist": 0, "hip": 0},
            "overweight": {"chest": 2, "waist": 2, "hip": 1},
            "obese": {"chest": 4, "waist": 4, "hip": 3}
        },
        "trapezoid": {
            "underweight": {"chest": -1, "waist": -1, "hip": -1},
            "normal": {"chest": 0, "waist": 0, "hip": 0},
            "overweight": {"chest": 2, "waist": 2, "hip": 1},
            "obese": {"chest": 3, "waist": 4, "hip": 2}
        }
    }
    
    FEMALE_CORRECTIONS = {
        "hourglass": {
            "underweight": {"chest": -2, "waist": -2, "hip": -2},
            "normal": {"chest": 0, "waist": 0, "hip": 0},
            "overweight": {"chest": 2, "waist": 1, "hip": 3},
            "obese": {"chest": 4, "waist": 3, "hip": 5}
        },
        "pear": {
            "underweight": {"chest": -1, "waist": -1, "hip": -2},
            "normal": {"chest": 0, "waist": 0, "hip": 0},
            "overweight": {"chest": 1, "waist": 2, "hip": 4},
            "obese": {"chest": 2, "waist": 4, "hip": 6}
        },
        "apple": {
            "underweight": {"chest": -1, "waist": -2, "hip": -1},
            "normal": {"chest": 0, "waist": 0, "hip": 0},
            "overweight": {"chest": 2, "waist": 4, "hip": 1},
            "obese": {"chest": 3, "waist": 6, "hip": 2}
        },
        "inverted_triangle": {
            "underweight": {"chest": -2, "waist": -1, "hip": 0},
            "normal": {"chest": 0, "waist": 0, "hip": 0},
            "overweight": {"chest": 2, "waist": 2, "hip": 1},
            "obese": {"chest": 4, "waist": 4, "hip": 2}
        },
        "rectangle": {
            "underweight": {"chest": -1, "waist": -1, "hip": -1},
            "normal": {"chest": 0, "waist": 0, "hip": 0},
            "overweight": {"chest": 2, "waist": 2, "hip": 2},
            "obese": {"chest": 4, "waist": 4, "hip": 4}
        }
    }
    
    @staticmethod
    def apply_corrections(measurements, gender, body_type, bmi_category):
        logger.debug(f"Applying corrections for gender: {gender}, body_type: {body_type}, bmi_category: {bmi_category}")
        """Apply BMI and body type corrections to measurements."""
        corrections = (MeasurementCorrector.MALE_CORRECTIONS if gender.lower() == 'male' 
                      else MeasurementCorrector.FEMALE_CORRECTIONS)
        
        if body_type not in corrections:
            print(f"Warning: Unknown body type '{body_type}', skipping corrections")
            return measurements
        
        adjustments = corrections[body_type].get(bmi_category, {})
        
        corrected = measurements.copy()
        
        for measurement in ['chest', 'waist', 'hip']:
            if measurement in adjustments and measurement in corrected:
                adjustment = adjustments[measurement]
                corrected[measurement]['circumference']['cm'] += adjustment
                corrected[measurement]['circumference']['inches'] = round(
                    corrected[measurement]['circumference']['cm'] * 0.393701, 2
                )
        
        return corrected
    
# --- ML Size Prediction Function ---
def predict_size_ml(height_cm, chest_cm, waist_cm, hip_cm, gender):
    logger.debug("Predicting clothing size using ML model.")
    if not ML_MODEL_AVAILABLE:
        logger.warning("ML model not available for size prediction.")
        return None
    try:
        logger.debug("Preparing input features for ML model.")
        gender_encoded = GENDER_ENCODER.transform([gender])[0]

        # IMPORTANT: feature order must match training
        X = np.array([[
            height_cm,
            chest_cm,
            waist_cm,
            hip_cm,
            gender_encoded
        ]])

        X_scaled = SCALER.transform(X)
        return SIZE_MODEL.predict(X_scaled)[0]

    except Exception as e:
        print("ML prediction failed:", e)
        return None
    
# --- Measurement Calculation Class ---
class CompleteBodyMeasurementsCalculator:
    """Enhanced calculator with BMI and body type corrections."""
    
    def __init__(self, gender, weight, height):
        self.gender = gender.lower()
        self.weight = weight  # in kg
        self.height = height  # in cm
        
        if self.gender not in ['male', 'female']:
            logger.error("Invalid gender provided for measurements calculator.")
            raise ValueError("Gender must be 'male' or 'female'")
        
        # Calculate BMI
        self.bmi = BMICalculator.calculate_bmi(weight, height)
        self.bmi_category = BMICalculator.categorize_bmi(self.bmi)
    
    def calculate_all_measurements_cached(self, front_obj, side_obj):
        """
        Calculate all measurements with caching.
        Cache key based on mesh files + gender + height + weight.
        """
        # Generate cache key
        try:
            front_hash = generate_image_hash(front_obj) if os.path.exists(front_obj) else None
            side_hash = generate_image_hash(side_obj) if os.path.exists(side_obj) else None
            
            if front_hash is None or side_hash is None:
                logger.warning("Could not generate mesh hashes, proceeding without cache")
                return self.calculate_all_measurements(front_obj, side_obj)
            
            cache_key = generate_cache_key(
                'measurements',
                front_hash,
                side_hash,
                self.gender,
                self.height,
                self.weight
            )
            
            # Check cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.info(f"✓ Cache hit for measurements: {cache_key}")
                return cached_result
            
            # Cache miss - calculate measurements
            logger.info(f"Cache miss for measurements: {cache_key}, calculating...")
            result = self.calculate_all_measurements(front_obj, side_obj)
            
            if result is not None:
                # Cache the result (1800 seconds = 30 minutes)
                cache.set(cache_key, result, timeout=1800)
                logger.info(f"✓ Cached measurements: {cache_key}")
            
            return result
            
        except Exception as e:
            logger.warning(f"Cache error in measurements: {e}, proceeding without cache")
            return self.calculate_all_measurements(front_obj, side_obj)

    def __init__(self, gender, weight, height):
        self.gender = gender.lower()
        self.weight = weight  # in kg
        self.height = height  # in cm
        
        if self.gender not in ['male', 'female']:
            logger.error("Invalid gender provided for measurements calculator.")
            raise ValueError("Gender must be 'male' or 'female'")
        
        # Calculate BMI
        self.bmi = BMICalculator.calculate_bmi(weight, height)
        self.bmi_category = BMICalculator.categorize_bmi(self.bmi)
    
    def load_obj_file(self, filepath, is_side_view=False):
        logger.debug(f"Loading OBJ file: {filepath} (side view: {is_side_view})")
        """Load OBJ file, detect units, and apply auto-rotation for side views."""
        try:
            logger.debug(f"Loading mesh from file: {filepath}")
            mesh = trimesh.load(filepath)
            mesh, scale_factor = self.detect_and_convert_units(mesh, filepath)
            if is_side_view:
                logger.debug("Applying auto-rotation for side view mesh.")
                mesh = self.auto_rotate_side_view(mesh)
            return mesh
        except Exception as e:
            logger.error(f"Error loading OBJ file: {str(e)}")
            print(f"Error loading {filepath}: {e}")
            return None
    
    def detect_and_convert_units(self, mesh, filepath):
        logger.debug(f"Detecting units for mesh from file: {filepath}")
        """Scales the mesh if dimensions suggest it's in meters."""
        bbox = mesh.bounds
        max_dim = max(bbox[1] - bbox[0])
        
        if max_dim < 5.0:
            scale_factor = 100.0
            scale_matrix = np.eye(4) * scale_factor
            scale_matrix[3, 3] = 1.0
            mesh.apply_transform(scale_matrix)
        else:
            scale_factor = 1.0
        
        return mesh, scale_factor
    
    def auto_rotate_side_view(self, mesh):
        """Rotate side view mesh if needed."""
        bbox = mesh.bounds
        x_range = bbox[1][0] - bbox[0][0]
        z_range = bbox[1][2] - bbox[0][2]
        
        if z_range < x_range * 0.1:
            angle = np.radians(90)
            rotation_matrix = np.array([
                [np.cos(angle), 0, np.sin(angle), 0],
                [0, 1, 0, 0],
                [-np.sin(angle), 0, np.cos(angle), 0],
                [0, 0, 0, 1]
            ])
            mesh.apply_transform(rotation_matrix)
        
        return mesh
    
    def detect_landmarks(self, mesh, percentage_from_top):
        logger.debug(f"Detecting landmarks at {percentage_from_top*100}% from top.")
        """Extract vertices within a vertical slice."""
        vertices = mesh.vertices
        bbox = mesh.bounds
        height = bbox[1][1] - bbox[0][1]
        
        target = bbox[1][1] - (height * percentage_from_top)
        tol = height * 0.02
        
        return vertices[
            (vertices[:, 1] >= target - tol) & 
            (vertices[:, 1] <= target + tol)
        ]
    
    def calculate_width(self, mesh, percentage_from_top):
        logger.debug(f"Calculating width at {percentage_from_top*100}% from top.")
        """Calculates width at a specified vertical percentage."""
        v = self.detect_landmarks(mesh, percentage_from_top)
        if len(v) == 0:
            return 0.0
        return abs(np.max(v[:, 0]) - np.min(v[:, 0]))
    
    def calculate_depth(self, mesh, percentage_from_top):
        logger.debug(f"Calculating depth at {percentage_from_top*100}% from top.")
        """Calculates depth at a specified vertical percentage."""
        v = self.detect_landmarks(mesh, percentage_from_top)
        if len(v) == 0:
            return 0.0
        return abs(np.max(v[:, 2]) - np.min(v[:, 2]))
    
    def ramanujan_ellipse_circumference(self, a, b):
        logger.debug(f"Calculating ellipse circumference with a={a}, b={b}.")
        """Approximation of ellipse circumference using Ramanujan's formula."""
        if a <= 0 or b <= 0:
            return 0.0
        return math.pi * (3*(a+b) - math.sqrt((3*a+b)*(a+3*b)))
    
    def get_semi_axes(self, width, depth, mtype):
        logger.debug(f"Getting semi-axes for measurement type: {mtype}.")
        """Adjusts width/depth to semi-axes for the ellipse formula."""
        if mtype == 'neck':
            a = width / 3
            b = depth / 3 if self.gender == 'male' else depth / 4
        elif mtype == 'chest':
            a = width / 3 if self.gender == 'male' else width / 2
            b = depth / 4
        elif mtype == 'waist':
            a = width / 3
            b = depth / 4
        elif mtype == 'hip':
            a = width / 3 if self.gender == 'male' else width / 2
            b = depth / 4
        else:
            a = width / 2
            b = depth / 2
        return a, b
    
    def adjust_chest_by_weight(self, chest_circumference):
        logger.debug("Adjusting chest circumference based on weight (legacy method).")
        """Adjust chest circumference based on weight (males only) - legacy method."""
        if self.gender != 'male':
            return chest_circumference
        
        if 55 <= self.weight <= 65:
            return chest_circumference + 3
        elif 67 <= self.weight <= 75:
            return chest_circumference + 7
        elif 75 < self.weight <= 85:
            return chest_circumference + 10
        else:
            return chest_circumference
    
    def estimate_shoulder_width(self, mesh, real_height):
        logger.debug("Estimating shoulder width.")
        """Estimates shoulder width."""
        vertices = mesh.vertices
        y = vertices[:, 1]
        Ymin, Ymax = np.min(y), np.max(y)
        H = Ymax - Ymin
        
        if H == 0:
            return 0.0
        
        scale = real_height / H
        mask = (y >= Ymin + H*0.79) & (y <= Ymin + H*0.95)
        slice_vertices = vertices[mask]
        
        if len(slice_vertices) == 0:
            return 0.0
        
        x_min = np.min(slice_vertices[:, 0])
        x_max = np.max(slice_vertices[:, 0])
        
        return abs(x_max - x_min) * scale
    
    def compute_arm_sections(self, total_arm_length):
        logger.debug("Computing arm sections.")
        """Estimates arm sections based on proportional breakdown."""
        hand_to_elbow = total_arm_length / 2
        shoulder_to_elbow = total_arm_length * 0.58
        return hand_to_elbow, shoulder_to_elbow
    
    def calculate_all_measurements(self, front_obj, side_obj):
        logger.debug("Calculating all measurements with corrections.")
        """Main method to calculate all measurements with BMI and body type corrections."""
        front_mesh = self.load_obj_file(front_obj, is_side_view=False)
        side_mesh = self.load_obj_file(side_obj, is_side_view=True)
        
        if front_mesh is None or side_mesh is None:
            return None
        
        cm_to_in = 0.393701
        results = {}
        
        measurement_points = [
            ('neck', 0.07),
            ('chest', 0.28),
            ('waist', 0.42),
            ('hip', 0.58)
        ]
        
        # Calculate raw circumferences
        for name, pct in measurement_points:
            logger.debug(f"Calculating {name} measurement.")
            w = self.calculate_width(front_mesh, pct)
            d = self.calculate_depth(side_mesh, pct)
            
            a, b = self.get_semi_axes(w, d, name)
            c = self.ramanujan_ellipse_circumference(a, b)
            
            # Apply legacy male chest adjustment
            if name == 'chest' and self.gender == 'male':
                c = c + 5.0
                c = self.adjust_chest_by_weight(c)
            
            results[name] = {
                'circumference': {'cm': round(c, 2), 'inches': round(c * cm_to_in, 2)}
            }
        
        # Shoulder width
        sw = self.estimate_shoulder_width(front_mesh, self.height)
        results['shoulder'] = {
            'width': {'cm': round(sw, 2), 'inches': round(sw * cm_to_in, 2)}
        }
        
        # Classify body type
        body_type = BodyTypeClassifier.classify_body_type(
            self.gender,
            results['chest']['circumference']['cm'],
            results['waist']['circumference']['cm'],
            results['hip']['circumference']['cm'],
            sw
        )
        
        # Apply BMI and body type corrections
        results = MeasurementCorrector.apply_corrections(
            results, self.gender, body_type, self.bmi_category
        )
        
        # Calculate recommended clothing size
        chest_cm = results['chest']['circumference']['cm']
        waist_cm = results['waist']['circumference']['cm']
        hip_cm   = results['hip']['circumference']['cm']

        ml_size = predict_size_ml(
                    self.height,
                    chest_cm,
                    waist_cm,
                    hip_cm,
                    self.gender
        )
        chest_in = chest_cm * 0.393701
        waist_in = waist_cm * 0.393701
        hip_in   = hip_cm * 0.393701

        recommended_size = ml_size if ml_size else ClothingSizeRecommender.recommend_size(
                            chest_in, waist_in, hip_in
                            )
        
        # Arm sections
        total_arm = 0.36 * self.height
        hand, shoulder = self.compute_arm_sections(total_arm)
        display_arm = total_arm + 4
        
        results['arm'] = {
            'hand_to_elbow': {'cm': round(hand, 2), 'inches': round(hand * cm_to_in, 2)},
            'shoulder_to_elbow': {'cm': round(shoulder, 2), 'inches': round(shoulder * cm_to_in, 2)},
            'total_length': {'cm': int(display_arm), 'inches': int(display_arm * cm_to_in)}
        }
        
        # Add metadata
        results['metadata'] = {
            'bmi': self.bmi,
            'bmi_category': self.bmi_category,
            'body_type': body_type,
            'recommended_size': recommended_size,
            'height': {'cm': self.height, 'inches': round(self.height * cm_to_in, 2)},
            'weight': {'kg': self.weight, 'lbs': round(self.weight * 2.20462, 2)}
        }
        
        return results

# --- HMR2 Processing Function ---

def process_image_to_mesh(img_path, output_path, model, detector, renderer, model_cfg):
    """Process image to 3D mesh using HMR2 with caching."""
    
    # Generate cache key based on image content
    img_hash = generate_image_hash(img_path)
    if img_hash is None:
        logger.warning("Could not generate image hash, proceeding without cache")
        return _process_image_to_mesh_internal(img_path, output_path, model, detector, renderer, model_cfg)
    
    cache_key = f"hmr2_mesh_{img_hash}"
    
    # Check if mesh exists in cache
    cached_mesh_data = cache.get(cache_key)
    
    if cached_mesh_data is not None:
        logger.info(f"✓ Cache hit for HMR2 mesh: {cache_key}")
        try:
            # Write cached mesh data to output file
            with open(output_path, 'w') as f:
                f.write(cached_mesh_data)
            return output_path
        except Exception as e:
            logger.warning(f"Failed to restore cached mesh: {e}")
    
    # Cache miss - process the image
    logger.info(f"Cache miss for HMR2 mesh: {cache_key}, processing...")
    result_path = _process_image_to_mesh_internal(img_path, output_path, model, detector, renderer, model_cfg)
    
    if result_path and os.path.exists(result_path):
        try:
            # Cache the mesh file content (3600 seconds = 60 minutes)
            with open(result_path, 'r') as f:
                mesh_data = f.read()
            cache.set(cache_key, mesh_data, timeout=3600)  # 60 minutes
            logger.info(f"✓ Cached HMR2 mesh: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to cache mesh: {e}")
    
    return result_path

def _process_image_to_mesh_internal(img_path, output_path, model, detector, renderer, model_cfg):
    """Internal function - actual HMR2 processing logic."""
    logger.debug(f"Processing image to 3D mesh: {img_path} -> {output_path}")
    img_cv2 = cv2.imread(str(img_path))
    
    det_out = detector(img_cv2)
    det_instances = det_out['instances']
    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
    boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    
    if len(boxes) == 0:
        return None
    
    dataset = ViTDetDataset(model_cfg, img_cv2, boxes)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    for batch in dataloader:
        logger.debug("Running HMR2 model inference.")
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)
        
        verts = out['pred_vertices'][0].detach().cpu().numpy()
        pred_cam = out['pred_cam']
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        
        scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
        pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()
        
        camera_translation = pred_cam_t_full[0]
        
        tmesh = renderer.vertices_to_trimesh(verts, camera_translation, (0.65, 0.74, 0.86))
        tmesh.export(output_path)
        
        return output_path
        
    return None
# --- HMR2 Model Initialization ---

model = None
model_cfg = None
detector = None
renderer = None

if HMR2_AVAILABLE:
    logger.debug("Loading HMR2 model and dependencies.")
    download_models(CACHE_DIR_4DHUMANS)
    model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
    model = model.to(device)
    model.eval()

    cfg_path = Path(hmr2.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    renderer = Renderer(model_cfg, faces=model.smpl.faces)
    logger.info("Model loaded successfully!")
else:
    logger.warning("HMR2 model or dependencies not available. Running in UI-only mode.")

# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/virtual-try-on')
def virtual_try_on():
    return render_template('virtual_try_on.html')

@app.route('/virtual-tryon-process', methods=['POST'])
def virtual_tryon_process():
    """Process virtual try-on request."""
    person_path = None
    clothing_path = None
    
    try:
        logger.debug("Starting virtual try-on process.")
        # Check for uploaded files
        if 'person_image' not in request.files or 'clothing_image' not in request.files:
            logger.debug("Missing person or clothing image in request.")
            return jsonify({'success': False, 'error': 'Both person and clothing images are required'})
        
        person_file = request.files['person_image']
        clothing_file = request.files['clothing_image']
        
        if person_file.filename == '' or clothing_file.filename == '':
            logger.debug("No files selected for upload.")
            return jsonify({'success': False, 'error': 'Please select both images'})
        
        if not (allowed_file(person_file.filename) and allowed_file(clothing_file.filename)):
            logger.debug("Invalid file type uploaded.")
            return jsonify({'success': False, 'error': 'Invalid file type. Use PNG, JPG, or JPEG'})
        
        # Get clothing type (0=upper, 1=lower, 2=full)
        clothing_type = int(request.form.get('clothing_type', 0))
        
        # Save uploaded files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        person_filename = secure_filename(f"person_{timestamp}_{person_file.filename}")
        clothing_filename = secure_filename(f"clothing_{timestamp}_{clothing_file.filename}")
        
        person_path = os.path.join(app.config['UPLOAD_FOLDER'], person_filename)
        clothing_path = os.path.join(app.config['UPLOAD_FOLDER'], clothing_filename)
        
        person_file.save(person_path)
        clothing_file.save(clothing_path)
        
        # Initialize virtual try-on service
        tryon_service = VirtualTryOnService()
        
        # Step 1: Get upload URLs
        logger.debug("Obtaining upload URLs from LightX API.")
        person_upload_url, person_image_url = tryon_service.get_upload_url(person_path)
        logger.debug(f"Person image URL obtained: {person_image_url[:50]}...")
        
        clothing_upload_url, clothing_image_url = tryon_service.get_upload_url(clothing_path)
        logger.debug(f"Clothing image URL obtained: {clothing_image_url[:50]}...")
        
        # Step 2: Upload images
        logger.debug("Uploading images to LightX API...")
        tryon_service.upload_image(person_upload_url, person_path)
        logger.debug("Person image uploaded successfully")
        
        tryon_service.upload_image(clothing_upload_url, clothing_path)
        logger.debug("Clothing image uploaded successfully")
        
        # Step 3: Start virtual try-on
        logger.debug("Starting virtual try-on...")
        order_id = tryon_service.start_virtual_tryon(
            person_image_url, 
            clothing_image_url, 
            clothing_type
        )
        logger.info(f"Virtual try-on started with order ID: {order_id}")
        
        # Step 4: Check status and get result
        logger.info(f"Checking status for order: {order_id}")
        result_url = tryon_service.check_status(order_id)
        logger.info(f"Result URL obtained: {result_url[:50]}...")
        
        # Step 5: Download result image
        result_filename = f"tryon_result_{timestamp}.jpg"
        result_path = os.path.join(app.config['TRYON_FOLDER'], result_filename)
        tryon_service.download_result_image(result_url, result_path)
        logger.info(f"Result image saved to: {result_path}")
        
        # Return success with result
        return jsonify({
            'success': True,
            'result_image_url': f'/tryon-result/{result_filename}',
            'download_url': f'/download-tryon/{result_filename}',
            'filename': result_filename
        })
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        return jsonify({'success': False, 'error': f'API request failed: {str(e)}'})
    except TimeoutError as e:
        logger.error(f"Timeout error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})
    except RuntimeError as e:
        logger.error(f"Runtime error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'An unexpected error occurred: {str(e)}'})
    finally:
        # Cleanup uploaded files
        for file_path in [person_path, clothing_path]:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.debug(f"Cleaned up file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete file {file_path}: {str(e)}")  

@app.route('/tryon-result/<filename>')
def serve_tryon_result(filename):
    logger.debug(f"Serving try-on result image: {filename}")
    """Serve the virtual try-on result image."""
    return send_file(
        os.path.join(app.config['TRYON_FOLDER'], filename),
        mimetype='image/jpeg'
    )

@app.route('/download-tryon/<filename>')
def download_tryon(filename):
    logger.debug(f"Downloading try-on result image: {filename}")
    """Download the virtual try-on result image."""
    return send_file(
        os.path.join(app.config['TRYON_FOLDER'], filename),
        mimetype='image/jpeg',
        as_attachment=True,
        download_name=filename
    )

@app.route('/process', methods=['POST'])
def process():
    logger.debug("Starting body measurement processing.")
    front_path = None
    side_path = None
    
    try:
        # Check if HMR2 is available
        if not HMR2_AVAILABLE:
            logger.debug("HMR2 model not installed.")
            return jsonify({'success': False, 'error': 'HMR2 model is not installed. Please install the HMR2 dependencies to enable body measurement functionality.'})
        
        # Get form data
        gender = request.form.get('gender', 'male')
        height = float(request.form.get('height', 170))
        height_unit = request.form.get('height_unit', 'cm')
        weight = float(request.form.get('weight', 70))
        weight_unit = request.form.get('weight_unit', 'kg')
        
        # Convert height to cm if needed
        if height_unit == 'm':
            height = height * 100
        
        # Convert weight to kg if needed
        if weight_unit == 'lbs':
            weight = weight * 0.453592
        
        # Check for uploaded files
        if 'front_image' not in request.files or 'side_image' not in request.files:
            return jsonify({'success': False, 'error': 'Both front and side images are required'})
        
        front_file = request.files['front_image']
        side_file = request.files['side_image']
        
        if front_file.filename == '' or side_file.filename == '':
            return jsonify({'success': False, 'error': 'Please select both images'})
        
        if not (allowed_file(front_file.filename) and allowed_file(side_file.filename)):
            return jsonify({'success': False, 'error': 'Invalid file type. Use PNG, JPG, or JPEG'})
        
        # Save uploaded files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        front_filename = secure_filename(f"front_{timestamp}_{front_file.filename}")
        side_filename = secure_filename(f"side_{timestamp}_{side_file.filename}")
        
        front_path = os.path.join(app.config['UPLOAD_FOLDER'], front_filename)
        side_path = os.path.join(app.config['UPLOAD_FOLDER'], side_filename)
        
        front_file.save(front_path)
        side_file.save(side_path)
        
        # === POSE VALIDATION ===
        if MEDIAPIPE_AVAILABLE:
            logger.debug("Performing pose validation using MediaPipe.")
            try:
                validator = PoseValidator()
                is_valid, validation_results = validator.validate_images_cached(front_path, side_path)
                
                if not is_valid:
                    # Build detailed error message
                    error_messages = []
                    
                    if not validation_results['front_accepted']:
                        if validation_results['front_angle'] is not None:
                            error_messages.append(
                                f"❌ Front image: Person appears to be bending. "
                                # f"Detected waist angle: {validation_results['front_angle']:.1f}° "
                                # f"(minimum required: {PoseValidator.WAIST_ANGLE_THRESHOLD}°). "
                                f"Please upload a front image where you're standing straight with arms by your sides."
                            )
                        else:
                            error_messages.append(
                                f"❌ Front image: {validation_results['front_message']}"
                            )
                    
                    if not validation_results['side_accepted']:
                        if validation_results['side_angle'] is not None:
                            error_messages.append(
                                f"❌ Side image: Person appears to be bending. "
                                f"Detected waist angle: {validation_results['side_angle']:.1f}° "
                                f"(minimum required: {PoseValidator.WAIST_ANGLE_THRESHOLD}°). "
                                f"Please upload a side image where you're standing straight."
                            )
                        else:
                            error_messages.append(
                                f" Side image: {validation_results['side_message']}"
                            )
                    
                    # Add general validation errors
                    if validation_results['errors']:
                        logger.debug("Adding general validation errors.")
                        for err in validation_results['errors']:
                            logger.debug(f"Validation error: {err}")
                            if err not in str(error_messages): 
                                logger.debug("Appending new error message.") 
                                error_messages.append(f"⚠️ {err}")
                    
                    error_message = "\n\n".join(error_messages)
                    error_message += "\n\n💡 Tips for better results:\n" \
                                    "• Stand upright with your back straight\n" \
                                    "• Keep arms relaxed at your sides\n" \
                                    "• Ensure full body is visible in frame\n" \
                                    "• Use good lighting and clear background."
                    
                    logger.debug("Pose validation failed.")
                    return jsonify({
                        'success': False,
                        'error': error_message,
                        'validation_details': validation_results
                    })
                
                logger.debug("✓ Pose validation passed for both images")
                logger.debug(f"  - Front: {validation_results['front_message']}")
                logger.debug(f"  - Side: {validation_results['side_message']}")
                
            except Exception as e:
                logger.warning(f"Pose validation failed with error: {str(e)}")
                logger.debug("Proceeding without pose validation...")
                # Continue processing even if validation fails
        else:
            logger.debug("Warning: MediaPipe not available, skipping pose validation")
        
        # === PROCEED WITH HMR2 PROCESSING ===
        # Process images to 3D meshes
        front_obj = os.path.join(app.config['OUTPUT_FOLDER'], f'front_mesh_{timestamp}.obj')
        side_obj = os.path.join(app.config['OUTPUT_FOLDER'], f'side_mesh_{timestamp}.obj')
        
        logger.debug("Processing front image with HMR2...")
        front_result = process_image_to_mesh(front_path, front_obj, model, detector, renderer, model_cfg)
        
        if front_result is None:
            return jsonify({
                'success': False,
                'error': 'Could not detect a person in the front image. Please ensure:\n'
                        '• The full body is visible in the frame\n'
                        '• You are standing against a clear background\n'
                        '• The image has good lighting\n'
                        '• You are the only person in the image'
            })
        
        logger.debug("Processing side image with HMR2...")
        side_result = process_image_to_mesh(side_path, side_obj, model, detector, renderer, model_cfg)
        
        if side_result is None:
            return jsonify({
                'success': False,
                'error': 'Could not detect a person in the side image. Please ensure:\n'
                        '• The full body is visible in the frame\n'
                        '• You are standing against a clear background\n'
                        '• The image has good lighting\n'
                        '• You are the only person in the image'
            })
        
        # Calculate measurements with BMI and body type corrections
        logger.debug("Calculating body measurements...")
        calculator = CompleteBodyMeasurementsCalculator(gender, weight, height)
        measurements = calculator.calculate_all_measurements_cached(front_obj, side_obj)
        
        if measurements is None:
            return jsonify({
                'success': False,
                'error': 'Error calculating measurements from the 3D models. This may happen if:\n'
                        '• The body pose is too complex\n'
                        '• Parts of the body are obscured\n'
                        '• The images are of low quality\n\n'
                        'Please try again with clearer images where you are standing straight.'
            })
        
        logger.debug("✓ Measurements calculated successfully!")
        
        # Cleanup uploaded files and temporary mesh files
        for f in [front_path, side_path, front_obj, side_obj]:
            if f and os.path.exists(f):
                try:
                    os.remove(f)
                except Exception as e:
                    logger.warning(f"Warning: Could not cleanup {f}: {str(e)}")
        
        return jsonify({'success': True, 'measurements': measurements})
        
    except ValueError as e:
        # User input validation errors
        return jsonify({'success': False, 'error': str(e)})
    except Exception as e:
        logger.error(f"Unexpected error in /process: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'An unexpected error occurred while processing your images. Please try again with different images.'
        })
    finally:
        # Ensure cleanup of uploaded files
        for file_path in [front_path, side_path]:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.debug(f"Cleaned up: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {file_path}: {str(e)}")


@app.route('/admin/clear-cache', methods=['POST'])
def clear_cache():
    """Clear all cache entries."""
    try:
        cache.clear()
        logger.info("Cache cleared successfully")
        return jsonify({'success': True, 'message': 'Cache cleared successfully'})
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/cache-stats', methods=['GET'])
def cache_stats():
    """Get cache statistics."""
    try:
        # This works with FileSystemCache
        cache_dir = app.config.get('CACHE_DIR', 'cache')
        if os.path.exists(cache_dir):
            files = os.listdir(cache_dir)
            total_size = sum(os.path.getsize(os.path.join(cache_dir, f)) 
                           for f in files if os.path.isfile(os.path.join(cache_dir, f)))
            return jsonify({
                'success': True,
                'cache_entries': len(files),
                'total_size_mb': round(total_size / (1024 * 1024), 2)
            })
        return jsonify({'success': True, 'cache_entries': 0, 'total_size_mb': 0})
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)