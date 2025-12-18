from flask import Flask, request, jsonify, render_template, send_file
import os
import torch
import cv2
import numpy as np
from pathlib import Path
from werkzeug.utils import secure_filename
import trimesh
import math
import requests
import time
from datetime import datetime

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
except ImportError as e:
    print(f"HMR2 not available: {e}")
    print("The application will run in UI-only mode. Install HMR2 for full functionality.")

# Force CPU loading for PyTorch models
_original_load = torch.load
def cpu_load(*args, **kwargs):
    kwargs['map_location'] = torch.device('cpu')
    return _original_load(*args, **kwargs)
torch.load = cpu_load
device = torch.device('cpu')

# --- Flask App Configuration ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['TRYON_FOLDER'] = 'tryon_results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['TRYON_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Virtual Try-On API Configuration
LIGHTX_API_KEY = "ae0ddbab09454d599116b0ec308dec7c_a43a8874d35d4cc88b85d224711d1d07_andoraitools"  
LIGHTX_BASE_URL = "https://api.lightxeditor.com/external/api/v2"
CONTENT_TYPE = "image/jpeg"

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Clothing Size Recommendation Class ---
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
        # Input validation - ensure we have valid measurements
        if not all([chest_in > 0, waist_in > 0, hip_in > 0]):
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
            headers=headers
        )
        
        res.raise_for_status()
        body = res.json()["body"]
        
        return body["uploadImage"], body["imageUrl"]
    
    @staticmethod
    def upload_image(upload_url, image_path):
        """Upload image to the provided URL."""
        with open(image_path, "rb") as f:
            res = requests.put(
                upload_url,
                data=f,
                headers={"Content-Type": CONTENT_TYPE}
            )
        res.raise_for_status()
    
    @staticmethod
    def start_virtual_tryon(person_url, outfit_url, segmentation_type=0):
        """Start virtual try-on process."""
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
            headers=headers
        )
        
        res.raise_for_status()
        return res.json()["body"]["orderId"]
    
    @staticmethod
    def check_status(order_id, max_attempts=60):
        """Check the status of virtual try-on order."""
        headers = {
            "Content-Type": "application/json",
            "x-api-key": LIGHTX_API_KEY
        }
        
        payload = {"orderId": order_id}
        
        for i in range(max_attempts):
            time.sleep(3)
            
            res = requests.post(
                f"{LIGHTX_BASE_URL}/order-status",
                json=payload,
                headers=headers
            )
            
            res.raise_for_status()
            body = res.json()["body"]
            
            if body["status"] == "active":
                return body["output"]
            
            if body["status"] == "failed":
                raise RuntimeError("Virtual try-on failed")
        
        raise TimeoutError("Virtual try-on timed out")
    
    @staticmethod
    def download_result_image(image_url, save_path):
        """Download the result image from URL."""
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return save_path

# --- BMI and Body Type Classes ---

class BodyTypeClassifier:
    """Classify body type based on measurements and characteristics."""
    
    @staticmethod
    def classify_body_type(gender, chest, waist, hip, shoulder_width=None):
        """
        Classify body type based on measurements.
        
        Body types:
        Male: Rectangle, Triangle (V-shape), Inverted Triangle, Trapezoid
        Female: Hourglass, Pear, Apple, Rectangle, Inverted Triangle
        """
        gender = gender.lower()
        
        if gender == 'male':
            return BodyTypeClassifier._classify_male(chest, waist, hip, shoulder_width)
        else:
            return BodyTypeClassifier._classify_female(chest, waist, hip)
    
    @staticmethod
    def _classify_male(chest, waist, hip, shoulder_width):
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
    """Calculate BMI and categorize."""
    
    @staticmethod
    def calculate_bmi(weight_kg, height_cm):
        """Calculate BMI: weight(kg) / (height(m))^2"""
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        return round(bmi, 2)
    
    @staticmethod
    def categorize_bmi(bmi):
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

# --- Measurement Calculation Class ---

class CompleteBodyMeasurementsCalculator:
    """Enhanced calculator with BMI and body type corrections."""
    
    def __init__(self, gender, weight, height):
        self.gender = gender.lower()
        self.weight = weight  # in kg
        self.height = height  # in cm
        
        if self.gender not in ['male', 'female']:
            raise ValueError("Gender must be 'male' or 'female'")
        
        # Calculate BMI
        self.bmi = BMICalculator.calculate_bmi(weight, height)
        self.bmi_category = BMICalculator.categorize_bmi(self.bmi)
    
    def load_obj_file(self, filepath, is_side_view=False):
        """Load OBJ file, detect units, and apply auto-rotation for side views."""
        try:
            mesh = trimesh.load(filepath)
            mesh, scale_factor = self.detect_and_convert_units(mesh, filepath)
            if is_side_view:
                mesh = self.auto_rotate_side_view(mesh)
            return mesh
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def detect_and_convert_units(self, mesh, filepath):
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
        """Calculates width at a specified vertical percentage."""
        v = self.detect_landmarks(mesh, percentage_from_top)
        if len(v) == 0:
            return 0.0
        return abs(np.max(v[:, 0]) - np.min(v[:, 0]))
    
    def calculate_depth(self, mesh, percentage_from_top):
        """Calculates depth at a specified vertical percentage."""
        v = self.detect_landmarks(mesh, percentage_from_top)
        if len(v) == 0:
            return 0.0
        return abs(np.max(v[:, 2]) - np.min(v[:, 2]))
    
    def ramanujan_ellipse_circumference(self, a, b):
        """Approximation of ellipse circumference using Ramanujan's formula."""
        if a <= 0 or b <= 0:
            return 0.0
        return math.pi * (3*(a+b) - math.sqrt((3*a+b)*(a+3*b)))
    
    def get_semi_axes(self, width, depth, mtype):
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
        """Estimates arm sections based on proportional breakdown."""
        hand_to_elbow = total_arm_length / 2
        shoulder_to_elbow = total_arm_length * 0.58
        return hand_to_elbow, shoulder_to_elbow
    
    def calculate_all_measurements(self, front_obj, side_obj):
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
        recommended_size = ClothingSizeRecommender.recommend_size(
            results['chest']['circumference']['inches'],
            results['waist']['circumference']['inches'],
            results['hip']['circumference']['inches']
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
    """Process image to 3D mesh using HMR2."""
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
    print("Loading HMR2 model...")
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
    print("Model loaded successfully!")
else:
    print("Running in UI-only mode. HMR2 model not loaded.")

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
    try:
        # Check for uploaded files
        if 'person_image' not in request.files or 'clothing_image' not in request.files:
            return jsonify({'success': False, 'error': 'Both person and clothing images are required'})
        
        person_file = request.files['person_image']
        clothing_file = request.files['clothing_image']
        
        if person_file.filename == '' or clothing_file.filename == '':
            return jsonify({'success': False, 'error': 'Please select both images'})
        
        if not (allowed_file(person_file.filename) and allowed_file(clothing_file.filename)):
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
        print("Getting upload URLs...")
        person_upload_url, person_image_url = tryon_service.get_upload_url(person_path)
        clothing_upload_url, clothing_image_url = tryon_service.get_upload_url(clothing_path)
        
        # Step 2: Upload images
        print("Uploading images to LightX API...")
        tryon_service.upload_image(person_upload_url, person_path)
        tryon_service.upload_image(clothing_upload_url, clothing_path)
        
        # Step 3: Start virtual try-on
        print("Starting virtual try-on...")
        order_id = tryon_service.start_virtual_tryon(
            person_image_url, 
            clothing_image_url, 
            clothing_type
        )
        
        # Step 4: Check status and get result
        print(f"Checking status for order: {order_id}")
        result_url = tryon_service.check_status(order_id)
        
        # Step 5: Download result image
        result_filename = f"tryon_result_{timestamp}.jpg"
        result_path = os.path.join(app.config['TRYON_FOLDER'], result_filename)
        tryon_service.download_result_image(result_url, result_path)
        
        # Cleanup uploaded files
        for f in [person_path, clothing_path]:
            if os.path.exists(f):
                os.remove(f)
        
        # Return success with result
        return jsonify({
            'success': True,
            'result_image_url': f'/tryon-result/{result_filename}',
            'download_url': f'/download-tryon/{result_filename}',
            'filename': result_filename
        })
        
    except requests.exceptions.RequestException as e:
        return jsonify({'success': False, 'error': f'API request failed: {str(e)}'})
    except TimeoutError:
        return jsonify({'success': False, 'error': 'Virtual try-on processing timed out. Please try again.'})
    except Exception as e:
        return jsonify({'success': False, 'error': f'An error occurred: {str(e)}'})


@app.route('/tryon-result/<filename>')
def serve_tryon_result(filename):
    """Serve the virtual try-on result image."""
    return send_file(
        os.path.join(app.config['TRYON_FOLDER'], filename),
        mimetype='image/jpeg'
    )

@app.route('/download-tryon/<filename>')
def download_tryon(filename):
    """Download the virtual try-on result image."""
    return send_file(
        os.path.join(app.config['TRYON_FOLDER'], filename),
        mimetype='image/jpeg',
        as_attachment=True,
        download_name=filename
    )

@app.route('/process', methods=['POST'])
def process():
    try:
        # Check if HMR2 is available
        if not HMR2_AVAILABLE:
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
        front_filename = secure_filename(f"front_{front_file.filename}")
        side_filename = secure_filename(f"side_{side_file.filename}")
        
        front_path = os.path.join(app.config['UPLOAD_FOLDER'], front_filename)
        side_path = os.path.join(app.config['UPLOAD_FOLDER'], side_filename)
        
        front_file.save(front_path)
        side_file.save(side_path)
        
        # Process images to 3D meshes
        front_obj = os.path.join(app.config['OUTPUT_FOLDER'], 'front_mesh.obj')
        side_obj = os.path.join(app.config['OUTPUT_FOLDER'], 'side_mesh.obj')
        
        front_result = process_image_to_mesh(front_path, front_obj, model, detector, renderer, model_cfg)
        side_result = process_image_to_mesh(side_path, side_obj, model, detector, renderer, model_cfg)
        
        if front_result is None or side_result is None:
            return jsonify({'success': False, 'error': 'Could not detect a person in one or both images. Please use clear, full-body photos.'})
        
        # Calculate measurements with BMI and body type corrections
        calculator = CompleteBodyMeasurementsCalculator(gender, weight, height)
        measurements = calculator.calculate_all_measurements(front_obj, side_obj)
        
        if measurements is None:
            return jsonify({'success': False, 'error': 'Error calculating measurements'})
        
        # Cleanup
        for f in [front_path, side_path]:
            if os.path.exists(f):
                os.remove(f)
        
        return jsonify({'success': True, 'measurements': measurements})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)