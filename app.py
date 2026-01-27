from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import time
import pandas as pd
import datetime
import pickle
from deepface import DeepFace
import threading
import json
import concurrent.futures
import numpy as np
from functools import lru_cache
import shutil
from pathlib import Path
import traceback
import sqlite3
from contextlib import contextmanager
import hashlib
import tempfile
import cloudinary
import cloudinary.uploader
import cloudinary.api
import requests
from io import BytesIO
from PIL import Image
import io

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configuration
DATASET_PATH = "dataset"
ATTENDANCE_FILE = "attendance.csv"
EMBEDDINGS_FILE = "embeddings.pkl"
TRAINING_STATUS_FILE = "training_status.json"
TEMP_DIR = "temp"
DATABASE_FILE = "attendance_system.db"
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
MIN_IMAGES_REQUIRED = 3
MAX_IMAGES_ALLOWED = 5

# Cloudinary Configuration
CLOUDINARY_CLOUD_NAME = os.environ.get('CLOUDINARY_CLOUD_NAME', 'your_cloud_name')
CLOUDINARY_API_KEY = os.environ.get('CLOUDINARY_API_KEY', 'your_api_key')
CLOUDINARY_API_SECRET = os.environ.get('CLOUDINARY_API_SECRET', 'your_api_secret')
CLOUDINARY_FOLDER = "face_attendance"

# Configure Cloudinary
cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET,
    secure=True
)

# Thread-safe globals
training_in_progress = False
training_status = {
    "is_training": False,
    "progress": 0,
    "total_students": 0,
    "processed_students": 0,
    "message": "",
    "error": None
}

# Cache for performance
_embeddings_cache = None
_embeddings_cache_time = 0
CACHE_TIMEOUT = 300  # 5 minutes

# Initialize directories
Path(DATASET_PATH).mkdir(exist_ok=True)
Path(TEMP_DIR).mkdir(exist_ok=True)

# ==================== CLOUDINARY FUNCTIONS ====================

def upload_to_cloudinary(image_file, roll_number, image_index, batch, dept, name):
    """Upload image to Cloudinary"""
    try:
        # Create unique public_id
        public_id = f"{CLOUDINARY_FOLDER}/{batch}/{dept}/{roll_number}_{name}/img_{image_index}"
        
        # Upload image
        result = cloudinary.uploader.upload(
            image_file,
            public_id=public_id,
            folder=f"{CLOUDINARY_FOLDER}/{batch}/{dept}/{roll_number}_{name}",
            overwrite=True,
            resource_type="image"
        )
        
        return {
            "url": result.get('secure_url'),
            "public_id": result.get('public_id'),
            "format": result.get('format'),
            "size": result.get('bytes')
        }
    except Exception as e:
        print(f"Error uploading to Cloudinary: {e}")
        raise

def delete_from_cloudinary(public_id):
    """Delete image from Cloudinary"""
    try:
        result = cloudinary.uploader.destroy(public_id)
        return result.get('result') == 'ok'
    except Exception as e:
        print(f"Error deleting from Cloudinary: {e}")
        return False

def download_from_cloudinary(url, temp_path):
    """Download image from Cloudinary URL to temp file"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        print(f"Error downloading from Cloudinary: {e}")
        return False

def get_cloudinary_images(batch=None, dept=None, roll_number=None):
    """Get all images from Cloudinary for a student or all students"""
    try:
        # Build search expression
        expression = f"{CLOUDINARY_FOLDER}"
        if batch:
            expression += f"/{batch}"
            if dept:
                expression += f"/{dept}"
                if roll_number:
                    expression += f"/{roll_number}_*"
        
        resources = cloudinary.api.resources(
            type="upload",
            prefix=expression,
            max_results=500,
            context=True
        )
        
        return resources.get('resources', [])
    except Exception as e:
        print(f"Error fetching from Cloudinary: {e}")
        return []

# ==================== DATABASE FUNCTIONS ====================

def init_database():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    # Students table - stores all registered students
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        roll_number TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        department TEXT NOT NULL,
        batch TEXT NOT NULL,
        registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_active BOOLEAN DEFAULT 1,
        image_count INTEGER DEFAULT 0,
        cloudinary_folder TEXT,
        profile_picture_url TEXT
    )
    ''')
    
    # ... rest of the function ...

def migrate_database():
    """Check and add missing columns to existing tables"""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    try:
        # Check if cloudinary_folder column exists in students table
        cursor.execute("PRAGMA table_info(students)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'cloudinary_folder' not in columns:
            print("Adding cloudinary_folder column to students table...")
            cursor.execute('''
                ALTER TABLE students 
                ADD COLUMN cloudinary_folder TEXT
            ''')
        
        # Check if student_images table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='student_images'")
        if not cursor.fetchone():
            print("Creating student_images table...")
            cursor.execute('''
            CREATE TABLE student_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER,
                image_url TEXT NOT NULL,
                public_id TEXT NOT NULL,
                image_index INTEGER,
                uploaded_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES students (id)
            )
            ''')
            
            # Create index
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_student_images_url 
            ON student_images(image_url)
            ''')
        
        conn.commit()
        print("Database migration completed successfully")
        
    except Exception as e:
        print(f"Error during database migration: {e}")
        conn.rollback()
    finally:
        conn.close()

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# ==================== HELPER FUNCTIONS ====================

def save_training_status():
    """Save training status to file"""
    try:
        with open(TRAINING_STATUS_FILE, "w") as f:
            json.dump(training_status, f)
    except Exception as e:
        print(f"Error saving training status: {e}")

def load_training_status():
    """Load training status from file"""
    global training_status
    try:
        if os.path.exists(TRAINING_STATUS_FILE):
            with open(TRAINING_STATUS_FILE, "r") as f:
                training_status = json.load(f)
    except Exception as e:
        print(f"Error loading training status: {e}")

@lru_cache(maxsize=1)
def load_embeddings_cached():
    """Load embeddings with LRU caching"""
    try:
        if os.path.exists(EMBEDDINGS_FILE):
            with open(EMBEDDINGS_FILE, "rb") as f:
                embeddings = pickle.load(f)
            print(f"Loaded {len(embeddings)} face embeddings from cache")
            return embeddings
    except Exception as e:
        print(f"Could not load embeddings: {e}")
    return []

def get_embeddings():
    """Get embeddings with time-based caching"""
    global _embeddings_cache, _embeddings_cache_time
    
    current_time = time.time()
    if (_embeddings_cache is not None and 
        (current_time - _embeddings_cache_time) < CACHE_TIMEOUT):
        return _embeddings_cache
    
    _embeddings_cache = load_embeddings_cached()
    _embeddings_cache_time = current_time
    return _embeddings_cache

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_single_image(args):
    """Process single image for embedding extraction"""
    img_path, student_info = args
    try:
        # Use faster settings for training
        embedding = DeepFace.represent(
            img_path=img_path,
            model_name="Facenet",
            enforce_detection=False,
            detector_backend="opencv",
            align=False,
            normalization="base"
        )[0]["embedding"]
        
        return {
            "roll": student_info["roll"],
            "name": student_info["name"],
            "dept": student_info["dept"],
            "batch": student_info["batch"],
            "embedding": np.array(embedding, dtype=np.float32)
        }
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

# ==================== MODIFIED ENDPOINTS ====================

@app.route("/train", methods=["POST"])
def train_model():
    """Fast training endpoint - optimized for speed"""
    global training_in_progress
    
    if training_in_progress:
        return jsonify({
            "success": False,
            "message": "Training already in progress"
        }), 400
    
    # Check if we have students in database
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM students WHERE is_active = 1')
        student_count = cursor.fetchone()[0]
        
        if student_count == 0:
            return jsonify({
                "success": False,
                "message": "No students registered"
            }), 400
    
    # Start training in background
    training_in_progress = True
    thread = threading.Thread(target=fast_train_model_thread, daemon=True)
    thread.start()
    
    return jsonify({
        "success": True,
        "message": "Training started with Cloudinary images",
        "status_endpoint": "/training_status",
        "estimated_time": f"{(student_count * 2):.1f} seconds"
    })

def fast_train_model_thread():
    """Ultra-fast training thread using Cloudinary images"""
    global training_in_progress, training_status
    
    try:
        training_status.update({
            "is_training": True,
            "progress": 0,
            "message": "Starting fast training with Cloudinary...",
            "error": None
        })
        save_training_status()
        
        # Get all active students
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT s.id, s.roll_number, s.name, s.department, s.batch,
                       GROUP_CONCAT(si.image_url) as image_urls
                FROM students s
                LEFT JOIN student_images si ON s.id = si.student_id
                WHERE s.is_active = 1
                GROUP BY s.id
            ''')
            students = cursor.fetchall()
        
        total_students = len(students)
        if total_students == 0:
            training_status.update({
                "is_training": False,
                "message": "No students found",
                "progress": 100
            })
            save_training_status()
            training_in_progress = False
            return
        
        embeddings = []
        processed_count = 0
        
        print(f"Fast training: Processing {total_students} students...")
        
        # Update status
        training_status.update({
            "total_students": total_students,
            "processed_students": 0,
            "message": f"Processing {total_students} students..."
        })
        save_training_status()
        
        # Process each student
        for idx, student in enumerate(students):
            student_info = dict(student)
            image_urls = student_info.get('image_urls', '').split(',') if student_info.get('image_urls') else []
            
            # Process each image for this student
            for img_idx, image_url in enumerate(image_urls):
                if not image_url:
                    continue
                    
                try:
                    # Create temp file
                    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                    temp_path = temp_file.name
                    temp_file.close()
                    
                    # Download image from Cloudinary
                    if download_from_cloudinary(image_url, temp_path):
                        # Extract embedding
                        embedding = DeepFace.represent(
                            img_path=temp_path,
                            model_name="Facenet",
                            enforce_detection=False,
                            detector_backend="opencv",
                            align=False,
                            normalization="base"
                        )[0]["embedding"]
                        
                        embeddings.append({
                            "roll": student_info['roll_number'],
                            "name": student_info['name'],
                            "dept": student_info['department'],
                            "batch": student_info['batch'],
                            "embedding": embedding
                        })
                    
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                        
                except Exception as e:
                    print(f"Error processing image {image_url}: {e}")
                    continue
            
            processed_count += 1
            
            # Update progress
            progress = int(((idx + 1) / total_students) * 100)
            training_status.update({
                "progress": progress,
                "processed_students": processed_count,
                "message": f"Processed {idx + 1}/{total_students} students"
            })
            save_training_status()
        
        # Save embeddings
        if embeddings:
            with open(EMBEDDINGS_FILE, "wb") as f:
                pickle.dump(embeddings, f, protocol=4)
            
            # Update cache
            global _embeddings_cache, _embeddings_cache_time
            _embeddings_cache = embeddings
            _embeddings_cache_time = time.time()
            
            training_status.update({
                "is_training": False,
                "progress": 100,
                "message": f"Training complete! Processed {len(embeddings)} embeddings",
                "processed_students": processed_count
            })
        else:
            training_status.update({
                "is_training": False,
                "error": "No valid embeddings generated",
                "message": "Training failed: No valid face embeddings found"
            })
        
        save_training_status()
        print(f"Fast training completed: {len(embeddings)} embeddings")
        
    except Exception as e:
        print(f"Training error: {e}")
        traceback.print_exc()
        training_status.update({
            "is_training": False,
            "error": str(e),
            "message": f"Training failed: {str(e)}"
        })
        save_training_status()
    finally:
        training_in_progress = False

@app.route("/register", methods=["POST"])
def register_student():
    """Fast student registration with Cloudinary storage"""
    try:
        # Get form data with defaults
        roll = request.form.get("roll", "").strip()
        name = request.form.get("name", "").strip()
        dept = request.form.get("dept", "").strip()
        batch = request.form.get("batch", "").strip()
        
        # Validate
        if not all([roll, name, dept, batch]):
            return jsonify({
                "success": False,
                "message": "All fields are required"
            }), 400
        
        # Check for existing student in database
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT roll_number FROM students WHERE roll_number = ?',
                (roll,)
            )
            existing = cursor.fetchone()
            
            if existing:
                return jsonify({
                    "success": False,
                    "message": f"Student {roll} already exists"
                }), 400
        
        # Process uploaded images
        saved_images = []
        uploaded_urls = []
        profile_picture_url = None

        for i in range(1, MAX_IMAGES_ALLOWED + 1):
            file_key = f"image_{i}"
            if file_key not in request.files:
                continue
                
            file = request.files[file_key]
            if file and file.filename and allowed_file(file.filename):
                # Upload to Cloudinary
                try:
                    upload_result = upload_to_cloudinary(
                        file, roll, i, batch, dept, name
                    )
                    
                    if upload_result:
                        saved_images.append({
                            "url": upload_result["url"],
                            "public_id": upload_result["public_id"],
                            "index": i
                        })
                        uploaded_urls.append(upload_result["url"])
                except Exception as e:
                    print(f"Error uploading image {i}: {e}")
        
        # Validate image count
        if len(saved_images) < MIN_IMAGES_REQUIRED:
            # Delete uploaded images if minimum not met
            for img in saved_images:
                delete_from_cloudinary(img["public_id"])
            
            return jsonify({
                "success": False,
                "message": f"Minimum {MIN_IMAGES_REQUIRED} images required. Got {len(saved_images)}."
            }), 400
        profile_picture_url = saved_images[0]["url"]
        # Save student to database - with backward compatibility
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Try to insert with cloudinary_folder first
            try:
                cursor.execute('''
                    INSERT INTO students 
                    (roll_number, name, department, batch, image_count, cloudinary_folder, profile_picture_url)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (roll, name, dept, batch, len(saved_images), f"{CLOUDINARY_FOLDER}/{batch}/{dept}/{roll}_{name}", profile_picture_url))
            except sqlite3.OperationalError as e:
                if "no such column" in str(e):
                    # Fallback: insert without cloudinary_folder
                    cursor.execute('''
                        INSERT INTO students 
                        (roll_number, name, department, batch, image_count)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (roll, name, dept, batch, len(saved_images)))
                else:
                    raise
            
            student_id = cursor.lastrowid
            
            # Save image URLs to student_images table if it exists
            try:
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='student_images'")
                if cursor.fetchone():
                    for img in saved_images:
                        cursor.execute('''
                            INSERT INTO student_images 
                            (student_id, image_url, public_id, image_index)
                            VALUES (?, ?, ?, ?)
                        ''', (student_id, img["url"], img["public_id"], img["index"]))
            except Exception as e:
                print(f"Warning: Could not save to student_images table: {e}")
            
            conn.commit()
        
        return jsonify({
            "success": True,
            "message": f"Registered with {len(saved_images)} images",
            "data": {
                "roll": roll,
                "name": name,
                "dept": dept,
                "batch": batch,
                "image_count": len(saved_images),
                "cloudinary_folder": f"{CLOUDINARY_FOLDER}/{batch}/{dept}/{roll}_{name}",
                "image_urls": uploaded_urls
            }
        })
        
    except Exception as e:
        print(f"Registration error: {e}")
        return jsonify({
            "success": False,
            "message": f"Registration failed: {str(e)}"
        }), 500

@app.route("/mark_attendance_auto", methods=["POST"])
def mark_attendance_auto():
    """Optimized attendance marking with Cloudinary-based recognition"""
    start_time = time.time()
    
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({
                "success": False,
                "message": "No image provided",
                "response_time": f"{(time.time() - start_time):.2f}s"
            }), 400
        
        image_file = request.files['image']
        if not image_file.filename:
            return jsonify({
                "success": False,
                "message": "Invalid image file",
                "response_time": f"{(time.time() - start_time):.2f}s"
            }), 400
        
        # Save temp file with unique name
        temp_id = int(time.time() * 1000)
        temp_path = Path(TEMP_DIR) / f"attendance_{temp_id}.jpg"
        image_file.save(str(temp_path))
        
        # Check if model is trained
        embeddings = get_embeddings()
        if not embeddings:
            temp_path.unlink(missing_ok=True)
            return jsonify({
                "success": False,
                "message": "Model not trained. Please train first.",
                "response_time": f"{(time.time() - start_time):.2f}s"
            }), 400
        
        # Face recognition with optimized settings
        print(f"Starting face recognition for {temp_path}")
        recog_start = time.time()
        
        try:
            # Create a temporary dataset folder structure for DeepFace
            temp_dataset_dir = Path(TEMP_DIR) / f"dataset_{temp_id}"
            temp_dataset_dir.mkdir(exist_ok=True, parents=True)
            
            # Get all image URLs from database
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT s.roll_number, s.name, s.department, s.batch, si.image_url
                    FROM students s
                    JOIN student_images si ON s.id = si.student_id
                    WHERE s.is_active = 1
                ''')
                
                student_images = cursor.fetchall()
                
                # Download images to temp dataset folder
                for row in student_images:
                    student_info = dict(row)
                    folder_path = temp_dataset_dir / student_info['batch'] / student_info['department'] / f"{student_info['roll_number']}_{student_info['name']}"
                    folder_path.mkdir(exist_ok=True, parents=True)
                    
                    # Create a temp file name
                    temp_img_path = folder_path / f"temp_{hashlib.md5(student_info['image_url'].encode()).hexdigest()[:8]}.jpg"
                    
                    # Download from Cloudinary
                    if download_from_cloudinary(student_info['image_url'], str(temp_img_path)):
                        print(f"Downloaded: {student_info['image_url']}")
            
            # Use DeepFace.find with temp dataset
            result = DeepFace.find(
                img_path=str(temp_path),
                db_path=str(temp_dataset_dir),
                model_name="Facenet",
                enforce_detection=False,
                detector_backend="opencv",
                distance_metric="cosine",
                align=False,
                silent=False,
                threshold=0.35,
                normalization="base"
            )
            
            # Clean up temp dataset
            shutil.rmtree(temp_dataset_dir)
            
        except Exception as e:
            print(f"DeepFace error: {e}")
            result = []
        
        recog_time = time.time() - recog_start
        print(f"Face recognition took {recog_time:.2f}s")
        
        # Clean up temp file
        temp_path.unlink(missing_ok=True)
        
        # Process result
        recognized = None
        if result and len(result) > 0 and not result[0].empty:
            # Get ALL matches for debugging
            print(f"Found {len(result[0])} potential matches")
            
            # Show top 3 matches for debugging
            for i in range(min(3, len(result[0]))):
                match = result[0].iloc[i]
                print(f"Match {i+1}: {match['identity']} - Distance: {match['distance']:.4f}")
            
            best_match = result[0].iloc[0]
            
            # Stricter confidence check
            if best_match["distance"] < 0.35:
                match_path = Path(best_match["identity"])
                try:
                    # Extract info from path
                    parts = match_path.relative_to(temp_dataset_dir).parts
                    if len(parts) >= 3:
                        batch, dept, student_folder = parts[:3]
                        try:
                            roll, name = student_folder.split("_", 1)
                        except:
                            roll, name = student_folder, "Unknown"
                        
                        # Validate roll number format
                        if not roll or roll.strip() == "":
                            print(f"ERROR: Empty roll number in path: {match_path}")
                            recognized = None
                        else:
                            recognized = {
                                "roll": roll.strip(),
                                "name": name.strip(),
                                "dept": dept.strip(),
                                "batch": batch.strip(),
                                "confidence": 1 - best_match["distance"],
                                "distance": float(best_match["distance"]),
                                "recognition_time": recog_time,
                                "match_path": str(match_path)
                            }
                            print(f"✅ Recognized: {name} (Roll: {roll}) with confidence: {1 - best_match['distance']:.2%}")
                except Exception as e:
                    print(f"Error parsing path {match_path}: {e}")
                    traceback.print_exc()
            else:
                print(f"❌ Best match distance {best_match['distance']:.4f} exceeds threshold 0.35")
        
        if not recognized:
            return jsonify({
                "success": False,
                "message": "Face not recognized or confidence too low",
                "response_time": f"{(time.time() - start_time):.2f}s"
            }), 400
        
        # ========== STRICT ONCE-PER-DAY PREVENTION ==========
        today = datetime.date.today().isoformat()
        now = datetime.datetime.now()
        
        # Check database for existing attendance
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check if already marked today in database
            cursor.execute('''
                SELECT time FROM attendance 
                WHERE roll_number = ? AND date = ?
            ''', (recognized['roll'], today))
            
            existing_attendance = cursor.fetchone()
            
            if existing_attendance:
                return jsonify({
                    "success": False,
                    "message": f"Attendance already marked for {recognized['name']} (Roll: {recognized['roll']}) today",
                    "student": recognized,
                    "duplicate_details": {
                        "type": "daily_duplicate",
                        "last_time": existing_attendance['time'],
                        "message": f"Roll {recognized['roll']} already marked today at {existing_attendance['time']}"
                    },
                    "response_time": f"{(time.time() - start_time):.2f}s"
                }), 409
            
            # Get student ID from database
            cursor.execute(
                'SELECT id FROM students WHERE roll_number = ?',
                (recognized['roll'],)
            )
            student = cursor.fetchone()
            student_id = student['id'] if student else None
            
            # Record attendance in database
            cursor.execute('''
                INSERT INTO attendance 
                (student_id, roll_number, name, department, batch, date, time, 
                 confidence_score, recognition_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                student_id,
                recognized['roll'],
                recognized['name'],
                recognized['dept'],
                recognized['batch'],
                today,
                now.strftime("%H:%M:%S"),
                recognized['confidence'],
                recog_time
            ))
            
            conn.commit()
        
        # Also update CSV file for backward compatibility
        attendance_path = Path(ATTENDANCE_FILE)
        if attendance_path.exists():
            df = pd.read_csv(ATTENDANCE_FILE)
        else:
            df = pd.DataFrame(columns=["Roll", "Name", "Dept", "Batch", "Date", "Time", "Timestamp"])
        
        # Add new entry to CSV
        new_entry = pd.DataFrame([{
            "Roll": recognized['roll'],
            "Name": recognized['name'],
            "Dept": recognized['dept'],
            "Batch": recognized['batch'],
            "Date": today,
            "Time": now.strftime("%H:%M:%S"),
            "Timestamp": now.isoformat()
        }])
        
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)
        
        print(f"✅ SUCCESS: Attendance marked for {recognized['name']} "
              f"(Roll: {recognized['roll']}) at {now.strftime('%H:%M:%S')}")
        
        total_time = time.time() - start_time
        return jsonify({
            "success": True,
            "message": f"Attendance marked successfully for {recognized['name']}",
            "student": recognized,
            "attendance": {
                "Roll": recognized['roll'],
                "Name": recognized['name'],
                "Dept": recognized['dept'],
                "Batch": recognized['batch'],
                "Date": today,
                "Time": now.strftime("%H:%M:%S")
            },
            "response_time": f"{total_time:.2f}s",
            "recognition_time": f"{recog_time:.2f}s"
        })
        
    except Exception as e:
        print(f"Attendance error: {e}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Attendance failed: {str(e)}",
            "response_time": f"{(time.time() - start_time):.2f}s"
        }), 500

@app.route("/students/<roll_number>", methods=["DELETE"])
def delete_student(roll_number):
    """Delete a specific student and all their data from Cloudinary and database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get student details before deletion
            cursor.execute(
                'SELECT id, name, department, batch, cloudinary_folder, profile_picture_url FROM students WHERE roll_number = ?',
                (roll_number,)
            )
            student = cursor.fetchone()
            
            if student is None:
                return jsonify({
                    "success": False,
                    "message": f"Student with roll number {roll_number} not found"
                }), 404
            
            student_info = dict(student)
            
            # Get all image public_ids for this student
            cursor.execute(
                'SELECT public_id FROM student_images WHERE student_id = ?',
                (student_info['id'],)
            )
            images = cursor.fetchall()
            
            # Delete images from Cloudinary
            cloudinary_deleted = 0
            for img in images:
                if delete_from_cloudinary(img['public_id']):
                    cloudinary_deleted += 1
            
            # Delete attendance records
            cursor.execute('DELETE FROM attendance WHERE student_id = ?', (student_info['id'],))
            attendance_deleted = cursor.rowcount
            
            # Delete student images from database
            cursor.execute('DELETE FROM student_images WHERE student_id = ?', (student_info['id'],))
            images_deleted = cursor.rowcount
            
            # Delete the student
            cursor.execute('DELETE FROM students WHERE id = ?', (student_info['id'],))
            student_deleted = cursor.rowcount
            
            conn.commit()
            
            return jsonify({
                "success": True,
                "message": f"Student {student_info['name']} (Roll: {roll_number}) deleted successfully",
                "details": {
                    "student_deleted": student_deleted,
                    "attendance_records_deleted": attendance_deleted,
                    "cloudinary_images_deleted": cloudinary_deleted,
                    "database_images_deleted": images_deleted,
                    "student_name": student_info['name'],
                    "roll_number": roll_number,
                    "department": student_info['department'],
                    "batch": student_info['batch']
                }
            })
            
    except Exception as e:
        print(f"Error deleting student {roll_number}: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to delete student: {str(e)}"
        }), 500

@app.route("/students/images/<roll_number>", methods=["GET"])
def get_student_images(roll_number):
    """Get all images for a specific student from Cloudinary"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get student info
            cursor.execute(
                'SELECT id, name, department, batch FROM students WHERE roll_number = ?',
                (roll_number,)
            )
            student = cursor.fetchone()
            
            if student is None:
                return jsonify({
                    "success": False,
                    "message": f"Student with roll number {roll_number} not found"
                }), 404
            
            # Get student images from database
            cursor.execute(
                'SELECT image_url, public_id, image_index FROM student_images WHERE student_id = ?',
                (student['id'],)
            )
            images = cursor.fetchall()
            
            image_list = []
            for img in images:
                image_list.append({
                    "url": img['image_url'],
                    "public_id": img['public_id'],
                    "index": img['image_index']
                })
            
            return jsonify({
                "success": True,
                "roll_number": roll_number,
                "student_name": student['name'],
                "department": student['department'],
                "batch": student['batch'],
                "total_images": len(image_list),
                "images": image_list
            })
            
    except Exception as e:
        print(f"Error getting student images: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/cloudinary/sync", methods=["POST"])
def sync_cloudinary():
    """Sync Cloudinary images with database"""
    try:
        # Get all resources from Cloudinary
        resources = cloudinary.api.resources(
            type="upload",
            prefix=CLOUDINARY_FOLDER,
            max_results=500,
            context=True
        )
        
        cloudinary_images = resources.get('resources', [])
        
        # Update database with Cloudinary images
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            for img in cloudinary_images:
                # Extract info from public_id
                public_id = img['public_id']
                parts = public_id.split('/')
                
                if len(parts) >= 4:
                    # Format: folder/batch/dept/roll_name/img_index
                    batch = parts[1]
                    dept = parts[2]
                    student_folder = parts[3]
                    
                    try:
                        roll, name = student_folder.split('_', 1)
                    except:
                        roll, name = student_folder, "Unknown"
                    
                    # Get or create student
                    cursor.execute(
                        'SELECT id FROM students WHERE roll_number = ?',
                        (roll,)
                    )
                    student = cursor.fetchone()
                    
                    if student:
                        # Check if image already exists
                        cursor.execute(
                            'SELECT id FROM student_images WHERE public_id = ?',
                            (public_id,)
                        )
                        existing = cursor.fetchone()
                        
                        if not existing:
                            # Insert image record
                            cursor.execute('''
                                INSERT INTO student_images 
                                (student_id, image_url, public_id, image_index)
                                VALUES (?, ?, ?, ?)
                            ''', (student['id'], img['secure_url'], public_id, 1))
            
            conn.commit()
        
        return jsonify({
            "success": True,
            "message": f"Synced {len(cloudinary_images)} images from Cloudinary",
            "total_images": len(cloudinary_images)
        })
        
    except Exception as e:
        print(f"Sync error: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

# ==================== EXISTING ENDPOINTS (UNCHANGED) ====================

@app.route("/attendance", methods=["GET"])
def get_attendance():
    """Get attendance with pagination (reads from database)"""
    try:
        with get_db_connection() as conn:
            # Optional pagination
            limit = request.args.get('limit', default=100, type=int)
            page = request.args.get('page', default=1, type=int)
            offset = (page - 1) * limit
            
            # Get total count
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM attendance')
            total = cursor.fetchone()[0]
            
            # Get paginated data
            cursor.execute('''
                SELECT 
                    id ,
                    roll_number as Roll,
                    name as Name,
                    department as Dept,
                    batch as Batch,
                    date as Date,
                    time as Time,
                    timestamp as Timestamp
                FROM attendance 
                ORDER BY date DESC, time DESC 
                LIMIT ? OFFSET ?
            ''', (limit, offset))
            
            rows = cursor.fetchall()
            records = [dict(row) for row in rows]
            
            return jsonify({
                "data": records,
                "total": total,
                "page": page,
                "limit": limit,
                "pages": (total + limit - 1) // limit
            })
            
    except Exception as e:
        print(f"Error getting attendance: {e}")
        return jsonify([])

@app.route("/students", methods=["GET"])
def get_students():
    """Get students with caching (reads from database)"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Optional limit
            limit = request.args.get('limit', default=100, type=int)
            
            cursor.execute('''
                SELECT 
                    roll_number as roll,
                    name as name,
                    department as dept,
                    batch as batch,
                    image_count as image_count,
                    cloudinary_folder,
                    profile_picture_url
                    
                FROM students 
                WHERE is_active = 1
                ORDER BY roll_number
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            students = []
            
            for row in rows:
                student_dict = dict(row)
                # Use cloudinary folder for reference
                students.append(student_dict)
            
            return jsonify(students)
            
    except Exception as e:
        print(f"Error getting students: {e}")
        return jsonify([])

@app.route("/training_status", methods=["GET"])
def get_training_status_route():
    """Get training status"""
    load_training_status()
    return jsonify(training_status)

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    # Check database connection
    db_status = False
    cloudinary_status = False
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT 1')
            db_status = True
    except:
        db_status = False
    
    # Check Cloudinary connection
    try:
        # Simple test call to Cloudinary
        cloudinary.api.ping()
        cloudinary_status = True
    except:
        cloudinary_status = False
    
    return jsonify({
        "status": "ok",
        "timestamp": datetime.datetime.now().isoformat(),
        "embeddings_loaded": len(get_embeddings()) > 0,
        "database_status": db_status,
        "cloudinary_status": cloudinary_status,
        "storage_type": "cloudinary"
    })

@app.route("/stats", methods=["GET"])
def get_stats():
    """Get system statistics (from database)"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Student count from database
            cursor.execute('SELECT COUNT(*) FROM students WHERE is_active = 1')
            student_count = cursor.fetchone()[0]
            
            # Total images count
            cursor.execute('SELECT SUM(image_count) as total_images FROM students WHERE is_active = 1')
            image_result = cursor.fetchone()
            image_count = image_result['total_images'] if image_result['total_images'] else 0
            
            # Attendance stats from database
            today = datetime.date.today().isoformat()
            cursor.execute('SELECT COUNT(*) FROM attendance WHERE date = ?', (today,))
            today_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM attendance')
            total_attendance = cursor.fetchone()[0]
            
            return jsonify({
                "students": student_count,
                "images": image_count,
                "embeddings": len(get_embeddings()),
                "attendance_today": today_count,
                "attendance_total": total_attendance,
                "storage": "cloudinary"
            })
            
    except Exception as e:
        print(f"Error getting stats: {e}")
        return jsonify({"error": str(e)}), 500

# ==================== OTHER EXISTING ENDPOINTS (UNCHANGED) ====================

@app.route("/db/students/detailed", methods=["GET"])
def get_detailed_students():
    """Get detailed student information from database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get query parameters
            search = request.args.get('search', '')
            dept = request.args.get('department', '')
            batch = request.args.get('batch', '')
            page = request.args.get('page', 1, type=int)
            limit = request.args.get('limit', 20, type=int)
            offset = (page - 1) * limit
            
            # Build query
            query = '''
                SELECT 
                    s.*,
                    COUNT(a.id) as total_attendance,
                    MAX(a.date) as last_attendance_date
                FROM students s
                LEFT JOIN attendance a ON s.roll_number = a.roll_number
                WHERE s.is_active = 1
            '''
            params = []
            
            if search:
                query += " AND (s.name LIKE ? OR s.roll_number LIKE ?)"
                params.extend([f'%{search}%', f'%{search}%'])
            
            if dept:
                query += " AND s.department = ?"
                params.append(dept)
            
            if batch:
                query += " AND s.batch = ?"
                params.append(batch)
            
            query += " GROUP BY s.id ORDER BY s.roll_number"
            
            # Get total count
            count_query = query.replace("SELECT s.*, COUNT(a.id) as total_attendance, MAX(a.date) as last_attendance_date", "SELECT COUNT(DISTINCT s.id)")
            count_query = count_query.replace("GROUP BY s.id ORDER BY s.roll_number", "")
            
            cursor.execute(count_query, params)
            total = cursor.fetchone()[0]
            
            # Add pagination
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            students = [dict(row) for row in rows]
            
            return jsonify({
                "success": True,
                "data": students,
                "pagination": {
                    "page": page,
                    "limit": limit,
                    "total": total,
                    "pages": (total + limit - 1) // limit
                }
            })
            
    except Exception as e:
        print(f"Error getting detailed students: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/db/attendance/report", methods=["GET"])
def get_attendance_report():
    """Get attendance report with filtering"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get query parameters
            start_date = request.args.get('start_date')
            end_date = request.args.get('end_date', datetime.date.today().isoformat())
            dept = request.args.get('department')
            batch = request.args.get('batch')
            
            # Build query
            query = '''
                SELECT 
                    date,
                    COUNT(*) as present_count,
                    GROUP_CONCAT(DISTINCT roll_number) as present_rolls
                FROM attendance
                WHERE 1=1
            '''
            params = []
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            
            if dept:
                query += " AND department = ?"
                params.append(dept)
            
            if batch:
                query += " AND batch = ?"
                params.append(batch)
            
            query += " GROUP BY date ORDER BY date"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Get total students for percentage calculation
            student_query = "SELECT COUNT(*) FROM students WHERE is_active = 1"
            student_params = []
            
            if dept:
                student_query += " AND department = ?"
                student_params.append(dept)
            
            if batch:
                student_query += " AND batch = ?"
                student_params.append(batch)
            
            cursor.execute(student_query, student_params)
            total_students = cursor.fetchone()[0]
            
            report = []
            for row in rows:
                row_dict = dict(row)
                row_dict['attendance_percentage'] = round((row_dict['present_count'] / total_students * 100), 2) if total_students > 0 else 0
                report.append(row_dict)
            
            return jsonify({
                "success": True,
                "report": report,
                "summary": {
                    "total_students": total_students,
                    "date_range": f"{start_date or 'Start'} to {end_date}"
                }
            })
            
    except Exception as e:
        print(f"Error getting attendance report: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/attendance/delete-all", methods=["DELETE"])
def delete_all_attendance():
    """Delete all attendance records from both database and CSV"""
    try:
        # Delete from database
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM attendance')
            deleted_count = cursor.rowcount
            conn.commit()
        
        # Also clear the CSV file for backward compatibility
        if os.path.exists(ATTENDANCE_FILE):
            # Create empty CSV with headers
            df = pd.DataFrame(columns=["Roll", "Name", "Dept", "Batch", "Date", "Time", "Timestamp"])
            df.to_csv(ATTENDANCE_FILE, index=False)
        
        print(f"✅ SUCCESS: Deleted {deleted_count} attendance records")
        
        return jsonify({
            "success": True,
            "message": f"Deleted {deleted_count} attendance records",
            "deleted_count": deleted_count
        })
        
    except Exception as e:
        print(f"Error deleting attendance: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to delete attendance: {str(e)}"
        }), 500

@app.route("/attendance/<int:attendance_id>", methods=["DELETE"])
def delete_attendance_record(attendance_id):
    """Delete a specific attendance record by ID"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get attendance record details before deletion
            cursor.execute('''
                SELECT roll_number, name, date, time 
                FROM attendance 
                WHERE id = ?
            ''', (attendance_id,))
            
            record = cursor.fetchone()
            
            if record is None:
                return jsonify({
                    "success": False,
                    "message": f"Attendance record with ID {attendance_id} not found"
                }), 404
            
            # Delete the record
            cursor.execute('DELETE FROM attendance WHERE id = ?', (attendance_id,))
            deleted_count = cursor.rowcount
            
            conn.commit()
            
            return jsonify({
                "success": True,
                "message": f"Attendance record deleted successfully",
                "details": {
                    "record_id": attendance_id,
                    "deleted_count": deleted_count,
                    "student_name": record['name'],
                    "roll_number": record['roll_number'],
                    "date": record['date'],
                    "time": record['time']
                }
            })
            
    except Exception as e:
        print(f"Error deleting attendance record {attendance_id}: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to delete attendance record: {str(e)}"
        }), 500

@app.route("/students/<roll_number>/attendance", methods=["DELETE"])
def delete_student_attendance(roll_number):
    """Delete all attendance records for a specific student"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check if student exists
            cursor.execute(
                'SELECT name FROM students WHERE roll_number = ?',
                (roll_number,)
            )
            student = cursor.fetchone()
            
            if student is None:
                return jsonify({
                    "success": False,
                    "message": f"Student with roll number {roll_number} not found"
                }), 404
            
            # Get count before deletion
            cursor.execute(
                'SELECT COUNT(*) as count FROM attendance WHERE roll_number = ?',
                (roll_number,)
            )
            count_result = cursor.fetchone()
            record_count = count_result['count'] if count_result else 0
            
            # Delete all attendance for this student
            cursor.execute('DELETE FROM attendance WHERE roll_number = ?', (roll_number,))
            deleted_count = cursor.rowcount
            
            conn.commit()
            
            return jsonify({
                "success": True,
                "message": f"Deleted {deleted_count} attendance records for {student['name']}",
                "details": {
                    "student_name": student['name'],
                    "roll_number": roll_number,
                    "records_deleted": deleted_count,
                    "total_records": record_count
                }
            })
            
    except Exception as e:
        print(f"Error deleting attendance for student {roll_number}: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to delete attendance records: {str(e)}"
        }), 500

# ==================== INITIALIZATION ====================

# Load initial data
load_training_status()

# Initialize database on startup
init_database()

if __name__ == "__main__":
    print("=" * 60)
    print("Enhanced Attendance System with Cloudinary Storage")
    print("=" * 60)
    print(f"Cloud Storage: Cloudinary")
    print(f"Database file: {DATABASE_FILE}")
    print(f"Loaded {len(get_embeddings())} embeddings")
    print("=" * 60)
    print("ALL EXISTING ENDPOINTS PRESERVED WITH SAME RESPONSE FORMAT")
    print("New cloud endpoints available for image management")
    print("=" * 60)
    
    # Use production server settings for better performance
    # app.run(
    #     debug=True,
    #     host='0.0.0.0',
    #     port=5000,
    #     threaded=True,
    #     processes=1
    # )
    
    app.run(debug=True)
