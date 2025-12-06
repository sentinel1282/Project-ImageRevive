"""
ImageRevive Flask Application
Complete version with resolution selection support
"""

import os
import uuid
import logging
from datetime import datetime
from threading import Thread
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import yaml

from orchestrator import ImageRestoreOrchestrator

# Setup
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = './uploads'
OUTPUT_FOLDER = './outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Job tracking
jobs = {}

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load config
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    logger.warning("config.yaml not found, using defaults")
    config = {
        'system': {'device': 'cpu'},
        'models': {
            'denoising': {},
            'super_resolution': {'target_resolution': '8K', 'quality_mode': 'ultra'},
            'colorization': {},
            'inpainting': {}
        },
        'orchestration': {
            'task_priority': ['denoising', 'super_resolution', 'colorization', 'inpainting'],
            'quality_threshold': 0.75
        }
    }

# Initialize orchestrator
orchestrator = ImageRestoreOrchestrator(config)

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Serve main page."""
    return render_template('index.html')


@app.route('/api/restore', methods=['POST'])
def restore_image():
    """Handle image restoration request with resolution selection."""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Get restoration tasks
        tasks_str = request.form.get('tasks', '')
        if not tasks_str:
            return jsonify({'error': 'No tasks specified'}), 400
        
        tasks = [t.strip() for t in tasks_str.split(',')]
        
        # Get resolution settings
        resolution = request.form.get('resolution', '8K')
        custom_width = request.form.get('custom_width')
        custom_height = request.form.get('custom_height')
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        upload_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_{filename}")
        file.save(upload_path)
        
        # Load image
        image_pil = Image.open(upload_path)
        image_array = np.array(image_pil)
        
        # Update orchestrator config with resolution settings
        if 'super_resolution' in tasks:
            orchestrator.sr_agent.config['target_resolution'] = resolution
            
            if resolution == 'custom':
                if custom_width:
                    orchestrator.sr_agent.config['custom_width'] = int(custom_width)
                if custom_height:
                    orchestrator.sr_agent.config['custom_height'] = int(custom_height)
        
        # Create job tracking
        jobs[job_id] = {
            'status': 'processing',
            'progress': 0,
            'input_path': upload_path,
            'tasks': tasks,
            'resolution': resolution,
            'custom_width': custom_width,
            'custom_height': custom_height,
            'created_at': datetime.now()
        }
        
        # Start processing in background thread
        thread = Thread(target=process_image_job, args=(job_id, image_array, tasks))
        thread.daemon = True
        thread.start()
        
        logger.info(f"Job {job_id} started - Tasks: {tasks}, Resolution: {resolution}")
        
        return jsonify({
            'job_id': job_id,
            'message': 'Processing started',
            'tasks': tasks,
            'resolution': resolution
        })
        
    except Exception as e:
        logger.error(f"Error in restore endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def process_image_job(job_id: str, image: np.ndarray, tasks: list):
    """Process image restoration job in background."""
    try:
        logger.info(f"Processing job {job_id}")
        jobs[job_id]['progress'] = 10
        
        # Restore image
        result = orchestrator.restore(image, tasks)
        
        jobs[job_id]['progress'] = 90
        
        if result['success']:
            # Save result
            output_path = os.path.join(OUTPUT_FOLDER, f"{job_id}_restored.png")
            output_image = Image.fromarray(result['image'])
            output_image.save(output_path, 'PNG', compress_level=1)
            
            jobs[job_id].update({
                'status': 'completed',
                'progress': 100,
                'output_path': output_path,
                'quality_score': result['quality_score'],
                'output_dimensions': f"{result['image'].shape[1]}x{result['image'].shape[0]}"
            })
            
            logger.info(f"Job {job_id} completed - Output: {result['image'].shape[1]}x{result['image'].shape[0]}")
        else:
            jobs[job_id].update({
                'status': 'failed',
                'error': result.get('error', 'Unknown error'),
                'progress': 0
            })
            logger.error(f"Job {job_id} failed: {result.get('error')}")
            
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        jobs[job_id].update({
            'status': 'failed',
            'error': str(e),
            'progress': 0
        })


@app.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """Get job status."""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    return jsonify({
        'status': job['status'],
        'progress': job.get('progress', 0),
        'quality_score': job.get('quality_score'),
        'error': job.get('error'),
        'output_dimensions': job.get('output_dimensions')
    })


@app.route('/api/download/<job_id>', methods=['GET'])
def download_result(job_id):
    """Download restored image."""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    
    if job['status'] != 'completed':
        return jsonify({'error': 'Job not completed'}), 400
    
    output_path = job.get('output_path')
    if not output_path or not os.path.exists(output_path):
        return jsonify({'error': 'Output file not found'}), 404
    
    return send_file(
        output_path,
        mimetype='image/png',
        as_attachment=True,
        download_name=f'restored_{job_id}.png'
    )


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'orchestrator': orchestrator is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    return jsonify({'error': 'File too large. Maximum size is 20MB'}), 413


@app.errorhandler(500)
def internal_error(error):
    """Handle internal server error."""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    logger.info("Starting ImageRevive Flask Application")
    logger.info(f"Upload folder: {UPLOAD_FOLDER}")
    logger.info(f"Output folder: {OUTPUT_FOLDER}")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )
