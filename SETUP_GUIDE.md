# ImageRevive - Complete Setup and Installation Guide

## System Requirements

### Minimum Requirements
- **OS**: Ubuntu 20.04+ / macOS 11+ / Windows 10+
- **Python**: 3.8 or higher
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space
- **GPU**: Optional but recommended (CUDA-capable)

### Recommended Requirements
- **OS**: Ubuntu 22.04 LTS
- **Python**: 3.10
- **RAM**: 16GB+
- **Storage**: 20GB+ SSD
- **GPU**: NVIDIA GPU with 8GB+ VRAM, CUDA 11.8+

## Installation Steps

### Step 1: Environment Setup

```bash
# Clone or download the project
cd ImageRevive

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch (choose appropriate version for your system)
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For macOS (M1/M2):
pip install torch torchvision

# Install other dependencies
pip install -r requirements.txt
```

### Step 3: Create Required Directories

```bash
mkdir -p models data/{train,validation,test} outputs uploads logs
```

### Step 4: Download Pre-trained Models (Optional)

Due to size constraints, pre-trained models need to be trained or downloaded separately.

```bash
# Create a download script
python scripts/setup_models.py
```

### Step 5: Configure the System

Edit `config.yaml` to match your system configuration:

```yaml
system:
  device: "cuda"  # Change to "cpu" if no GPU
  batch_size: 4   # Reduce if you have less RAM/VRAM
```

### Step 6: Test Installation

```bash
# Test basic imports
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torchvision; print(f'TorchVision version: {torchvision.__version__}')"
python -c "from langgraph.graph import StateGraph; print('LangGraph OK')"

# Run health check
python -c "from app import app; print('Flask app OK')"
```

## Running the Application

### Method 1: Development Server (Quick Start)

```bash
# Activate virtual environment
source venv/bin/activate

# Run Flask development server
python app.py
```

The application will be available at: `http://localhost:5000`

### Method 2: Production Server (Gunicorn)

```bash
# Install gunicorn (if not already installed)
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Method 3: Docker (Recommended for Production)

```bash
# Build Docker image
docker build -t imagerevive:latest .

# Run container
docker run -p 5000:5000 -v $(pwd)/outputs:/app/outputs imagerevive:latest
```

## Dataset Preparation

### Downloading Datasets

```bash
# Create dataset download script
python scripts/download_datasets.py --dataset DIV2K --output ./data
python scripts/download_datasets.py --dataset FFHQ --output ./data
```

### Dataset Structure

```
data/
├── train/
│   ├── HR/  # High-resolution images
│   └── LR/  # Low-resolution images
├── validation/
│   ├── HR/
│   └── LR/
└── test/
    ├── HR/
    └── LR/
```

## Training Models

### Training Individual Components

```bash
# Train denoising model
python training/train_denoiser.py \
    --config config.yaml \
    --data ./data/train \
    --epochs 100 \
    --batch-size 16

# Train super-resolution model
python training/train_sr.py \
    --config config.yaml \
    --data ./data/train \
    --epochs 100 \
    --scale 4

# Train colorization model
python training/train_colorizer.py \
    --config config.yaml \
    --data ./data/train \
    --epochs 100

# Train inpainting model
python training/train_inpainter.py \
    --config config.yaml \
    --data ./data/train \
    --epochs 100
```

### Monitor Training

```bash
# Using TensorBoard
tensorboard --logdir ./logs/tensorboard

# Using Weights & Biases (if configured)
# Training will automatically log to W&B
```

## API Usage

### REST API Endpoints

#### 1. Restore Single Image

```bash
curl -X POST http://localhost:5000/api/restore \
  -F "file=@input.jpg" \
  -F "tasks=denoising,super_resolution"
```

Response:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "tasks": ["denoising", "super_resolution"]
}
```

#### 2. Check Job Status

```bash
curl http://localhost:5000/api/status/550e8400-e29b-41d4-a716-446655440000
```

Response:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress": 100,
  "created_at": "2024-12-06T10:30:00",
  "completed_at": "2024-12-06T10:31:30"
}
```

#### 3. Download Result

```bash
curl -O http://localhost:5000/api/download/550e8400-e29b-41d4-a716-446655440000
```

#### 4. Batch Processing

```bash
curl -X POST http://localhost:5000/api/batch \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg" \
  -F "tasks=denoising,super_resolution"
```

### Python API

```python
from orchestrator import ImageRestoreOrchestrator
import numpy as np
from PIL import Image
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize orchestrator
orchestrator = ImageRestoreOrchestrator(config)

# Load image
image = np.array(Image.open('input.jpg'))

# Perform restoration
result = orchestrator.restore(
    image=image,
    tasks=['denoising', 'super_resolution']
)

# Save result
if result['success']:
    restored = Image.fromarray(result['image'])
    restored.save('output.png')
    print(f"Quality score: {result['quality_score']:.4f}")
else:
    print(f"Error: {result['error']}")
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Solution**: Reduce batch size in config.yaml

```yaml
system:
  batch_size: 2  # Reduce from 4 to 2
```

#### 2. Import Errors

**Solution**: Reinstall dependencies

```bash
pip install --force-reinstall -r requirements.txt
```

#### 3. Slow Processing

**Solution**: Enable GPU acceleration or reduce image size

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

#### 4. Port Already in Use

**Solution**: Change port in config.yaml or kill existing process

```bash
# Find process using port 5000
lsof -ti:5000 | xargs kill -9

# Or use different port
python app.py --port 5001
```

### Logging and Debugging

```bash
# View application logs
tail -f logs/imagerevive.log

# Enable debug mode
export FLASK_ENV=development
python app.py
```

## Performance Optimization

### 1. Model Quantization

```python
# Enable INT8 quantization for faster inference
config['system']['precision'] = 'int8'
```

### 2. Batch Processing

```bash
# Process multiple images in parallel
python scripts/batch_process.py \
    --input ./images \
    --output ./results \
    --tasks denoising,super_resolution \
    --batch-size 8
```

### 3. GPU Optimization

```yaml
# In config.yaml
optimization:
  use_amp: true
  compile_models: true
  channels_last: true
  benchmark_cudnn: true
```

## Security Considerations

### Production Deployment

1. **Set strong secret keys**
```python
app.config['SECRET_KEY'] = 'your-secret-key-here'
```

2. **Enable HTTPS**
```bash
gunicorn --certfile=cert.pem --keyfile=key.pem app:app
```

3. **Rate limiting**
```python
from flask_limiter import Limiter
limiter = Limiter(app, key_func=get_remote_address)
```

4. **Input validation**
- Already implemented in the application
- File size limits: 10MB
- Allowed extensions: JPG, PNG, BMP, TIFF

## Updating the System

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Update models
python scripts/download_models.py --update
```

## Backup and Recovery

```bash
# Backup models
tar -czf models_backup.tar.gz models/

# Backup configuration
cp config.yaml config.yaml.backup

# Restore models
tar -xzf models_backup.tar.gz
```

## Support and Resources

- Documentation: See README.md
- Issues: Open an issue on GitHub
- Logs: Check `logs/imagerevive.log`
- Health check: `http://localhost:5000/health`

## Next Steps

1. Train models on your dataset (optional)
2. Configure settings in config.yaml
3. Test with sample images
4. Deploy to production
5. Monitor performance and quality metrics

For detailed information on each component, refer to the respective documentation files in the `docs/` directory.
