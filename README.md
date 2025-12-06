# ImageRevive: Multi-Agent AI Image Restoration

## Overview

ImageRevive is a professional-grade multi-agent AI framework that intelligently orchestrates specialized deep learning models to restore degraded images. Using LangGraph for agent coordination, the system performs four core restoration tasks with state-of-the-art results.

## Core Capabilities

- **Denoising**: Remove noise while preserving structural details using diffusion models
- **Super-Resolution**: Enhance low-resolution images (2x-4x) to high-quality outputs
- **Colorization**: Intelligently colorize grayscale images with realistic palettes
- **Inpainting**: Seamlessly fill missing/damaged regions with contextual content

## Applications

- Historical photograph preservation for cultural heritage digitization
- Medical imaging enhancement for diagnostic accuracy
- Forensic image analysis and enhancement
- Professional photo restoration services

## Project Structure

```
ImageRevive/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── orchestrator.py          # LangGraph orchestration
│   │   ├── denoising_agent.py       # Noise removal agent
│   │   ├── super_resolution_agent.py # SR agent
│   │   ├── colorization_agent.py    # Colorization agent
│   │   └── inpainting_agent.py      # Inpainting agent
│   ├── models/
│   │   ├── __init__.py
│   │   ├── diffusion_denoiser.py    # DDPM-based denoiser
│   │   ├── sr_transformer.py        # SwinIR-based SR
│   │   ├── colorizer.py             # Stable Diffusion colorization
│   │   └── inpainter.py             # Context-aware inpainting
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset_loader.py        # Dataset management
│   │   └── preprocessing.py         # Data preprocessing
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py               # PSNR, SSIM, LPIPS
│   │   ├── image_utils.py           # Image processing utilities
│   │   └── visualization.py         # Result visualization
│   └── app.py                       # Flask web application
├── notebooks/
│   └── eda_and_analysis.ipynb       # Exploratory data analysis
├── models/                          # Pre-trained model weights
├── data/                            # Dataset directory
│   ├── train/
│   ├── validation/
│   └── test/
├── outputs/                         # Restored images
├── config/
│   └── config.yaml                  # Configuration file
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package setup
└── README.md                        # This file
```
# ImageRevive - Quick Start Instructions

## 🚀 Getting Started in 5 Minutes

### Prerequisites Check
```bash
python --version  # Should be 3.8+
pip --version
```

### Step 1: Project Setup (2 minutes)

```bash
# Create project directory
mkdir ImageRevive
cd ImageRevive

# Download all files to this directory
# (Place all downloaded files here)

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 2: Install Dependencies (2 minutes)

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch (choose based on your system)

# For NVIDIA GPU with CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For macOS (M1/M2):
pip install torch torchvision

# Install other dependencies
pip install -r requirements.txt
```

### Step 3: Initialize Project (30 seconds)

```bash
# Run setup script
python setup.py
```

This creates all necessary directories and validates the installation.

### Step 4: Start the Application (10 seconds)

```bash
# Start Flask server
python app.py
```

You should see:
```
 * Running on http://0.0.0.0:5000
```

### Step 5: Use the Application

Open your web browser and navigate to:
```
http://localhost:5000
```

You'll see the ImageRevive web interface where you can:
1. Upload an image
2. Select restoration tasks (Denoising, Super-Resolution, Colorization, Inpainting)
3. Click "Start Restoration"
4. Download the restored image

---

## 📁 Complete File List

### Core Files
- `app.py` - Flask web application
- `orchestrator.py` - LangGraph multi-agent orchestrator
- `denoising_agent.py` - Denoising agent implementation
- `super_resolution_agent.py` - Super-resolution agent
- `colorization_agent.py` - Colorization agent
- `inpainting_agent.py` - Inpainting agent
- `metrics.py` - Quality metrics (PSNR, SSIM, LPIPS)

### Configuration
- `config.yaml` - System configuration
- `requirements.txt` - Python dependencies

### Setup & Testing
- `setup.py` - Project initialization script
- `test_all.py` - Comprehensive test suite

### Documentation
- `README.md` - Project overview
- `SETUP_GUIDE.md` - Detailed setup instructions
- `DOCUMENTATION.md` - Complete technical documentation
- `RUN_INSTRUCTIONS.md` - This file

### Web Interface
- `templates/index.html` - Web UI

---

## 🔧 Project Structure After Setup

```
ImageRevive/
├── src/
│   ├── agents/          # Agent implementations
│   ├── models/          # Model architectures
│   ├── data/            # Data loaders
│   └── utils/           # Utilities
├── models/              # Pre-trained model weights
├── data/                # Datasets
│   ├── train/
│   ├── validation/
│   └── test/
├── outputs/             # Restored images
├── uploads/             # Uploaded images
├── logs/                # Application logs
├── templates/           # HTML templates
├── config.yaml          # Configuration
├── app.py              # Main application
└── requirements.txt    # Dependencies
```

---

## 🧪 Testing the Installation

```bash
# Run comprehensive tests
python test_all.py
```

This validates:
- ✓ All modules import correctly
- ✓ CUDA/GPU availability
- ✓ Image processing capabilities
- ✓ All agents initialize properly
- ✓ Orchestrator works correctly
- ✓ Flask application runs
- ✓ File operations work
- ✓ Configuration is valid

---

## 💻 Using the API

### Command Line Example

```bash
# Upload and restore an image
curl -X POST http://localhost:5000/api/restore \
  -F "file=@path/to/image.jpg" \
  -F "tasks=denoising,super_resolution"

# Response:
# {
#   "job_id": "abc123",
#   "status": "processing"
# }

# Check status
curl http://localhost:5000/api/status/abc123

# Download result
curl -O http://localhost:5000/api/download/abc123
```

### Python Example

```python
import requests

# Upload image
files = {'file': open('image.jpg', 'rb')}
data = {'tasks': 'denoising,super_resolution'}
response = requests.post(
    'http://localhost:5000/api/restore',
    files=files,
    data=data
)

job_id = response.json()['job_id']

# Check status
status = requests.get(
    f'http://localhost:5000/api/status/{job_id}'
).json()

# Download result
result = requests.get(
    f'http://localhost:5000/api/download/{job_id}'
)
with open('restored.png', 'wb') as f:
    f.write(result.content)
```

---

## 🎯 Available Restoration Tasks

### 1. Denoising
- **Purpose**: Remove noise while preserving details
- **Use when**: Image has grain, artifacts, or noise
- **Processing time**: ~2-5 seconds
- **Task name**: `denoising`

### 2. Super-Resolution
- **Purpose**: Enhance resolution 2x-4x
- **Use when**: Image is low resolution
- **Processing time**: ~3-7 seconds
- **Task name**: `super_resolution`

### 3. Colorization
- **Purpose**: Add realistic colors to grayscale images
- **Use when**: Image is black and white
- **Processing time**: ~2-4 seconds
- **Task name**: `colorization`

### 4. Inpainting
- **Purpose**: Fill missing or damaged regions
- **Use when**: Image has holes, scratches, or missing areas
- **Processing time**: ~2-5 seconds
- **Task name**: `inpainting`

---

## ⚙️ Configuration Options

Edit `config.yaml` to customize:

```yaml
# Use GPU or CPU
system:
  device: "cuda"  # or "cpu"

# Adjust for your hardware
system:
  batch_size: 4  # Reduce if out of memory

# Web server settings
webapp:
  host: "0.0.0.0"
  port: 5000
  debug: false
```

---

## 🐛 Common Issues & Solutions

### Issue: "CUDA out of memory"
**Solution**: 
```yaml
# In config.yaml
system:
  batch_size: 2  # Reduce from 4
```

### Issue: "Module not found"
**Solution**:
```bash
pip install --force-reinstall -r requirements.txt
```

### Issue: "Port 5000 already in use"
**Solution**:
```bash
# Kill existing process
lsof -ti:5000 | xargs kill -9

# Or change port in config.yaml
webapp:
  port: 5001
```

### Issue: Slow processing
**Solution**:
```bash
# Check if GPU is being used
python -c "import torch; print(torch.cuda.is_available())"

# If False, ensure CUDA PyTorch is installed
```

---

## 📊 Performance Expectations

### Processing Time (NVIDIA GPU)
- Small images (256×256): 1-3 seconds per task
- Medium images (512×512): 3-7 seconds per task
- Large images (1024×1024): 8-15 seconds per task

### CPU Processing
- 3-5x slower than GPU
- Recommended for testing only

### Quality Metrics
- PSNR: 30-40 dB (higher is better)
- SSIM: 0.85-0.95 (higher is better)
- Quality Score: 0.80-0.95 (higher is better)

---

## 🔄 Updating the System

```bash
# Update dependencies
pip install -r requirements.txt --upgrade

# Update configuration
cp config.yaml config.yaml.backup
# Edit config.yaml as needed
```

---

## 📚 Additional Resources

- **README.md**: Project overview and features
- **SETUP_GUIDE.md**: Detailed installation guide
- **DOCUMENTATION.md**: Complete technical documentation
- **Logs**: Check `logs/imagerevive.log` for errors

---

## 🎓 Training Your Own Models (Advanced)

```bash
# Download datasets
python scripts/download_datasets.py --dataset DIV2K

# Train super-resolution model
python training/train_sr.py \
  --data ./data \
  --epochs 100 \
  --batch-size 16

# Monitor training
tensorboard --logdir logs/tensorboard
```

---

## 🚢 Production Deployment

### Using Gunicorn
```bash
gunicorn -w 4 -b 0.0.0.0:5000 \
         --timeout 300 \
         app:app
```

### Using Docker
```bash
docker build -t imagerevive .
docker run -p 5000:5000 imagerevive
```

---

## 📞 Support

If you encounter issues:

1. Check `logs/imagerevive.log` for error messages
2. Run `python test_all.py` to diagnose problems
3. Verify all files are present and properly placed
4. Ensure Python 3.8+ is installed
5. Check that all dependencies installed correctly

---

## ✅ Verification Checklist

Before using the application:

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Project structure created (`python setup.py`)
- [ ] Tests pass (`python test_all.py`)
- [ ] Application starts (`python app.py`)
- [ ] Web interface accessible at http://localhost:5000

---

## 🎉 You're Ready!

Your ImageRevive installation is complete. Start restoring images by:

1. Opening http://localhost:5000 in your browser
2. Uploading an image
3. Selecting restoration tasks
4. Clicking "Start Restoration"

For advanced usage, API integration, or training custom models, see DOCUMENTATION.md.

**Happy Image Restoring! 🖼️✨**

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for optimal performance)
- 16GB+ RAM
- 10GB+ disk space for models and datasets

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd ImageRevive
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Download Pre-trained Models

```bash
python scripts/download_models.py
```

### Step 5: Prepare Datasets (Optional for Training)

```bash
python scripts/prepare_datasets.py --dataset DIV2K --output ./data
```

## Quick Start

### Web Interface

1. Start the Flask server:
```bash
python src/app.py
```

2. Open browser to `http://localhost:5000`

3. Upload an image and select restoration tasks

4. Click "Restore" and download results

### Command Line Interface

```bash
python src/cli.py --input image.jpg --tasks denoising,super_resolution --output restored.png
```

## Configuration

Edit `config/config.yaml` to customize:

- Model parameters
- Processing settings
- Agent orchestration logic
- Performance optimization

## Datasets

### Supported Datasets

1. **DIV2K**: Primary training dataset for super-resolution
2. **FFHQ**: Face-specific fine-tuning
3. **ImageNet**: Colorization training
4. **Urban100**: Evaluation benchmark

### Dataset Preparation

```bash
# Download DIV2K
python scripts/download_datasets.py --dataset DIV2K

# Preprocess data
python scripts/preprocess_data.py --input ./data/raw --output ./data/processed
```

## Training

### Train Individual Models

```bash
# Train denoising model
python src/training/train_denoiser.py --config config/denoiser_config.yaml

# Train super-resolution model
python src/training/train_sr.py --config config/sr_config.yaml

# Train colorization model
python src/training/train_colorizer.py --config config/colorizer_config.yaml

# Train inpainting model
python src/training/train_inpainter.py --config config/inpainter_config.yaml
```

### Fine-tuning

```bash
python src/training/finetune.py --model sr_model --dataset FFHQ --epochs 50
```

## Evaluation

### Performance Metrics

- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)
- **LPIPS** (Learned Perceptual Image Patch Similarity)
- **FID** (Fréchet Inception Distance)

### Run Evaluation

```bash
python src/evaluation/evaluate.py --test_dir ./data/test --output ./results/metrics.json
```

## Architecture

### Multi-Agent Orchestration

ImageRevive uses LangGraph to coordinate specialized agents:

1. **Task Analyzer**: Analyzes input image and determines required restoration tasks
2. **Denoising Agent**: Removes noise using diffusion models
3. **Super-Resolution Agent**: Enhances resolution using transformer-based networks
4. **Colorization Agent**: Adds realistic colors to grayscale images
5. **Inpainting Agent**: Fills missing regions with contextual content
6. **Quality Validator**: Ensures output quality meets standards

### Processing Pipeline

```
Input Image → Task Analysis → Agent Selection → Sequential Processing → Quality Validation → Output
```

## Performance Optimization

- **Model Quantization**: INT8 quantization for faster inference
- **Batch Processing**: Process multiple images simultaneously
- **GPU Acceleration**: CUDA optimization for neural networks
- **Caching**: Result caching to minimize redundant computations

## API Documentation

### REST API Endpoints

- `POST /api/restore`: Submit image for restoration
- `GET /api/status/<job_id>`: Check processing status
- `GET /api/download/<job_id>`: Download restored image
- `GET /api/models`: List available models
- `POST /api/batch`: Batch processing endpoint

### Python API

```python
from src.agents.orchestrator import ImageRestoreOrchestrator

orchestrator = ImageRestoreOrchestrator()
result = orchestrator.restore(
    image_path="input.jpg",
    tasks=["denoising", "super_resolution"],
    output_path="output.png"
)
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or image resolution
2. **Slow Processing**: Enable GPU acceleration or reduce model complexity
3. **Poor Quality**: Adjust model parameters or try different task combinations

### Logs

Check `logs/imagerevive.log` for detailed execution logs.

## Contributing

Contributions are welcome! Please follow PEP 8 style guidelines.

## License

MIT License - See LICENSE file for details

## Citation

If you use ImageRevive in your research, please cite:

```bibtex
@software{imagerevive2024,
  title={ImageRevive: Multi-Agent AI Image Restoration Framework},
  author={},
  year={2024},
  url={https://github.com/yourusername/imagerevive}
}
```

## Acknowledgments

- Inspired by research from 4KAgent and AgenticIR projects
- Built with PyTorch, LangChain, and LangGraph
- Utilizes state-of-the-art models: DDPM, SwinIR, Stable Diffusion, GFPGAN

## Contact

For questions and support, please open an issue on GitHub.
