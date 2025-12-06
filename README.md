# ImageRevive: Multi-Agent AI Image Restoration Framework

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

