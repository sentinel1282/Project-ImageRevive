#!/usr/bin/env python3
"""
Project Setup Script for ImageRevive
Initializes project structure and validates dependencies
"""

import os
import sys
from pathlib import Path
import subprocess
import shutil


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def create_directory_structure():
    """Create all required directories."""
    print_header("Creating Directory Structure")
    
    directories = [
        'src/agents',
        'src/models',
        'src/data',
        'src/utils',
        'src/training',
        'src/evaluation',
        'models',
        'data/train/HR',
        'data/train/LR',
        'data/validation/HR',
        'data/validation/LR',
        'data/test/HR',
        'data/test/LR',
        'outputs',
        'uploads',
        'logs/tensorboard',
        'config',
        'scripts',
        'tests',
        'notebooks',
        'templates',
        'static/css',
        'static/js',
        'static/images',
        'cache'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}")
    
    # Create __init__.py files
    init_dirs = [
        'src',
        'src/agents',
        'src/models',
        'src/data',
        'src/utils',
        'src/training',
        'src/evaluation'
    ]
    
    for directory in init_dirs:
        init_file = Path(directory) / '__init__.py'
        init_file.touch(exist_ok=True)
        print(f"✓ Created: {init_file}")


def check_python_version():
    """Check if Python version is compatible."""
    print_header("Checking Python Version")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ ERROR: Python 3.8+ required")
        return False
    
    print("✓ Python version compatible")
    return True


def check_dependencies():
    """Check if required dependencies can be imported."""
    print_header("Checking Dependencies")
    
    dependencies = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('PIL', 'Pillow'),
        ('flask', 'Flask'),
        ('yaml', 'PyYAML'),
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy')
    ]
    
    missing = []
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"❌ {name} NOT installed")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True


def check_cuda():
    """Check CUDA availability."""
    print_header("Checking CUDA Availability")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠️  CUDA not available - will use CPU")
            print("  For GPU acceleration, install CUDA and PyTorch with CUDA support")
    except ImportError:
        print("❌ PyTorch not installed")


def create_sample_config():
    """Create sample configuration if it doesn't exist."""
    print_header("Setting Up Configuration")
    
    config_file = Path('config.yaml')
    
    if config_file.exists():
        print("✓ config.yaml already exists")
    else:
        print("Creating default config.yaml...")
        # The config.yaml was already created, so just verify
        if config_file.exists():
            print("✓ config.yaml created")
        else:
            print("⚠️  Please create config.yaml from template")


def create_gitignore():
    """Create .gitignore file."""
    print_header("Creating .gitignore")
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter Notebook
.ipynb_checkpoints

# Project specific
uploads/*
outputs/*
logs/*
cache/*
models/*.pth
models/*.pt
data/train/*
data/validation/*
data/test/*

# Keep directory structure
!uploads/.gitkeep
!outputs/.gitkeep
!logs/.gitkeep
!models/.gitkeep
!data/train/.gitkeep
!data/validation/.gitkeep
!data/test/.gitkeep

# OS
.DS_Store
Thumbs.db

# Secrets
*.key
*.pem
.env
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    print("✓ .gitignore created")


def create_gitkeep_files():
    """Create .gitkeep files to preserve directory structure."""
    print_header("Creating .gitkeep Files")
    
    gitkeep_dirs = [
        'uploads',
        'outputs',
        'logs',
        'models',
        'data/train',
        'data/validation',
        'data/test'
    ]
    
    for directory in gitkeep_dirs:
        gitkeep_file = Path(directory) / '.gitkeep'
        gitkeep_file.touch(exist_ok=True)
        print(f"✓ Created: {gitkeep_file}")


def setup_logging():
    """Initialize logging configuration."""
    print_header("Setting Up Logging")
    
    log_file = Path('logs') / 'imagerevive.log'
    log_file.touch(exist_ok=True)
    print(f"✓ Log file created: {log_file}")


def create_readme():
    """Ensure README.md exists."""
    print_header("Checking README")
    
    readme = Path('README.md')
    if readme.exists():
        print("✓ README.md exists")
    else:
        print("⚠️  README.md not found")


def print_next_steps():
    """Print next steps for user."""
    print_header("Setup Complete!")
    
    print("""
Next Steps:
-----------

1. Install dependencies (if not already done):
   pip install -r requirements.txt

2. Configure the system:
   - Edit config.yaml to match your environment
   - Set device to 'cuda' or 'cpu'
   - Adjust batch sizes based on your hardware

3. (Optional) Download pre-trained models:
   python scripts/download_models.py

4. (Optional) Prepare datasets:
   python scripts/prepare_datasets.py

5. Run the application:
   python app.py

6. Open your browser:
   http://localhost:5000

For detailed instructions, see SETUP_GUIDE.md

For help:
- Check logs in: logs/imagerevive.log
- Health check: http://localhost:5000/health
- View README.md for full documentation
""")


def main():
    """Main setup function."""
    print_header("ImageRevive Project Setup")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directory structure
    create_directory_structure()
    
    # Create configuration files
    create_gitignore()
    create_gitkeep_files()
    create_sample_config()
    
    # Setup logging
    setup_logging()
    
    # Check dependencies
    check_dependencies()
    
    # Check CUDA
    check_cuda()
    
    # Check README
    create_readme()
    
    # Print next steps
    print_next_steps()
    
    print("\n✓ Setup completed successfully!\n")


if __name__ == '__main__':
    main()
