#!/usr/bin/env python3
"""
Comprehensive Testing Script for ImageRevive
Tests all components and validates functionality
"""

import sys
import time
import logging
from pathlib import Path
import numpy as np
from PIL import Image
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_test_header(test_name):
    """Print formatted test header."""
    print("\n" + "=" * 70)
    print(f"  TEST: {test_name}")
    print("=" * 70)


def test_imports():
    """Test if all required modules can be imported."""
    print_test_header("Module Imports")
    
    modules = [
        'torch',
        'torchvision',
        'numpy',
        'PIL',
        'flask',
        'yaml',
        'scipy',
        'langchain',
        'langgraph'
    ]
    
    results = {}
    for module in modules:
        try:
            __import__(module)
            logger.info(f"✓ {module} imported successfully")
            results[module] = True
        except ImportError as e:
            logger.error(f"✗ {module} import failed: {str(e)}")
            results[module] = False
    
    success = all(results.values())
    logger.info(f"\nImport Test: {'PASSED' if success else 'FAILED'}")
    return success


def test_device_availability():
    """Test CUDA/device availability."""
    print_test_header("Device Availability")
    
    try:
        import torch
        
        # Check CUDA
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA Available: {cuda_available}")
        
        if cuda_available:
            logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
        
        # Check MPS (Apple Silicon)
        mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        logger.info(f"MPS Available: {mps_available}")
        
        # Set device
        if cuda_available:
            device = torch.device('cuda')
        elif mps_available:
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        
        logger.info(f"Using device: {device}")
        
        return True
        
    except Exception as e:
        logger.error(f"Device test failed: {str(e)}")
        return False


def test_image_processing():
    """Test basic image processing capabilities."""
    print_test_header("Image Processing")
    
    try:
        # Create test image
        test_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        logger.info(f"Created test image: shape={test_image.shape}")
        
        # Test PIL conversion
        pil_image = Image.fromarray(test_image)
        logger.info(f"✓ PIL conversion successful")
        
        # Test numpy conversion
        np_image = np.array(pil_image)
        assert np_image.shape == test_image.shape
        logger.info(f"✓ NumPy conversion successful")
        
        # Test torch conversion
        tensor = torch.from_numpy(test_image).permute(2, 0, 1).float() / 255.0
        assert tensor.shape == (3, 256, 256)
        logger.info(f"✓ Torch conversion successful")
        
        logger.info("\nImage Processing Test: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"Image processing test failed: {str(e)}")
        return False


def test_metrics():
    """Test metrics computation."""
    print_test_header("Metrics Computation")
    
    try:
        from metrics import compute_psnr, compute_ssim, compute_quality_score
        
        # Create test images
        image1 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        image2 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        
        # Test PSNR
        psnr = compute_psnr(image1, image2)
        logger.info(f"✓ PSNR computed: {psnr:.2f} dB")
        
        # Test SSIM
        ssim = compute_ssim(image1, image2)
        logger.info(f"✓ SSIM computed: {ssim:.4f}")
        
        # Test quality score
        quality = compute_quality_score(image1, image2)
        logger.info(f"✓ Quality score computed: {quality:.4f}")
        
        logger.info("\nMetrics Test: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"Metrics test failed: {str(e)}")
        return False


def test_agents():
    """Test individual agents."""
    print_test_header("Agent Functionality")
    
    try:
        import yaml
        from denoising_agent import DenoisingAgent
        from super_resolution_agent import SuperResolutionAgent
        from colorization_agent import ColorizationAgent
        from inpainting_agent import InpaintingAgent
        
        # Load config
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test image
        test_image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        
        # Test Denoising Agent
        logger.info("Testing Denoising Agent...")
        denoiser = DenoisingAgent(config['models']['denoising'], device)
        logger.info("✓ Denoising Agent initialized")
        
        # Test Super-Resolution Agent
        logger.info("Testing Super-Resolution Agent...")
        sr_agent = SuperResolutionAgent(config['models']['super_resolution'], device)
        logger.info("✓ Super-Resolution Agent initialized")
        
        # Test Colorization Agent
        logger.info("Testing Colorization Agent...")
        colorizer = ColorizationAgent(config['models']['colorization'], device)
        logger.info("✓ Colorization Agent initialized")
        
        # Test Inpainting Agent
        logger.info("Testing Inpainting Agent...")
        inpainter = InpaintingAgent(config['models']['inpainting'], device)
        logger.info("✓ Inpainting Agent initialized")
        
        logger.info("\nAgent Test: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"Agent test failed: {str(e)}")
        return False


def test_orchestrator():
    """Test orchestrator functionality."""
    print_test_header("Orchestrator")
    
    try:
        import yaml
        from orchestrator import ImageRestoreOrchestrator
        
        # Load config
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize orchestrator
        logger.info("Initializing orchestrator...")
        orchestrator = ImageRestoreOrchestrator(config)
        logger.info("✓ Orchestrator initialized")
        
        # Create test image
        test_image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        
        # Test restoration (with minimal tasks for speed)
        logger.info("Testing restoration workflow...")
        result = orchestrator.restore(
            image=test_image,
            tasks=['denoising']
        )
        
        if result['success']:
            logger.info("✓ Restoration completed")
            logger.info(f"  Quality score: {result['quality_score']:.4f}")
            logger.info(f"  Completed tasks: {result['completed_tasks']}")
        else:
            logger.error(f"✗ Restoration failed: {result['error']}")
            return False
        
        logger.info("\nOrchestrator Test: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"Orchestrator test failed: {str(e)}")
        return False


def test_flask_app():
    """Test Flask application."""
    print_test_header("Flask Application")
    
    try:
        from app import app
        
        # Test app creation
        logger.info("✓ Flask app created")
        
        # Test client
        client = app.test_client()
        
        # Test health endpoint
        response = client.get('/health')
        assert response.status_code == 200
        logger.info("✓ Health endpoint working")
        
        # Test main page
        response = client.get('/')
        assert response.status_code == 200
        logger.info("✓ Main page accessible")
        
        # Test models endpoint
        response = client.get('/api/models')
        assert response.status_code == 200
        logger.info("✓ Models endpoint working")
        
        logger.info("\nFlask App Test: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"Flask app test failed: {str(e)}")
        return False


def test_file_operations():
    """Test file I/O operations."""
    print_test_header("File Operations")
    
    try:
        # Test directory creation
        test_dir = Path('test_outputs')
        test_dir.mkdir(exist_ok=True)
        logger.info("✓ Directory creation successful")
        
        # Test image save/load
        test_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        test_path = test_dir / 'test_image.png'
        
        Image.fromarray(test_image).save(test_path)
        logger.info("✓ Image save successful")
        
        loaded_image = np.array(Image.open(test_path))
        assert loaded_image.shape == test_image.shape
        logger.info("✓ Image load successful")
        
        # Cleanup
        test_path.unlink()
        test_dir.rmdir()
        logger.info("✓ Cleanup successful")
        
        logger.info("\nFile Operations Test: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"File operations test failed: {str(e)}")
        return False


def test_configuration():
    """Test configuration loading."""
    print_test_header("Configuration")
    
    try:
        import yaml
        
        # Load config
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info("✓ Configuration loaded")
        
        # Validate required sections
        required_sections = ['system', 'models', 'orchestration', 'webapp']
        for section in required_sections:
            assert section in config, f"Missing section: {section}"
            logger.info(f"✓ Section '{section}' present")
        
        logger.info("\nConfiguration Test: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"Configuration test failed: {str(e)}")
        return False


def run_all_tests():
    """Run all tests and generate report."""
    print("\n" + "=" * 70)
    print("  ImageRevive - Comprehensive Test Suite")
    print("=" * 70)
    
    tests = [
        ("Imports", test_imports),
        ("Device Availability", test_device_availability),
        ("Image Processing", test_image_processing),
        ("Configuration", test_configuration),
        ("Metrics", test_metrics),
        ("File Operations", test_file_operations),
        ("Agents", test_agents),
        ("Orchestrator", test_orchestrator),
        ("Flask Application", test_flask_app)
    ]
    
    results = {}
    start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Unexpected error in {test_name}: {str(e)}")
            results[test_name] = False
    
    elapsed_time = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 70)
    print("  TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:.<50} {status}")
    
    print("\n" + "-" * 70)
    print(f"Total: {passed}/{total} tests passed")
    print(f"Time: {elapsed_time:.2f} seconds")
    print("=" * 70 + "\n")
    
    return all(results.values())


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
