"""
Metrics Module - Fixed Version
Handles images of different sizes for super-resolution
"""

import logging
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def resize_to_match(image1: np.ndarray, image2: np.ndarray) -> tuple:
    """
    Resize images to match if they have different sizes.
    
    Args:
        image1: First image
        image2: Second image
        
    Returns:
        Tuple of resized images with matching dimensions
    """
    if image1.shape != image2.shape:
        # Use the larger size
        target_h = max(image1.shape[0], image2.shape[0])
        target_w = max(image1.shape[1], image2.shape[1])
        
        from PIL import Image as PILImage
        
        # Resize image1 if needed
        if image1.shape[:2] != (target_h, target_w):
            pil_img1 = PILImage.fromarray(image1)
            pil_img1 = pil_img1.resize((target_w, target_h), PILImage.BICUBIC)
            image1 = np.array(pil_img1)
        
        # Resize image2 if needed
        if image2.shape[:2] != (target_h, target_w):
            pil_img2 = PILImage.fromarray(image2)
            pil_img2 = pil_img2.resize((target_w, target_h), PILImage.BICUBIC)
            image2 = np.array(pil_img2)
    
    return image1, image2


def compute_psnr(
    original: np.ndarray,
    restored: np.ndarray,
    max_value: float = 255.0
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        original: Original image [H, W, C]
        restored: Restored image [H, W, C]
        max_value: Maximum possible pixel value
        
    Returns:
        PSNR in dB
    """
    try:
        # Resize if shapes don't match
        if original.shape != restored.shape:
            logger.warning(f"Shape mismatch: {original.shape} vs {restored.shape}, resizing...")
            original, restored = resize_to_match(original, restored)
        
        # Convert to float
        original = original.astype(np.float64)
        restored = restored.astype(np.float64)
        
        # Calculate MSE
        mse = np.mean((original - restored) ** 2)
        
        if mse == 0:
            return float('inf')
        
        # Calculate PSNR
        psnr = 20 * np.log10(max_value / np.sqrt(mse))
        
        return float(psnr)
        
    except Exception as e:
        logger.error(f"Error computing PSNR: {e}")
        return 0.0


def compute_ssim(
    original: np.ndarray,
    restored: np.ndarray,
    max_value: float = 255.0,
    k1: float = 0.01,
    k2: float = 0.03,
    sigma: float = 1.5
) -> float:
    """
    Compute Structural Similarity Index (SSIM).
    
    Args:
        original: Original image [H, W, C]
        restored: Restored image [H, W, C]
        max_value: Maximum possible pixel value
        k1, k2: SSIM constants
        sigma: Gaussian kernel standard deviation
        
    Returns:
        SSIM score in range [-1, 1], typically [0, 1]
    """
    try:
        # Resize if shapes don't match
        if original.shape != restored.shape:
            logger.warning(f"Shape mismatch: {original.shape} vs {restored.shape}, resizing...")
            original, restored = resize_to_match(original, restored)
        
        # Convert to float
        original = original.astype(np.float64)
        restored = restored.astype(np.float64)
        
        c1 = (k1 * max_value) ** 2
        c2 = (k2 * max_value) ** 2
        
        # Apply Gaussian filter
        from scipy.ndimage import gaussian_filter
        
        mu1 = gaussian_filter(original, sigma=sigma, mode='reflect')
        mu2 = gaussian_filter(restored, sigma=sigma, mode='reflect')
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = gaussian_filter(original ** 2, sigma=sigma, mode='reflect') - mu1_sq
        sigma2_sq = gaussian_filter(restored ** 2, sigma=sigma, mode='reflect') - mu2_sq
        sigma12 = gaussian_filter(original * restored, sigma=sigma, mode='reflect') - mu1_mu2
        
        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        
        return float(np.mean(ssim_map))
        
    except Exception as e:
        logger.error(f"Error computing SSIM: {e}")
        return 0.0


class LPIPSMetric:
    """Learned Perceptual Image Patch Similarity (LPIPS) metric."""
    
    def __init__(self, device: torch.device, network: str = 'alex'):
        """
        Initialize LPIPS metric.
        
        Args:
            device: Torch device
            network: Backbone network ('alex', 'vgg', 'squeeze')
        """
        self.device = device
        
        try:
            import lpips
            self.loss_fn = lpips.LPIPS(net=network).to(device)
            logger.info(f"LPIPS metric initialized with {network} network")
        except ImportError:
            logger.warning("lpips not installed, LPIPS metric unavailable")
            self.loss_fn = None
    
    def compute(
        self,
        original: np.ndarray,
        restored: np.ndarray
    ) -> Optional[float]:
        """
        Compute LPIPS distance.
        
        Args:
            original: Original image [H, W, C] in range [0, 255]
            restored: Restored image [H, W, C] in range [0, 255]
            
        Returns:
            LPIPS distance (lower is better)
        """
        if self.loss_fn is None:
            return None
        
        try:
            # Resize if shapes don't match
            if original.shape != restored.shape:
                logger.warning(f"Shape mismatch: {original.shape} vs {restored.shape}, resizing...")
                original, restored = resize_to_match(original, restored)
            
            # Normalize to [-1, 1]
            original = (original.astype(np.float32) / 127.5) - 1.0
            restored = (restored.astype(np.float32) / 127.5) - 1.0
            
            # Convert to tensors [1, C, H, W]
            original_tensor = torch.from_numpy(original).permute(2, 0, 1).unsqueeze(0)
            restored_tensor = torch.from_numpy(restored).permute(2, 0, 1).unsqueeze(0)
            
            original_tensor = original_tensor.to(self.device)
            restored_tensor = restored_tensor.to(self.device)
            
            # Compute LPIPS
            with torch.no_grad():
                distance = self.loss_fn(original_tensor, restored_tensor)
            
            return float(distance.item())
            
        except Exception as e:
            logger.error(f"Error computing LPIPS: {e}")
            return None


def compute_quality_score(
    original: np.ndarray,
    restored: np.ndarray,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Compute overall quality score combining multiple metrics.
    
    For super-resolution, we don't compare to original since sizes differ.
    Instead, we compute image quality metrics on the restored image.
    
    Args:
        original: Original image (may be different size)
        restored: Restored image
        weights: Optional weights for each metric
        
    Returns:
        Combined quality score in range [0, 1]
    """
    try:
        # For super-resolution, just compute metrics on restored image
        # since comparing different sizes doesn't make sense
        
        if original.shape == restored.shape:
            # Same size - can compute comparative metrics
            if weights is None:
                weights = {'psnr': 0.3, 'ssim': 0.7}
            
            scores = {}
            
            # PSNR (normalize to [0, 1])
            psnr = compute_psnr(original, restored)
            scores['psnr'] = min(psnr / 50.0, 1.0)  # 50dB is excellent
            
            # SSIM (already in [0, 1])
            ssim = compute_ssim(original, restored)
            scores['ssim'] = max(ssim, 0.0)
            
            # Weighted average
            quality = sum(scores[k] * weights.get(k, 0.0) for k in scores)
        else:
            # Different sizes (super-resolution case)
            # Compute quality based on restored image characteristics
            quality = compute_image_quality(restored)
        
        return float(quality)
        
    except Exception as e:
        logger.error(f"Error computing quality score: {e}")
        return 0.5  # Return neutral score on error


def compute_image_quality(image: np.ndarray) -> float:
    """
    Compute image quality based on intrinsic characteristics.
    Used for super-resolution where we can't compare to original.
    
    Args:
        image: Input image
        
    Returns:
        Quality score in range [0, 1]
    """
    try:
        scores = []
        
        # Sharpness
        sharpness = compute_image_sharpness(image)
        sharpness_score = min(sharpness / 1000.0, 1.0)  # Normalize
        scores.append(sharpness_score)
        
        # Contrast
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        contrast = np.std(gray)
        contrast_score = min(contrast / 50.0, 1.0)  # Normalize
        scores.append(contrast_score)
        
        # Edge strength
        edge_score = compute_edge_strength(image)
        scores.append(edge_score)
        
        # Average all scores
        quality = np.mean(scores)
        
        return float(quality)
        
    except Exception as e:
        logger.error(f"Error computing image quality: {e}")
        return 0.5


def compute_image_sharpness(image: np.ndarray) -> float:
    """
    Compute image sharpness using Laplacian variance.
    
    Args:
        image: Input image [H, W, C]
        
    Returns:
        Sharpness score (higher is sharper)
    """
    try:
        from scipy.ndimage import laplace
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Compute Laplacian
        laplacian = laplace(gray)
        
        # Variance of Laplacian
        sharpness = np.var(laplacian)
        
        return float(sharpness)
        
    except Exception as e:
        logger.error(f"Error computing sharpness: {e}")
        return 0.0


def compute_edge_strength(image: np.ndarray) -> float:
    """
    Compute edge strength in image.
    
    Args:
        image: Input image
        
    Returns:
        Edge strength score in [0, 1]
    """
    try:
        from scipy.ndimage import sobel
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Compute gradients
        sx = sobel(gray, axis=0)
        sy = sobel(gray, axis=1)
        
        # Edge magnitude
        edge_mag = np.sqrt(sx**2 + sy**2)
        
        # Normalize
        edge_strength = np.mean(edge_mag) / 255.0
        
        return float(min(edge_strength * 2, 1.0))
        
    except Exception as e:
        logger.error(f"Error computing edge strength: {e}")
        return 0.0


def compute_color_distribution(image: np.ndarray) -> Dict[str, Any]:
    """
    Analyze color distribution in image.
    
    Args:
        image: Input image [H, W, C]
        
    Returns:
        Dictionary with color statistics
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        return {'error': 'Image must be RGB'}
    
    stats = {}
    
    for i, channel in enumerate(['red', 'green', 'blue']):
        channel_data = image[:, :, i]
        stats[channel] = {
            'mean': float(np.mean(channel_data)),
            'std': float(np.std(channel_data)),
            'min': float(np.min(channel_data)),
            'max': float(np.max(channel_data)),
            'median': float(np.median(channel_data))
        }
    
    return stats


def compute_noise_level(image: np.ndarray) -> float:
    """
    Estimate noise level in image.
    
    Args:
        image: Input image [H, W, C]
        
    Returns:
        Estimated noise standard deviation
    """
    try:
        from scipy.ndimage import median_filter
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Apply median filter
        filtered = median_filter(gray, size=3)
        
        # Compute noise as difference
        noise = gray - filtered
        
        # Estimate noise std using median absolute deviation
        noise_std = np.median(np.abs(noise)) / 0.6745
        
        return float(noise_std)
        
    except Exception as e:
        logger.error(f"Error computing noise level: {e}")
        return 0.0


def evaluate_restoration(
    original: np.ndarray,
    restored: np.ndarray,
    compute_lpips: bool = False,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Comprehensive evaluation of image restoration.
    
    Args:
        original: Original image
        restored: Restored image
        compute_lpips: Whether to compute LPIPS (requires GPU)
        device: Torch device for LPIPS
        
    Returns:
        Dictionary with all metrics
    """
    results = {}
    
    try:
        # Check if sizes match
        size_match = original.shape == restored.shape
        
        if size_match:
            # Can compute comparative metrics
            results['psnr'] = compute_psnr(original, restored)
            results['ssim'] = compute_ssim(original, restored)
            results['quality_score'] = compute_quality_score(original, restored)
        else:
            # Different sizes - compute intrinsic quality only
            logger.info("Image sizes differ, computing intrinsic quality metrics only")
            results['quality_score'] = compute_image_quality(restored)
            results['psnr'] = 0.0  # Not applicable
            results['ssim'] = 0.0  # Not applicable
        
        # Always compute these on restored image
        results['sharpness'] = compute_image_sharpness(restored)
        results['noise_level'] = compute_noise_level(restored)
        results['edge_strength'] = compute_edge_strength(restored)
        
        if compute_lpips and device is not None and size_match:
            lpips_metric = LPIPSMetric(device)
            lpips_score = lpips_metric.compute(original, restored)
            if lpips_score is not None:
                results['lpips'] = lpips_score
        
        logger.info(f"Evaluation complete: Quality={results['quality_score']:.4f}")
        
    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}")
        results['error'] = str(e)
        results['quality_score'] = 0.5
    
    return results
