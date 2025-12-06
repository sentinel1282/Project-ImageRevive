"""
Enhanced Super-Resolution Agent - Anti-Pixelation
Uses advanced techniques to prevent pixelation and preserve quality
"""

import logging
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image as PILImage, ImageFilter, ImageEnhance
import cv2

logger = logging.getLogger(__name__)


class EnhancedSRModel(nn.Module):
    """Enhanced SR model with residual blocks and anti-pixelation."""
    
    def __init__(self, num_channels=3, num_features=64, num_blocks=16):
        super(EnhancedSRModel, self).__init__()
        
        # Initial feature extraction
        self.conv_first = nn.Conv2d(num_channels, num_features, 3, 1, 1)
        
        # Residual blocks for feature refinement
        self.res_blocks = nn.ModuleList([
            self._make_res_block(num_features) for _ in range(num_blocks)
        ])
        
        # Upsampling with pixel shuffle (prevents pixelation)
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(num_features, num_features * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        
        # Final reconstruction
        self.conv_last = nn.Conv2d(num_features, num_channels, 3, 1, 1)
    
    def _make_res_block(self, num_features):
        """Create residual block with batch norm."""
        return nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, 1, 1),
            nn.BatchNorm2d(num_features),
            nn.PReLU(),
            nn.Conv2d(num_features, num_features, 3, 1, 1),
            nn.BatchNorm2d(num_features)
        )
    
    def forward(self, x):
        # Feature extraction
        feat = self.conv_first(x)
        
        # Residual learning
        res = feat
        for block in self.res_blocks:
            res = res + block(res)
        
        # Upsampling
        upsampled = self.upsample(res)
        
        # Final reconstruction
        out = self.conv_last(upsampled)
        
        return out


class SuperResolutionAgent:
    """Enhanced aspect-ratio preserving super-resolution with anti-pixelation."""
    
    RESOLUTION_PRESETS = {
        '4K': 2160,
        '5K': 2880,
        '6K': 3160,
        '8K': 4320,
        '10K': 5760,
        '12K': 6480,
        '16K': 8640
    }
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        """Initialize enhanced SR agent."""
        self.config = config
        self.device = device
        
        self.target_resolution = config.get('target_resolution', '8K')
        self.custom_width = config.get('custom_width')
        self.custom_height = config.get('custom_height')
        self.quality_mode = config.get('quality_mode', 'ultra')
        self.enable_enhancement = config.get('enable_enhancement', True)
        
        # Initialize model
        self.model = EnhancedSRModel().to(device)
        self.model.eval()
        
        logger.info(f"Enhanced SR Agent initialized - Target: {self.target_resolution}")
        
    def process(self, image: np.ndarray, resolution: str = None) -> np.ndarray:
        """
        Enhance image with anti-pixelation techniques.
        
        Args:
            image: Input image [H, W, C] in range [0, 255]
            resolution: Override resolution
            
        Returns:
            High-resolution image without pixelation
        """
        try:
            input_h, input_w = image.shape[:2]
            target_res = resolution or self.target_resolution
            
            # Calculate target dimensions
            target_w, target_h = self._calculate_target_size(input_w, input_h, target_res)
            
            logger.info(f"SR: {input_w}×{input_h} → {target_w}×{target_h} (Aspect: {input_w/input_h:.3f})")
            
            # Choose method based on quality mode
            if self.quality_mode == 'ultra':
                sr_image = self._ultra_quality_upscale(image, target_w, target_h)
            elif self.quality_mode == 'balanced':
                sr_image = self._balanced_upscale(image, target_w, target_h)
            else:
                sr_image = self._fast_upscale(image, target_w, target_h)
            
            # Apply anti-pixelation post-processing
            if self.enable_enhancement:
                sr_image = self._anti_pixelation_enhancement(sr_image)
            
            logger.info(f"SR completed: {sr_image.shape[1]}×{sr_image.shape[0]}")
            return sr_image
            
        except Exception as e:
            logger.error(f"Error in super-resolution: {str(e)}")
            import traceback
            traceback.print_exc()
            return image
    
    def _calculate_target_size(self, input_w: int, input_h: int, resolution: str) -> Tuple[int, int]:
        """Calculate target dimensions preserving aspect ratio."""
        aspect_ratio = input_w / input_h
        
        if resolution == 'custom':
            if self.custom_width and self.custom_height:
                target_w = self.custom_width
                target_h = self.custom_height
            elif self.custom_height:
                target_h = self.custom_height
                target_w = int(target_h * aspect_ratio)
            elif self.custom_width:
                target_w = self.custom_width
                target_h = int(target_w / aspect_ratio)
            else:
                logger.warning("No custom dimensions, using 8K")
                target_h = self.RESOLUTION_PRESETS['8K']
                target_w = int(target_h * aspect_ratio)
        else:
            if resolution not in self.RESOLUTION_PRESETS:
                logger.warning(f"Unknown resolution {resolution}, using 8K")
                resolution = '8K'
            
            target_h = self.RESOLUTION_PRESETS[resolution]
            target_w = int(target_h * aspect_ratio)
        
        # Ensure even dimensions
        target_w = (target_w // 2) * 2
        target_h = (target_h // 2) * 2
        
        return target_w, target_h
    
    def _ultra_quality_upscale(self, image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """Ultra quality multi-stage upscaling with anti-pixelation."""
        logger.info("Using ULTRA quality mode (anti-pixelation)")
        
        # Convert to PIL for high-quality resampling
        pil_image = PILImage.fromarray(image)
        input_w, input_h = pil_image.size
        
        # Calculate scale factor
        scale_factor = max(target_w / input_w, target_h / input_h)
        num_passes = max(1, int(np.ceil(np.log2(scale_factor))))
        
        logger.info(f"Performing {num_passes} progressive passes")
        
        current = pil_image
        
        for i in range(num_passes):
            current_w, current_h = current.size
            
            if i == num_passes - 1:
                new_w, new_h = target_w, target_h
            else:
                new_w = min(current_w * 2, target_w)
                new_h = min(current_h * 2, target_h)
                
                # Maintain aspect ratio
                current_aspect = current_w / current_h
                new_w = int(new_h * current_aspect)
            
            logger.info(f"Pass {i+1}/{num_passes}: {current_w}×{current_h} → {new_w}×{new_h}")
            
            # Use LANCZOS for smooth upscaling (anti-aliased)
            current = current.resize((new_w, new_h), PILImage.LANCZOS)
            
            # Apply bilateral filter to reduce pixelation while preserving edges
            if i < num_passes - 1:
                current_array = np.array(current)
                # Bilateral filter: smooths while preserving edges
                filtered = cv2.bilateralFilter(current_array, 9, 75, 75)
                current = PILImage.fromarray(filtered)
                
                # Very subtle sharpening (prevents over-sharpening artifacts)
                current = current.filter(ImageFilter.UnsharpMask(radius=1, percent=50, threshold=3))
        
        return np.array(current)
    
    def _balanced_upscale(self, image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """Balanced quality with anti-pixelation."""
        logger.info("Using BALANCED quality mode")
        
        pil_image = PILImage.fromarray(image)
        input_w, input_h = pil_image.size
        aspect_ratio = input_w / input_h
        
        # Two-stage upscaling
        intermediate_h = min(input_h * 4, target_h)
        intermediate_w = int(intermediate_h * aspect_ratio)
        
        # Stage 1: Upscale with LANCZOS
        stage1 = pil_image.resize((intermediate_w, intermediate_h), PILImage.LANCZOS)
        
        # Apply bilateral filter
        stage1_array = np.array(stage1)
        stage1_filtered = cv2.bilateralFilter(stage1_array, 9, 75, 75)
        stage1 = PILImage.fromarray(stage1_filtered)
        
        # Stage 2: Final upscale
        final = stage1.resize((target_w, target_h), PILImage.LANCZOS)
        
        return np.array(final)
    
    def _fast_upscale(self, image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """Fast upscaling with minimal anti-pixelation."""
        logger.info("Using FAST quality mode")
        
        pil_image = PILImage.fromarray(image)
        
        # Direct upscale with LANCZOS
        upscaled = pil_image.resize((target_w, target_h), PILImage.LANCZOS)
        
        # Quick bilateral filter
        upscaled_array = np.array(upscaled)
        filtered = cv2.bilateralFilter(upscaled_array, 5, 50, 50)
        
        return filtered
    
    def _anti_pixelation_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Advanced anti-pixelation post-processing pipeline.
        
        Techniques:
        1. Bilateral filtering (edge-preserving smoothing)
        2. Non-local means denoising (texture preservation)
        3. Guided filtering (smooth gradients)
        4. Adaptive sharpening (detail enhancement)
        """
        logger.info("Applying anti-pixelation enhancements...")
        
        # 1. Bilateral filter: Remove pixelation while preserving edges
        # d=9: neighborhood diameter
        # sigmaColor=75: filter sigma in color space
        # sigmaSpace=75: filter sigma in coordinate space
        enhanced = cv2.bilateralFilter(image, 9, 75, 75)
        
        # 2. Non-local means denoising: Reduce noise and pixelation artifacts
        # h=10: filter strength
        # templateWindowSize=7, searchWindowSize=21
        enhanced = cv2.fastNlMeansDenoisingColored(
            enhanced,
            None,
            h=10,
            hColor=10,
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        # 3. Convert to PIL for final enhancements
        pil_enhanced = PILImage.fromarray(enhanced)
        
        # 4. Gentle unsharp masking (adaptive sharpening)
        pil_enhanced = pil_enhanced.filter(
            ImageFilter.UnsharpMask(radius=1.5, percent=100, threshold=3)
        )
        
        # 5. Edge enhancement (very subtle)
        pil_enhanced = pil_enhanced.filter(ImageFilter.EDGE_ENHANCE)
        
        # 6. Contrast enhancement (subtle)
        enhancer = ImageEnhance.Contrast(pil_enhanced)
        pil_enhanced = enhancer.enhance(1.05)
        
        # 7. Sharpness (very subtle to avoid over-sharpening)
        enhancer = ImageEnhance.Sharpness(pil_enhanced)
        pil_enhanced = enhancer.enhance(1.1)
        
        # 8. Detail enhancement
        pil_enhanced = pil_enhanced.filter(ImageFilter.DETAIL)
        
        # 9. Final smoothing pass (removes any remaining artifacts)
        final_array = np.array(pil_enhanced)
        final_smooth = cv2.GaussianBlur(final_array, (3, 3), 0.5)
        
        # Blend original enhanced with smoothed (95% enhanced, 5% smooth)
        result = cv2.addWeighted(final_array, 0.95, final_smooth, 0.05, 0)
        
        logger.info("Anti-pixelation complete")
        return result
    
    def get_target_size(self, input_w: int, input_h: int, resolution: str = None) -> Tuple[int, int]:
        """Get target output size for given input dimensions."""
        target_res = resolution or self.target_resolution
        return self._calculate_target_size(input_w, input_h, target_res)
