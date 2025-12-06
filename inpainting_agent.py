"""
Inpainting Agent - Fixed Version
Removes tensor size mismatch errors
"""

import logging
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

logger = logging.getLogger(__name__)


class SimpleInpainter(nn.Module):
    """Simple inpainting model without complex skip connections."""
    
    def __init__(self, in_channels: int = 4):  # 3 for image + 1 for mask
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        
        # Middle
        self.mid = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.dec3 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dec1 = nn.Conv2d(64, 3, 3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for inpainting.
        
        Args:
            x: Input image [B, 3, H, W]
            mask: Binary mask [B, 1, H, W]
        
        Returns:
            Inpainted image [B, 3, H, W]
        """
        # Concatenate image and mask
        x = torch.cat([x, mask], dim=1)
        
        # Encoder
        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.enc2(e1))
        e3 = self.relu(self.enc3(e2))
        
        # Middle
        m = self.mid(e3)
        
        # Decoder
        d3 = self.relu(self.dec3(m))
        d2 = self.relu(self.dec2(d3))
        d1 = self.sigmoid(self.dec1(d2))
        
        return d1


class InpaintingAgent:
    """Agent responsible for image inpainting."""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        """
        Initialize inpainting agent.
        
        Args:
            config: Configuration dictionary
            device: Torch device (cuda/cpu)
        """
        self.config = config
        self.device = device
        
        logger.info("Initializing inpainting agent...")
        
        # Use simple model
        self.model = SimpleInpainter().to(device)
        
        # Try to load pretrained weights if available
        model_path = config.get('model_path')
        if model_path and os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=device)
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded inpainting model from {model_path}")
                self.use_model = True
            except Exception as e:
                logger.warning(f"Could not load model: {e}")
                logger.warning("Using simple inpainting")
                self.use_model = False
        else:
            logger.warning("No pre-trained model, using simple inpainting")
            self.use_model = False
        
        self.model.eval()
        
    @torch.no_grad()
    def process(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Inpaint missing regions in image.
        
        Args:
            image: Input image [H, W, C] in range [0, 255]
            mask: Binary mask [H, W], 0 for regions to inpaint, 1 for valid
                  If None, attempts to detect damaged regions automatically
            
        Returns:
            Inpainted image as numpy array
        """
        try:
            # Generate mask if not provided
            if mask is None:
                mask = self._detect_damaged_regions(image)
                if mask is None:
                    logger.info("No damaged regions detected, returning original")
                    return image
            
            logger.info(f"Inpainting image of shape: {image.shape}")
            
            # Preprocessing
            image_tensor, mask_tensor = self._preprocess(image, mask)
            
            # Inpainting
            if self.use_model:
                inpainted_tensor = self.model(image_tensor, mask_tensor)
            else:
                # Fallback: simple interpolation
                inpainted_tensor = self._simple_inpaint(image_tensor, mask_tensor)
            
            # Blend with original valid regions
            output = image_tensor * mask_tensor + inpainted_tensor * (1 - mask_tensor)
            
            # Postprocessing
            inpainted = self._postprocess(output)
            
            logger.info("Inpainting completed successfully")
            return inpainted
            
        except Exception as e:
            logger.error(f"Error in inpainting: {str(e)}")
            logger.warning("Returning original image")
            return image
    
    def _preprocess(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> tuple:
        """Preprocess image and mask for model input."""
        # Normalize image to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # Handle grayscale
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        # Ensure mask is binary
        if mask.dtype != np.uint8:
            mask = (mask > 0.5).astype(np.uint8)
        
        # Expand mask dimensions if needed
        if len(mask.shape) == 2:
            mask = mask[:, :, np.newaxis]
        
        # Convert to tensors [1, C, H, W]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).permute(2, 0, 1).unsqueeze(0)
        
        image_tensor = image_tensor.to(self.device).float()
        mask_tensor = mask_tensor.to(self.device).float()
        
        return image_tensor, mask_tensor
    
    def _simple_inpaint(
        self,
        image: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Simple inpainting using interpolation."""
        # Use nearest neighbor interpolation for holes
        # This is a very simple fallback
        
        # Create blurred version
        kernel_size = 21
        sigma = 5.0
        
        # Create Gaussian kernel
        kernel = self._gaussian_kernel(kernel_size, sigma)
        kernel = kernel.to(self.device)
        
        # Apply blur
        blurred = F.conv2d(
            image,
            kernel.repeat(3, 1, 1, 1),
            padding=kernel_size // 2,
            groups=3
        )
        
        return torch.clamp(blurred, 0, 1)
    
    def _gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """Create Gaussian kernel."""
        x = torch.arange(kernel_size).float() - kernel_size // 2
        gauss = torch.exp(-x.pow(2.0) / (2 * sigma ** 2))
        kernel = gauss.unsqueeze(0) * gauss.unsqueeze(1)
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)
    
    def _postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor back to numpy array."""
        # Convert to numpy [H, W, C]
        image = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Clip to valid range
        image = np.clip(image, 0, 1)
        
        # Convert back to uint8
        image = (image * 255).astype(np.uint8)
        
        return image
    
    def _detect_damaged_regions(
        self,
        image: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Automatically detect damaged regions in image.
        
        Args:
            image: Input image
            
        Returns:
            Binary mask (1 for valid, 0 for damaged) or None if no damage
        """
        try:
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = np.mean(image, axis=2)
            else:
                gray = image
            
            # Detect very dark or very bright regions (potential damage)
            dark_threshold = 10
            bright_threshold = 245
            
            damaged = (gray < dark_threshold) | (gray > bright_threshold)
            
            # Check if significant damage detected
            damage_ratio = np.sum(damaged) / damaged.size
            
            if damage_ratio < 0.01:  # Less than 1% damaged
                return None
            
            if damage_ratio > 0.8:  # More than 80% damaged
                logger.warning("Image appears heavily damaged")
                return None
            
            # Create mask (1 for valid, 0 for damaged)
            mask = (~damaged).astype(np.uint8)
            
            return mask
            
        except Exception as e:
            logger.error(f"Error detecting damage: {e}")
            return None
