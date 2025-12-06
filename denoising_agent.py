"""
Denoising Agent - Fixed Version
Removes tensor size mismatch errors
"""

import logging
from typing import Dict, Any
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os

logger = logging.getLogger(__name__)


class SimpleDenoiser(nn.Module):
    """Simple denoising model without complex skip connections."""
    
    def __init__(self, in_channels: int = 3):
        super().__init__()
        
        # Simple sequential denoising network
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_channels, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class DenoisingAgent:
    """Agent responsible for image denoising."""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        """
        Initialize denoising agent.
        
        Args:
            config: Configuration dictionary
            device: Torch device (cuda/cpu)
        """
        self.config = config
        self.device = device
        
        logger.info("Initializing denoising agent...")
        
        # Use simple model
        self.model = SimpleDenoiser(in_channels=3).to(device)
        
        # Try to load pretrained weights if available
        model_path = config.get('model_path')
        if model_path and os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=device)
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded denoising model from {model_path}")
                self.use_model = True
            except Exception as e:
                logger.warning(f"Could not load model: {e}")
                logger.warning("Using basic denoising")
                self.use_model = False
        else:
            logger.warning("No pre-trained model, using basic denoising")
            self.use_model = False
        
        self.model.eval()
        
    @torch.no_grad()
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Remove noise from image.
        
        Args:
            image: Input image as numpy array [H, W, C] in range [0, 255]
            
        Returns:
            Denoised image as numpy array
        """
        try:
            logger.info(f"Denoising image of shape: {image.shape}")
            
            # Preprocessing
            original_shape = image.shape
            image_tensor = self._preprocess(image)
            
            # Denoising
            if self.use_model:
                denoised_tensor = self.model(image_tensor)
            else:
                # Fallback: simple bilateral filtering
                denoised_tensor = self._fallback_denoise(image_tensor)
            
            # Postprocessing
            denoised = self._postprocess(denoised_tensor, original_shape)
            
            logger.info("Denoising completed successfully")
            return denoised
            
        except Exception as e:
            logger.error(f"Error in denoising: {str(e)}")
            # Return original image if denoising fails
            logger.warning("Returning original image")
            return image
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        # Normalize to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # Handle grayscale
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        # Convert to tensor [1, C, H, W]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(self.device).float()
        
        return image_tensor
    
    def _fallback_denoise(self, noisy_image: torch.Tensor) -> torch.Tensor:
        """Simple averaging denoising fallback."""
        # Apply Gaussian blur for denoising
        kernel_size = 5
        sigma = 1.0
        
        # Create Gaussian kernel
        kernel = self._gaussian_kernel(kernel_size, sigma)
        kernel = kernel.to(self.device)
        
        # Apply convolution
        denoised = torch.nn.functional.conv2d(
            noisy_image,
            kernel.repeat(3, 1, 1, 1),
            padding=kernel_size // 2,
            groups=3
        )
        
        return torch.clamp(denoised, 0, 1)
    
    def _gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """Create Gaussian kernel for denoising."""
        x = torch.arange(kernel_size).float() - kernel_size // 2
        gauss = torch.exp(-x.pow(2.0) / (2 * sigma ** 2))
        kernel = gauss.unsqueeze(0) * gauss.unsqueeze(1)
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)
    
    def _postprocess(
        self,
        tensor: torch.Tensor,
        original_shape: tuple
    ) -> np.ndarray:
        """Convert tensor back to numpy array."""
        # Convert to numpy [H, W, C]
        image = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Clip to valid range
        image = np.clip(image, 0, 1)
        
        # Convert back to uint8
        image = (image * 255).astype(np.uint8)
        
        return image
    
    def estimate_noise_level(self, image: np.ndarray) -> float:
        """
        Estimate noise level in image.
        
        Args:
            image: Input image
            
        Returns:
            Estimated noise standard deviation
        """
        # Simple noise estimation
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Use Laplacian variance
        from scipy.ndimage import laplace
        lap = laplace(gray)
        noise_std = np.std(lap)
        
        return float(noise_std)
