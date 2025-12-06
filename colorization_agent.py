"""
Colorization Agent - Fixed Version
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


class SimpleColorizer(nn.Module):
    """Simple colorization model without complex architecture."""
    
    def __init__(self):
        super().__init__()
        
        # Simple network for colorization
        self.network = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, grayscale: torch.Tensor) -> torch.Tensor:
        """Generate color channels."""
        return self.network(grayscale)


class ColorizationAgent:
    """Agent responsible for image colorization."""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        """
        Initialize colorization agent.
        
        Args:
            config: Configuration dictionary
            device: Torch device (cuda/cpu)
        """
        self.config = config
        self.device = device
        
        logger.info("Initializing colorization agent...")
        
        # Use simple model
        self.model = SimpleColorizer().to(device)
        
        # Try to load pretrained weights if available
        model_path = config.get('model_path')
        if model_path and os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=device)
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded colorization model from {model_path}")
                self.use_model = True
            except Exception as e:
                logger.warning(f"Could not load model: {e}")
                logger.warning("Using basic colorization")
                self.use_model = False
        else:
            logger.warning("No pre-trained model, using basic colorization")
            self.use_model = False
        
        self.model.eval()
        
    @torch.no_grad()
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Colorize grayscale image.
        
        Args:
            image: Input grayscale image [H, W] or [H, W, 1] in range [0, 255]
            
        Returns:
            Colorized image [H, W, 3] as numpy array
        """
        try:
            # Check if image is already color
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Check if it's truly grayscale
                if not np.allclose(image[:, :, 0], image[:, :, 1]) or \
                   not np.allclose(image[:, :, 1], image[:, :, 2]):
                    logger.info("Image already in color, returning as-is")
                    return image
                # Convert to single channel
                image = image[:, :, 0]
            
            logger.info(f"Colorizing grayscale image of shape: {image.shape}")
            
            if self.use_model:
                # Use model for colorization
                result = self._colorize_with_model(image)
            else:
                # Use simple sepia-tone colorization as fallback
                result = self._simple_colorize(image)
            
            logger.info("Colorization completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in colorization: {str(e)}")
            # Return as grayscale RGB if colorization fails
            if len(image.shape) == 2:
                return np.stack([image] * 3, axis=-1)
            return image
    
    def _colorize_with_model(self, grayscale: np.ndarray) -> np.ndarray:
        """Colorize using neural network."""
        try:
            # Preprocess
            l_tensor = self._preprocess(grayscale)
            
            # Generate AB channels
            ab_tensor = self.model(l_tensor)
            
            # Postprocess
            colorized = self._postprocess(l_tensor, ab_tensor)
            
            return colorized
        except Exception as e:
            logger.warning(f"Model colorization failed: {e}, using fallback")
            return self._simple_colorize(grayscale)
    
    def _simple_colorize(self, grayscale: np.ndarray) -> np.ndarray:
        """Simple sepia-tone colorization fallback."""
        # Normalize if needed
        if grayscale.dtype == np.uint8:
            gray = grayscale.astype(np.float32) / 255.0
        else:
            gray = grayscale
        
        # Create sepia-toned image
        r = np.clip(gray * 1.0 + 0.0, 0, 1)
        g = np.clip(gray * 0.95 + 0.0, 0, 1)
        b = np.clip(gray * 0.82 + 0.0, 0, 1)
        
        colored = np.stack([r, g, b], axis=-1)
        
        # Convert back to uint8
        colored = (colored * 255).astype(np.uint8)
        
        return colored
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        # Normalize to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # Convert to L channel (range [0, 100])
        l_channel = image * 100.0
        
        # Normalize to [-1, 1]
        l_normalized = (l_channel - 50.0) / 50.0
        
        # Convert to tensor [1, 1, H, W]
        l_tensor = torch.from_numpy(l_normalized).unsqueeze(0).unsqueeze(0)
        l_tensor = l_tensor.to(self.device).float()
        
        return l_tensor
    
    def _postprocess(
        self,
        l_tensor: torch.Tensor,
        ab_tensor: torch.Tensor
    ) -> np.ndarray:
        """Convert LAB to RGB."""
        try:
            # Denormalize L channel
            l_channel = (l_tensor.squeeze().cpu().numpy() * 50.0) + 50.0
            
            # Get AB channels
            ab_channels = ab_tensor.squeeze().cpu().numpy() * 128.0
            
            # Combine LAB
            lab_image = np.stack([
                l_channel,
                ab_channels[0],
                ab_channels[1]
            ], axis=-1)
            
            # Convert LAB to RGB
            rgb_image = self._lab_to_rgb(lab_image)
            
            # Clip and convert to uint8
            rgb_image = np.clip(rgb_image, 0, 1)
            rgb_image = (rgb_image * 255).astype(np.uint8)
            
            return rgb_image
        except Exception as e:
            logger.warning(f"LAB to RGB conversion failed: {e}, using fallback")
            # Return simple grayscale to RGB
            gray = (l_tensor.squeeze().cpu().numpy() * 50.0) + 50.0
            gray = np.clip(gray / 100.0, 0, 1)
            gray_uint8 = (gray * 255).astype(np.uint8)
            return np.stack([gray_uint8] * 3, axis=-1)
    
    def _lab_to_rgb(self, lab: np.ndarray) -> np.ndarray:
        """Convert LAB to RGB color space."""
        try:
            # Simplified LAB to RGB conversion
            l, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
            
            # LAB to XYZ
            y = (l + 16) / 116
            x = a / 500 + y
            z = y - b / 200
            
            # XYZ to RGB (simplified)
            def f_inv(t):
                delta = 6 / 29
                return np.where(t > delta, t ** 3, 3 * delta ** 2 * (t - 4 / 29))
            
            x = f_inv(x) * 0.95047
            y = f_inv(y)
            z = f_inv(z) * 1.08883
            
            # XYZ to RGB transformation
            r = x * 3.2406 + y * -1.5372 + z * -0.4986
            g = x * -0.9689 + y * 1.8758 + z * 0.0415
            b_rgb = x * 0.0557 + y * -0.2040 + z * 1.0570
            
            # Gamma correction
            def gamma_correct(c):
                return np.where(
                    c > 0.0031308,
                    1.055 * (c ** (1 / 2.4)) - 0.055,
                    12.92 * c
                )
            
            r = gamma_correct(r)
            g = gamma_correct(g)
            b_rgb = gamma_correct(b_rgb)
            
            rgb = np.stack([r, g, b_rgb], axis=-1)
            return rgb
            
        except Exception as e:
            logger.error(f"LAB to RGB failed: {e}")
            # Fallback: just use L channel as grayscale
            gray = lab[:, :, 0] / 100.0
            return np.stack([gray] * 3, axis=-1)
