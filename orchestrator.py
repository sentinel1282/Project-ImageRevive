"""
ImageRevive Orchestrator - Fixed Version
Fixes recursion limit error by improving workflow routing
"""

import logging
from typing import Dict, List, Optional, Any, TypedDict
from pathlib import Path
import torch
from PIL import Image
import numpy as np

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

from denoising_agent import DenoisingAgent
from super_resolution_agent import SuperResolutionAgent
from colorization_agent import ColorizationAgent
from inpainting_agent import InpaintingAgent

logger = logging.getLogger(__name__)


class ImageState(TypedDict):
    """State definition for image restoration workflow."""
    
    image: np.ndarray
    original_image: np.ndarray
    tasks: List[str]
    completed_tasks: List[str]
    metadata: Dict[str, Any]
    quality_score: float
    error: Optional[str]
    retry_count: int  # Track retries to prevent infinite loops


class ImageRestoreOrchestrator:
    """
    Orchestrates multiple specialized agents for comprehensive image restoration.
    
    Uses LangGraph to manage workflow state and agent coordination.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize orchestrator with configuration.
        
        Args:
            config: Configuration dictionary with model and system settings
        """
        self.config = config
        self.device = torch.device(
            config['system']['device'] 
            if torch.cuda.is_available() 
            else 'cpu'
        )
        
        logger.info(f"Initializing ImageRestoreOrchestrator on {self.device}")
        
        # Initialize specialized agents
        self._initialize_agents()
        
        # Build workflow graph
        self.workflow = self._build_workflow()
        
    def _initialize_agents(self):
        """Initialize all specialized restoration agents."""
        try:
            self.denoising_agent = DenoisingAgent(
                self.config['models']['denoising'],
                self.device
            )
            logger.info("Denoising agent initialized")
            
            self.sr_agent = SuperResolutionAgent(
                self.config['models']['super_resolution'],
                self.device
            )
            logger.info("Super-resolution agent initialized")
            
            self.colorization_agent = ColorizationAgent(
                self.config['models']['colorization'],
                self.device
            )
            logger.info("Colorization agent initialized")
            
            self.inpainting_agent = InpaintingAgent(
                self.config['models']['inpainting'],
                self.device
            )
            logger.info("Inpainting agent initialized")
            
        except Exception as e:
            logger.error(f"Error initializing agents: {str(e)}")
            raise
    
    def _build_workflow(self) -> StateGraph:
        """
        Build LangGraph workflow for orchestrating restoration tasks.
        
        Returns:
            StateGraph: Compiled workflow graph
        """
        workflow = StateGraph(ImageState)
        
        # Add nodes for each restoration task
        workflow.add_node("analyze", self._analyze_image)
        workflow.add_node("denoise", self._denoise_step)
        workflow.add_node("super_resolve", self._super_resolve_step)
        workflow.add_node("colorize", self._colorize_step)
        workflow.add_node("inpaint", self._inpaint_step)
        workflow.add_node("validate", self._validate_quality)
        
        # Define workflow edges
        workflow.set_entry_point("analyze")
        
        # Conditional routing based on tasks
        workflow.add_conditional_edges(
            "analyze",
            self._route_next_task,
            {
                "denoise": "denoise",
                "super_resolve": "super_resolve",
                "colorize": "colorize",
                "inpaint": "inpaint",
                "validate": "validate",
                END: END
            }
        )
        
        # Each task routes to next task
        for task in ["denoise", "super_resolve", "colorize", "inpaint"]:
            workflow.add_conditional_edges(
                task,
                self._route_next_task,
                {
                    "denoise": "denoise",
                    "super_resolve": "super_resolve",
                    "colorize": "colorize",
                    "inpaint": "inpaint",
                    "validate": "validate",
                    END: END
                }
            )
        
        # Validation always ends (no retries to prevent loops)
        workflow.add_edge("validate", END)
        
        return workflow.compile()
    
    def _analyze_image(self, state: ImageState) -> ImageState:
        """
        Analyze input image and determine required tasks.
        
        Args:
            state: Current image state
            
        Returns:
            Updated state with analysis metadata
        """
        logger.info("Analyzing input image")
        
        try:
            image = state['image']
            
            # Calculate image statistics
            metadata = {
                'height': image.shape[0],
                'width': image.shape[1],
                'channels': image.shape[2] if len(image.shape) > 2 else 1,
                'dtype': str(image.dtype),
                'mean': float(np.mean(image)),
                'std': float(np.std(image)),
                'min': float(np.min(image)),
                'max': float(np.max(image))
            }
            
            # Detect potential issues
            if metadata['std'] < 10:
                metadata['potential_noise'] = True
                
            if metadata['width'] < 512 or metadata['height'] < 512:
                metadata['low_resolution'] = True
                
            if metadata['channels'] == 1:
                metadata['grayscale'] = True
            
            state['metadata'].update(metadata)
            logger.info(f"Image analysis complete: {metadata}")
            
            return state
            
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            state['error'] = str(e)
            return state
    
    def _denoise_step(self, state: ImageState) -> ImageState:
        """Execute denoising task."""
        logger.info("Executing denoising step")
        
        try:
            image = state['image']
            denoised = self.denoising_agent.process(image)
            
            state['image'] = denoised
            state['completed_tasks'].append('denoising')
            
            logger.info("Denoising completed successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error in denoising: {str(e)}")
            state['error'] = str(e)
            return state
    
    def _super_resolve_step(self, state: ImageState) -> ImageState:
        """Execute super-resolution task."""
        logger.info("Executing super-resolution step")
        
        try:
            image = state['image']
            sr_image = self.sr_agent.process(image)
            
            state['image'] = sr_image
            state['completed_tasks'].append('super_resolution')
            
            logger.info("Super-resolution completed successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error in super-resolution: {str(e)}")
            state['error'] = str(e)
            return state
    
    def _colorize_step(self, state: ImageState) -> ImageState:
        """Execute colorization task."""
        logger.info("Executing colorization step")
        
        try:
            image = state['image']
            
            # Check if image is grayscale
            if len(image.shape) == 2 or state['metadata'].get('grayscale'):
                colorized = self.colorization_agent.process(image)
                state['image'] = colorized
                state['completed_tasks'].append('colorization')
                logger.info("Colorization completed successfully")
            else:
                logger.info("Image already in color, skipping colorization")
                state['completed_tasks'].append('colorization')
            
            return state
            
        except Exception as e:
            logger.error(f"Error in colorization: {str(e)}")
            state['error'] = str(e)
            return state
    
    def _inpaint_step(self, state: ImageState) -> ImageState:
        """Execute inpainting task."""
        logger.info("Executing inpainting step")
        
        try:
            image = state['image']
            mask = state['metadata'].get('inpainting_mask')
            
            if mask is not None:
                inpainted = self.inpainting_agent.process(image, mask)
                state['image'] = inpainted
                state['completed_tasks'].append('inpainting')
                logger.info("Inpainting completed successfully")
            else:
                logger.info("No inpainting mask provided, skipping")
                state['completed_tasks'].append('inpainting')
            
            return state
            
        except Exception as e:
            logger.error(f"Error in inpainting: {str(e)}")
            state['error'] = str(e)
            return state
    
    def _validate_quality(self, state: ImageState) -> ImageState:
        """Validate output quality."""
        logger.info("Validating output quality")
        
        try:
            from metrics import compute_quality_score, compute_image_quality
            
            original_shape = state['original_image'].shape
            current_shape = state['image'].shape
            
            logger.info(f"Original shape: {original_shape}, Current shape: {current_shape}")
            
            # For super-resolution, images may be different sizes
            if original_shape == current_shape:
                # Same size - can compare directly
                logger.info("Computing comparative quality metrics")
                quality = compute_quality_score(
                    state['original_image'],
                    state['image']
                )
            else:
                # Different sizes - compute intrinsic quality only
                logger.info("Images have different sizes, computing intrinsic quality")
                quality = compute_image_quality(state['image'])
            
            state['quality_score'] = quality
            logger.info(f"Quality score: {quality:.4f}")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in quality validation: {str(e)}")
            state['error'] = str(e)
            state['quality_score'] = 0.5  # Neutral score on error
            return state
    
    def _route_next_task(self, state: ImageState) -> str:
        """
        Determine next task based on pending tasks and current state.
        
        Args:
            state: Current workflow state
            
        Returns:
            Next task name or END
        """
        # Check for errors - always end on error
        if state.get('error'):
            logger.warning(f"Error detected, ending workflow: {state['error']}")
            return END
        
        # Check retry count to prevent infinite loops
        retry_count = state.get('retry_count', 0)
        if retry_count > 3:
            logger.warning("Max retries reached, ending workflow")
            return END
        
        # Get remaining tasks in priority order
        task_priority = self.config['orchestration']['task_priority']
        completed = set(state['completed_tasks'])
        requested = set(state['tasks'])
        
        for task in task_priority:
            if task in requested and task not in completed:
                # Map task names to node names
                task_mapping = {
                    'denoising': 'denoise',
                    'super_resolution': 'super_resolve',
                    'colorization': 'colorize',
                    'inpainting': 'inpaint'
                }
                next_node = task_mapping.get(task, 'validate')
                logger.info(f"Routing to: {next_node}")
                return next_node
        
        # All tasks complete, proceed to validation
        logger.info("All tasks complete, routing to validation")
        return 'validate'
    
    def restore(
        self,
        image: np.ndarray,
        tasks: List[str],
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Restore image using specified tasks.
        
        Args:
            image: Input image as numpy array
            tasks: List of restoration tasks to perform
            mask: Optional mask for inpainting
            
        Returns:
            Dictionary with restored image and metadata
        """
        logger.info(f"Starting restoration with tasks: {tasks}")
        
        # Initialize state
        initial_state: ImageState = {
            'image': image.copy(),
            'original_image': image.copy(),
            'tasks': tasks,
            'completed_tasks': [],
            'metadata': {},
            'quality_score': 0.0,
            'error': None,
            'retry_count': 0
        }
        
        if mask is not None:
            initial_state['metadata']['inpainting_mask'] = mask
        
        # Execute workflow
        try:
            # Set recursion limit
            config = {"recursion_limit": 50}
            
            final_state = self.workflow.invoke(initial_state, config=config)
            
            return {
                'success': final_state['error'] is None,
                'image': final_state['image'],
                'quality_score': final_state['quality_score'],
                'completed_tasks': final_state['completed_tasks'],
                'metadata': final_state['metadata'],
                'error': final_state['error']
            }
            
        except Exception as e:
            logger.error(f"Error in workflow execution: {str(e)}")
            return {
                'success': False,
                'image': image,
                'quality_score': 0.0,
                'completed_tasks': [],
                'metadata': {},
                'error': str(e)
            }
