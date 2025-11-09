"""
Animation Renderer

Renders animated mathematical art to video files.
"""

from typing import Optional, Union, Callable, List
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from pathlib import Path
from core.renderers.static_renderer import StaticRenderer
from core.generators.base import ArtGenerator
from core.color_maps import ColorMap, get_palette


class AnimationRenderer:
    """
    Render animated mathematical art.
    
    Supports parameter sweeps and time-based evolution.
    """
    
    def __init__(
        self,
        generator: Union[ArtGenerator, Callable],
        frames: int = 120,
        fps: int = 30,
        width: int = 1024,
        height: int = 1024,
        dpi: int = 100,
        colormap: Union[str, ColorMap] = "viridis",
        background: str = "black",
    ):
        """
        Initialize animation renderer.
        
        Args:
            generator: Generator object or callable that takes time parameter
            frames: Number of frames
            fps: Frames per second
            width: Frame width
            height: Frame height
            dpi: DPI
            colormap: Color mapping
            background: Background color
        """
        self.generator = generator
        self.frames = frames
        self.fps = fps
        self.width = width
        self.height = height
        self.dpi = dpi
        self.background = background
        
        if isinstance(colormap, str):
            self.colormap = get_palette(colormap)
        else:
            self.colormap = colormap
        
        self.static_renderer: Optional[StaticRenderer] = None
    
    def _generate_frame(self, t: float) -> NDArray:
        """
        Generate points for a single frame.
        
        Args:
            t: Time parameter (0 to 1)
            
        Returns:
            Points array
        """
        if isinstance(self.generator, ArtGenerator):
            # If generator has time-dependent method
            if hasattr(self.generator, 'generate') and hasattr(self.generator.generate, '__code__'):
                # Check if generate accepts time parameter
                code = self.generator.generate.__code__
                if 't' in code.co_varnames:
                    return self.generator.generate(t=t)
                else:
                    return self.generator.generate()
            else:
                return self.generator.generate()
        else:
            # Callable function
            return self.generator(t)
    
    def render_to_frames(
        self,
        output_dir: str,
        name_pattern: str = "frame_{:04d}.png"
    ):
        """
        Render animation as individual frame images.
        
        Args:
            output_dir: Output directory
            name_pattern: Filename pattern with frame number placeholder
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create renderer
        renderer = StaticRenderer(
            width=self.width,
            height=self.height,
            dpi=self.dpi,
            colormap=self.colormap,
            background=self.background
        )
        
        try:
            for i in range(self.frames):
                t = i / (self.frames - 1) if self.frames > 1 else 0
                
                # Generate points
                points = self._generate_frame(t)
                
                # Render
                renderer.clear()
                renderer.render(points)
                
                # Save
                filename = output_path / name_pattern.format(i)
                renderer.save(str(filename))
                
                print(f"Rendered frame {i+1}/{self.frames}")
        finally:
            renderer.close()
    
    def render_to_video(
        self,
        filename: str,
        codec: str = "libx264",
        bitrate: int = 5000,
        **kwargs
    ):
        """
        Render animation directly to video file.
        
        Args:
            filename: Output filename (MP4, etc.)
            codec: Video codec
            bitrate: Video bitrate
            **kwargs: Additional FFmpeg parameters
        """
        # Setup figure
        fig_width = self.width / self.dpi
        fig_height = self.height / self.dpi
        
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=self.dpi)
        ax = fig.add_subplot(111)
        
        fig.patch.set_facecolor(self.background)
        ax.set_facecolor(self.background)
        ax.axis('off')
        ax.set_aspect('equal')
        
        # Initialize with first frame
        points_0 = self._generate_frame(0)
        x_0, y_0 = points_0[:, 0], points_0[:, 1]
        colors_0 = self.colormap.map(np.linspace(0, 1, len(x_0)))
        
        scatter = ax.scatter(x_0, y_0, c=colors_0, s=1)
        
        # Set initial limits
        margin = 0.1
        x_range = x_0.max() - x_0.min()
        y_range = y_0.max() - y_0.min()
        ax.set_xlim(x_0.min() - margin * x_range, x_0.max() + margin * x_range)
        ax.set_ylim(y_0.min() - margin * y_range, y_0.max() + margin * y_range)
        
        def update(frame):
            """Update function for animation."""
            t = frame / (self.frames - 1) if self.frames > 1 else 0
            
            # Generate new points
            points = self._generate_frame(t)
            x, y = points[:, 0], points[:, 1]
            
            # Update scatter plot
            scatter.set_offsets(np.c_[x, y])
            colors = self.colormap.map(np.linspace(0, 1, len(x)))
            scatter.set_color(colors)
            
            return scatter,
        
        # Create animation
        anim = FuncAnimation(
            fig,
            update,
            frames=self.frames,
            interval=1000/self.fps,
            blit=True
        )
        
        # Save video
        try:
            if filename.endswith('.gif'):
                writer = PillowWriter(fps=self.fps)
            else:
                writer = FFMpegWriter(fps=self.fps, codec=codec, bitrate=bitrate, **kwargs)
            
            anim.save(filename, writer=writer)
            print(f"Saved animation to: {filename}")
        except Exception as e:
            print(f"Error saving animation: {e}")
            print("Note: FFmpeg must be installed for video export")
        finally:
            plt.close(fig)
    
    def render_to_gif(
        self,
        filename: str,
        optimize: bool = True,
        **kwargs
    ):
        """
        Render animation as GIF.
        
        Args:
            filename: Output filename
            optimize: Optimize GIF size
            **kwargs: Additional parameters
        """
        if not filename.endswith('.gif'):
            filename += '.gif'
        
        self.render_to_video(filename, **kwargs)


class ParameterSweepAnimator:
    """
    Animate by sweeping a parameter value.
    """
    
    def __init__(
        self,
        generator_class: type,
        param_name: str,
        param_values: Union[List, NDArray],
        fixed_params: dict = None,
        width: int = 1024,
        height: int = 1024,
        fps: int = 30,
        colormap: str = "viridis",
        background: str = "black",
    ):
        """
        Initialize parameter sweep animator.
        
        Args:
            generator_class: Generator class
            param_name: Parameter name to sweep
            param_values: Array of parameter values
            fixed_params: Fixed parameters
            width: Frame width
            height: Frame height
            fps: Frames per second
            colormap: Color mapping
            background: Background color
        """
        self.generator_class = generator_class
        self.param_name = param_name
        self.param_values = np.array(param_values)
        self.fixed_params = fixed_params or {}
        self.width = width
        self.height = height
        self.fps = fps
        self.colormap = colormap
        self.background = background
    
    def render(self, filename: str):
        """Render parameter sweep animation."""
        def generator_func(t: float) -> NDArray:
            # Interpolate parameter value
            idx = int(t * (len(self.param_values) - 1))
            idx = min(idx, len(self.param_values) - 1)
            param_value = self.param_values[idx]
            
            # Create generator with this parameter value
            params = self.fixed_params.copy()
            params[self.param_name] = param_value
            
            gen = self.generator_class(**params)
            return gen.generate()
        
        # Create animation renderer
        anim_renderer = AnimationRenderer(
            generator=generator_func,
            frames=len(self.param_values),
            fps=self.fps,
            width=self.width,
            height=self.height,
            colormap=self.colormap,
            background=self.background,
        )
        
        anim_renderer.render_to_video(filename)
