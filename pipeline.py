import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict, Any, Callable
from dataclasses import dataclass
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import to_tensor, normalize
import warnings
from contextlib import contextmanager
from functools import wraps

from transformers import PretrainedConfig, PreTrainedModel, CLIPTextModel, CLIPTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPooling
from diffusers import DiffusionPipeline, DDIMScheduler
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from .scheduling_omni import OmniScheduler

# Optimization imports
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe
    HAS_TRANSFORMER_ENGINE = True
except ImportError:
    HAS_TRANSFORMER_ENGINE = False

try:
    from torch._dynamo import config as dynamo_config
    HAS_TORCH_COMPILE = hasattr(torch, 'compile')
except ImportError:
    HAS_TORCH_COMPILE = False

# -----------------------------------------------------------------------------
# 1. Advanced Configuration (8B Scale)
# -----------------------------------------------------------------------------

class OmniMMDitV2Config(PretrainedConfig):
    model_type = "omnimm_dit_v2"
    
    def __init__(
        self,
        vocab_size: int = 49408,
        hidden_size: int = 4096,          # 4096 dim for ~7B-8B scale
        intermediate_size: int = 11008,   # Llama-style MLP expansion
        num_hidden_layers: int = 32,      # Deep network
        num_attention_heads: int = 32,
        num_key_value_heads: Optional[int] = 8,  # GQA (Grouped Query Attention)
        hidden_act: str = "silu",
        max_position_embeddings: int = 4096,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        # DiT Specifics
        patch_size: int = 2,
        in_channels: int = 4,             # VAE Latent channels
        out_channels: int = 4, # x2 for variance if learned
        frequency_embedding_size: int = 256,
        # Multi-Modal Specifics
        max_condition_images: int = 3,    # Support 1-3 input images
        visual_embed_dim: int = 1024,     # e.g., SigLIP or CLIP Vision
        text_embed_dim: int = 4096,       # T5-XXL or similar
        use_temporal_attention: bool = True, # For Video generation
        # Optimization Configs
        use_fp8_quantization: bool = False,
        use_compilation: bool = False,
        compile_mode: str = "reduce-overhead",
        use_flash_attention: bool = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.frequency_embedding_size = frequency_embedding_size
        self.max_condition_images = max_condition_images
        self.visual_embed_dim = visual_embed_dim
        self.text_embed_dim = text_embed_dim
        self.use_temporal_attention = use_temporal_attention
        self.use_fp8_quantization = use_fp8_quantization
        self.use_compilation = use_compilation
        self.compile_mode = compile_mode
        self.use_flash_attention = use_flash_attention
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

# -----------------------------------------------------------------------------
# 2. Professional Building Blocks (RoPE, SwiGLU, AdaLN)
# -----------------------------------------------------------------------------

class OmniRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class OmniRotaryEmbedding(nn.Module):
    """Complex implementation of Rotary Positional Embeddings for DiT"""
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, seq_len=None):
        t = torch.arange(seq_len or x.shape[1], device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

class OmniSwiGLU(nn.Module):
    """Swish-Gated Linear Unit for High-Performance FFN"""
    def __init__(self, config: OmniMMDitV2Config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TimestepEmbedder(nn.Module):
    """Fourier feature embedding for timesteps"""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(max_period)) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, dtype):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(dtype)
        return self.mlp(t_freq)

# -----------------------------------------------------------------------------
# 2.5. Data Processing Utilities
# -----------------------------------------------------------------------------

class OmniImageProcessor:
    """Advanced image preprocessing for multi-modal diffusion models"""
    
    def __init__(
        self,
        image_mean: List[float] = [0.485, 0.456, 0.406],
        image_std: List[float] = [0.229, 0.224, 0.225],
        size: Tuple[int, int] = (512, 512),
        interpolation: str = "bicubic",
        do_normalize: bool = True,
        do_center_crop: bool = False,
    ):
        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size
        self.do_normalize = do_normalize
        self.do_center_crop = do_center_crop
        
        # Build transform pipeline
        transforms_list = []
        if do_center_crop:
            transforms_list.append(T.CenterCrop(min(size)))
        
        interp_mode = {
            "bilinear": T.InterpolationMode.BILINEAR,
            "bicubic": T.InterpolationMode.BICUBIC,
            "lanczos": T.InterpolationMode.LANCZOS,
        }.get(interpolation, T.InterpolationMode.BICUBIC)
        
        transforms_list.append(T.Resize(size, interpolation=interp_mode, antialias=True))
        self.transform = T.Compose(transforms_list)
    
    def preprocess(
        self,
        images: Union[Image.Image, np.ndarray, torch.Tensor, List[Union[Image.Image, np.ndarray, torch.Tensor]]],
        return_tensors: str = "pt",
    ) -> torch.Tensor:
        """
        Preprocess images for model input.
        
        Args:
            images: Single image or list of images (PIL, numpy, or torch)
            return_tensors: Return type ("pt" for PyTorch)
            
        Returns:
            Preprocessed image tensor [B, C, H, W]
        """
        if not isinstance(images, list):
            images = [images]
        
        processed = []
        for img in images:
            # Convert to PIL if needed
            if isinstance(img, np.ndarray):
                if img.dtype == np.uint8:
                    img = Image.fromarray(img)
                else:
                    img = Image.fromarray((img * 255).astype(np.uint8))
            elif isinstance(img, torch.Tensor):
                img = T.ToPILImage()(img)
            
            # Apply transforms
            img = self.transform(img)
            
            # Convert to tensor
            if not isinstance(img, torch.Tensor):
                img = to_tensor(img)
            
            # Normalize
            if self.do_normalize:
                img = normalize(img, self.image_mean, self.image_std)
            
            processed.append(img)
        
        # Stack into batch
        if return_tensors == "pt":
            return torch.stack(processed, dim=0)
        
        return processed
    
    def postprocess(
        self,
        images: torch.Tensor,
        output_type: str = "pil",
    ) -> Union[List[Image.Image], np.ndarray, torch.Tensor]:
        """
        Postprocess model output to desired format.
        
        Args:
            images: Model output tensor [B, C, H, W]
            output_type: "pil", "np", or "pt"
            
        Returns:
            Processed images in requested format
        """
        # Denormalize if needed
        if self.do_normalize:
            mean = torch.tensor(self.image_mean).view(1, 3, 1, 1).to(images.device)
            std = torch.tensor(self.image_std).view(1, 3, 1, 1).to(images.device)
            images = images * std + mean
        
        # Clamp to valid range
        images = torch.clamp(images, 0, 1)
        
        if output_type == "pil":
            images = images.cpu().permute(0, 2, 3, 1).numpy()
            images = (images * 255).round().astype(np.uint8)
            return [Image.fromarray(img) for img in images]
        elif output_type == "np":
            return images.cpu().numpy()
        else:
            return images


class OmniVideoProcessor:
    """Video frame processing for temporal diffusion models"""
    
    def __init__(
        self,
        image_processor: OmniImageProcessor,
        num_frames: int = 16,
        frame_stride: int = 1,
    ):
        self.image_processor = image_processor
        self.num_frames = num_frames
        self.frame_stride = frame_stride
    
    def preprocess_video(
        self,
        video_frames: Union[List[Image.Image], np.ndarray, torch.Tensor],
        temporal_interpolation: bool = True,
    ) -> torch.Tensor:
        """
        Preprocess video frames for temporal model.
        
        Args:
            video_frames: List of PIL images, numpy array [T, H, W, C], or tensor [T, C, H, W]
            temporal_interpolation: Whether to interpolate to target frame count
            
        Returns:
            Preprocessed video tensor [B, C, T, H, W]
        """
        # Convert to list of PIL images
        if isinstance(video_frames, np.ndarray):
            if video_frames.ndim == 4:  # [T, H, W, C]
                video_frames = [Image.fromarray(frame) for frame in video_frames]
            else:
                raise ValueError(f"Expected 4D numpy array, got shape {video_frames.shape}")
        elif isinstance(video_frames, torch.Tensor):
            if video_frames.ndim == 4:  # [T, C, H, W]
                video_frames = [T.ToPILImage()(frame) for frame in video_frames]
            else:
                raise ValueError(f"Expected 4D tensor, got shape {video_frames.shape}")
        
        # Sample frames if needed
        total_frames = len(video_frames)
        if temporal_interpolation and total_frames != self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            video_frames = [video_frames[i] for i in indices]
        
        # Process each frame
        processed_frames = []
        for frame in video_frames[:self.num_frames]:
            frame_tensor = self.image_processor.preprocess(frame, return_tensors="pt")[0]
            processed_frames.append(frame_tensor)
        
        # Stack: [T, C, H, W] -> [1, C, T, H, W]
        video_tensor = torch.stack(processed_frames, dim=1).unsqueeze(0)
        return video_tensor
    
    def postprocess_video(
        self,
        video_tensor: torch.Tensor,
        output_type: str = "pil",
    ) -> Union[List[Image.Image], np.ndarray, torch.Tensor]:
        """
        Postprocess video output.
        
        Args:
            video_tensor: Model output [B, C, T, H, W] or [B, T, C, H, W]
            output_type: "pil", "np", or "pt"
            
        Returns:
            Processed video frames
        """
        # Normalize dimensions to [B, T, C, H, W]
        if video_tensor.ndim == 5:
            if video_tensor.shape[1] in [3, 4]:  # [B, C, T, H, W]
                video_tensor = video_tensor.permute(0, 2, 1, 3, 4)
        
        batch_size, num_frames = video_tensor.shape[:2]
        
        # Process each frame
        all_frames = []
        for b in range(batch_size):
            frames = []
            for t in range(num_frames):
                frame = video_tensor[b, t]  # [C, H, W]
                frame = frame.unsqueeze(0)  # [1, C, H, W]
                processed = self.image_processor.postprocess(frame, output_type=output_type)
                frames.extend(processed)
            all_frames.append(frames)
        
        return all_frames[0] if batch_size == 1 else all_frames


class OmniLatentProcessor:
    """VAE latent space encoding/decoding with scaling and normalization"""
    
    def __init__(
        self,
        vae: Any,
        scaling_factor: float = 0.18215,
        do_normalize_latents: bool = True,
    ):
        self.vae = vae
        self.scaling_factor = scaling_factor
        self.do_normalize_latents = do_normalize_latents
    
    @torch.no_grad()
    def encode(
        self,
        images: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = False,
    ) -> torch.Tensor:
        """
        Encode images to latent space.
        
        Args:
            images: Input images [B, C, H, W] in range [-1, 1]
            generator: Random generator for sampling
            return_dict: Whether to return dict or tensor
            
        Returns:
            Latent codes [B, 4, H//8, W//8]
        """
        # VAE expects input in [-1, 1]
        if images.min() >= 0:
            images = images * 2.0 - 1.0
        
        # Encode
        latent_dist = self.vae.encode(images).latent_dist
        latents = latent_dist.sample(generator=generator)
        
        # Scale latents
        latents = latents * self.scaling_factor
        
        # Additional normalization for stability
        if self.do_normalize_latents:
            latents = (latents - latents.mean()) / (latents.std() + 1e-6)
        
        return latents if not return_dict else {"latents": latents}
    
    @torch.no_grad()
    def decode(
        self,
        latents: torch.Tensor,
        return_dict: bool = False,
    ) -> torch.Tensor:
        """
        Decode latents to image space.
        
        Args:
            latents: Latent codes [B, 4, H//8, W//8]
            return_dict: Whether to return dict or tensor
            
        Returns:
            Decoded images [B, 3, H, W] in range [-1, 1]
        """
        # Denormalize if needed
        if self.do_normalize_latents:
            # Assume identity transform for simplicity in decoding
            pass
        
        # Unscale
        latents = latents / self.scaling_factor
        
        # Decode
        images = self.vae.decode(latents).sample
        
        return images if not return_dict else {"images": images}
    
    @torch.no_grad()
    def encode_video(
        self,
        video_frames: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Encode video frames to latent space.
        
        Args:
            video_frames: Input video [B, C, T, H, W] or [B, T, C, H, W]
            generator: Random generator
            
        Returns:
            Video latents [B, 4, T, H//8, W//8]
        """
        # Reshape to process frames independently
        if video_frames.shape[2] not in [3, 4]:  # [B, T, C, H, W]
            B, T, C, H, W = video_frames.shape
            video_frames = video_frames.reshape(B * T, C, H, W)
            
            # Encode
            latents = self.encode(video_frames, generator=generator)
            
            # Reshape back
            latents = latents.reshape(B, T, *latents.shape[1:])
            latents = latents.permute(0, 2, 1, 3, 4)  # [B, 4, T, H//8, W//8]
        else:  # [B, C, T, H, W]
            B, C, T, H, W = video_frames.shape
            video_frames = video_frames.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            
            latents = self.encode(video_frames, generator=generator)
            latents = latents.reshape(B, T, *latents.shape[1:])
            latents = latents.permute(0, 2, 1, 3, 4)
        
        return latents

# -----------------------------------------------------------------------------
# 3. Core Architecture: OmniMMDitBlock (3D-Attention + Modulation)
# -----------------------------------------------------------------------------

class OmniMMDitBlock(nn.Module):
    def __init__(self, config: OmniMMDitV2Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Self-Attention with QK-Norm
        self.norm1 = OmniRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = nn.MultiheadAttention(
            config.hidden_size, config.num_attention_heads, batch_first=True
        )
        
        self.q_norm = OmniRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = OmniRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Cross-Attention for multimodal fusion
        self.norm2 = OmniRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_attn = nn.MultiheadAttention(
            config.hidden_size, config.num_attention_heads, batch_first=True
        )

        # Feed-Forward Network with SwiGLU activation
        self.norm3 = OmniRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = OmniSwiGLU(config)

        # Adaptive Layer Normalization with zero initialization
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True)
        )

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        encoder_hidden_states: torch.Tensor, # Text embeddings
        visual_context: Optional[torch.Tensor], # Reference image embeddings
        timestep_emb: torch.Tensor,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        
        # AdaLN Modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(timestep_emb)[:, None].chunk(6, dim=-1)
        )

        # Self-Attention block
        normed_hidden = self.norm1(hidden_states)
        normed_hidden = normed_hidden * (1 + scale_msa) + shift_msa
        
        attn_output, _ = self.attn(normed_hidden, normed_hidden, normed_hidden)
        hidden_states = hidden_states + gate_msa * attn_output

        # Cross-Attention with multimodal conditioning
        if visual_context is not None:
             context = torch.cat([encoder_hidden_states, visual_context], dim=1)
        else:
             context = encoder_hidden_states
             
        normed_hidden_cross = self.norm2(hidden_states)
        cross_output, _ = self.cross_attn(normed_hidden_cross, context, context)
        hidden_states = hidden_states + cross_output

        # Feed-Forward block
        normed_ffn = self.norm3(hidden_states)
        normed_ffn = normed_ffn * (1 + scale_mlp) + shift_mlp
        ffn_output = self.ffn(normed_ffn)
        hidden_states = hidden_states + gate_mlp * ffn_output

        return hidden_states

# -----------------------------------------------------------------------------
# 4. The Model: OmniMMDitV2
# -----------------------------------------------------------------------------

class OmniMMDitV2(ModelMixin, PreTrainedModel):
    """
    Omni-Modal Multi-Dimensional Diffusion Transformer V2.
    Supports: Text-to-Image, Image-to-Image (Edit), Image-to-Video.
    """
    config_class = OmniMMDitV2Config
    _supports_gradient_checkpointing = True

    def __init__(self, config: OmniMMDitV2Config):
        super().__init__(config)
        self.config = config

        # Initialize optimizer for advanced features
        self.optimizer = ModelOptimizer(
            fp8_config=FP8Config(enabled=config.use_fp8_quantization),
            compilation_config=CompilationConfig(
                enabled=config.use_compilation,
                mode=config.compile_mode,
            ),
            mixed_precision_config=MixedPrecisionConfig(
                enabled=True,
                dtype="bfloat16",
            ),
        )

        # Input Latent Projection (Patchify)
        self.x_embedder = nn.Linear(config.in_channels * config.patch_size * config.patch_size, config.hidden_size, bias=True)
        
        # Time & Vector Embeddings
        self.t_embedder = TimestepEmbedder(config.hidden_size, config.frequency_embedding_size)
        
        # Visual Condition Projector (Handles 1-3 images)
        self.visual_projector = nn.Sequential(
            nn.Linear(config.visual_embed_dim, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
        # Positional Embeddings (Absolute + RoPE dynamically handled)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_position_embeddings, config.hidden_size), requires_grad=False)

        # Transformer Backbone
        self.blocks = nn.ModuleList([
            OmniMMDitBlock(config, i) for i in range(config.num_hidden_layers)
        ])
        
        # Final Layer (AdaLN-Zero + Linear)
        self.final_layer = nn.Sequential(
            OmniRMSNorm(config.hidden_size, eps=config.rms_norm_eps),
            nn.Linear(config.hidden_size, config.patch_size * config.patch_size * config.out_channels, bias=True)
        )
        
        self.initialize_weights()
        
        # Apply optimizations if enabled
        if config.use_fp8_quantization or config.use_compilation:
            self._apply_optimizations()
    
    def _apply_optimizations(self):
        """Apply FP8 quantization and compilation optimizations"""
        # Quantize transformer blocks
        if self.config.use_fp8_quantization:
            for i, block in enumerate(self.blocks):
                self.blocks[i] = self.optimizer.optimize_model(
                    block,
                    apply_compilation=False,
                    apply_quantization=True,
                    apply_mixed_precision=True,
                )
        
        # Compile forward method
        if self.config.use_compilation and HAS_TORCH_COMPILE:
            self.forward = torch.compile(
                self.forward,
                mode=self.config.compile_mode,
                dynamic=True,
            )

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def unpatchify(self, x, h, w):
        c = self.config.out_channels
        p = self.config.patch_size
        h_ = h // p
        w_ = w // p
        x = x.reshape(shape=(x.shape[0], h_, w_, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h, w))
        return imgs

    def forward(
        self,
        hidden_states: torch.Tensor, # Noisy Latents [B, C, H, W] or [B, C, F, H, W]
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor, # Text Embeddings
        visual_conditions: Optional[List[torch.Tensor]] = None, # List of [B, L, D]
        video_frames: Optional[int] = None, # If generating video
        return_dict: bool = True,
    ) -> Union[torch.Tensor, BaseOutput]:
        
        batch_size, channels, _, _ = hidden_states.shape
        
        # Patchify input latents
        p = self.config.patch_size
        h, w = hidden_states.shape[-2], hidden_states.shape[-1]
        x = hidden_states.unfold(2, p, p).unfold(3, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(batch_size, -1, channels * p * p)
        
        # Positional and temporal embeddings
        x = self.x_embedder(x)
        x = x + self.pos_embed[:, :x.shape[1], :]
        
        t = self.t_embedder(timestep, x.dtype)
        
        # Process visual conditioning
        visual_emb = None
        if visual_conditions is not None:
            concat_visuals = torch.cat(visual_conditions, dim=1)
            visual_emb = self.visual_projector(concat_visuals)

        # Transformer blocks
        for block in self.blocks:
            x = block(
                hidden_states=x, 
                encoder_hidden_states=encoder_hidden_states,
                visual_context=visual_emb,
                timestep_emb=t
            )

        # Output projection
        x = self.final_layer[0](x)
        x = self.final_layer[1](x)
        
        # Unpatchify to image space
        output = self.unpatchify(x, h, w)

        if not return_dict:
            return (output,)

        return BaseOutput(sample=output)

# -----------------------------------------------------------------------------
# 4.5 π-Flow Policy Network (coarse trajectory predictor)
# -----------------------------------------------------------------------------


class PiFlowPolicyNetwork(nn.Module):
    """
    Lightweight π-Flow policy network: predicts multi-step velocity trajectories in one forward pass for few-step sampling.
    Relies only on text/visual global aggregated features + time embeddings and outputs velocity fields matching latent shape.
    """

    def __init__(
        self,
        text_hidden_size: int,
        visual_embed_dim: int,
        latent_channels: int,
        hidden_size: int = 1024,
    ):
        super().__init__()
        self.text_proj = nn.Linear(text_hidden_size, hidden_size)
        self.vis_proj = nn.Linear(visual_embed_dim, hidden_size)
        self.time_proj = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, latent_channels),
        )
        self.latent_channels = latent_channels

    def forward(
        self,
        text_embeddings: torch.Tensor,
        visual_embeddings_list: Optional[List[torch.Tensor]],
        timesteps: torch.Tensor,
        latent_shape: torch.Size,
    ) -> torch.Tensor:
        """
        Args:
            text_embeddings: [B, L, D_txt]
            visual_embeddings_list: list of [B, L_vis, D_vis] or None
            timesteps: [S] step values in [0,1]
            latent_shape: target latent shape (B, C, ...)
        Returns:
            policy_velocities: [S, *latent_shape]
        """
        device = text_embeddings.device
        dtype = text_embeddings.dtype
        batch_size = text_embeddings.shape[0]

        text_ctx = text_embeddings.mean(dim=1)

        if visual_embeddings_list:
            vis_tokens = [v.mean(dim=1) for v in visual_embeddings_list]
            vis_ctx = torch.stack(vis_tokens, dim=0).mean(dim=0)
        else:
            vis_ctx = torch.zeros_like(text_ctx)

        txt_feat = self.text_proj(text_ctx)
        vis_feat = self.vis_proj(vis_ctx.to(device=device, dtype=dtype))

        time_feat = self.time_proj(timesteps.unsqueeze(-1).to(device=device, dtype=dtype))

        velocities = []
        for t_feat in time_feat:
            fused = torch.cat([txt_feat, vis_feat, t_feat.expand_as(txt_feat)], dim=-1)
            step_token = self.fuse(fused).tanh()  # [B, C]

            step_field = step_token
            while len(step_field.shape) < len(latent_shape):
                step_field = step_field.unsqueeze(-1)
            step_field = step_field.expand(batch_size, *latent_shape[1:])
            velocities.append(step_field)

        policy_velocities = torch.stack(velocities, dim=0)
        return policy_velocities

# -----------------------------------------------------------------------------
# 5. The "Fancy" Pipeline
# -----------------------------------------------------------------------------

class OmniMMDitV2Pipeline(DiffusionPipeline):
    """
    Omni-Modal Diffusion Transformer Pipeline.
    
    Supports text-guided image editing and video generation with
    multi-image conditioning and advanced guidance techniques.
    """
    model: OmniMMDitV2
    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel
    vae: Any # AutoencoderKL
    scheduler: Union[DDIMScheduler, OmniScheduler]
    
    _optional_components = ["visual_encoder"]

    def __init__(
        self,
        model: OmniMMDitV2,
        vae: Any,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        scheduler: Union[DDIMScheduler, OmniScheduler],
        visual_encoder: Optional[Any] = None,
    ):
        super().__init__()
        self.register_modules(
            model=model,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            visual_encoder=visual_encoder
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        
        # Initialize data processors
        self.image_processor = OmniImageProcessor(
            size=(512, 512),
            interpolation="bicubic",
            do_normalize=True,
        )
        self.video_processor = OmniVideoProcessor(
            image_processor=self.image_processor,
            num_frames=16,
        )
        self.latent_processor = OmniLatentProcessor(
            vae=vae,
            scaling_factor=0.18215,
        )
        
        # Initialize model optimizer
        self.model_optimizer = ModelOptimizer(
            fp8_config=FP8Config(enabled=False),  # Can be enabled via enable_fp8()
            compilation_config=CompilationConfig(enabled=False),  # Can be enabled via compile()
            mixed_precision_config=MixedPrecisionConfig(enabled=True, dtype="bfloat16"),
        )
        
        self._is_compiled = False
        self._is_fp8_enabled = False
        self.policy_network = PiFlowPolicyNetwork(
            text_hidden_size=self.text_encoder.config.hidden_size,
            visual_embed_dim=self.model.config.visual_embed_dim,
            latent_channels=self.model.config.in_channels,
            hidden_size=min(1024, self.model.config.hidden_size),
        )
    
    def enable_fp8_quantization(self):
        """Enable FP8 quantization for faster inference"""
        if not HAS_TRANSFORMER_ENGINE:
            warnings.warn("Transformer Engine not available. Install with: pip install transformer-engine")
            return self
        
        self.model_optimizer.fp8_config.enabled = True
        self.model = self.model_optimizer.optimize_model(
            self.model,
            apply_compilation=False,
            apply_quantization=True,
            apply_mixed_precision=False,
        )
        self._is_fp8_enabled = True
        return self
    
    def compile_model(
        self,
        mode: str = "reduce-overhead",
        fullgraph: bool = False,
        dynamic: bool = True,
    ):
        """
        Compile model using torch.compile for faster inference.
        
        Args:
            mode: Compilation mode - "default", "reduce-overhead", "max-autotune"
            fullgraph: Whether to compile the entire model as one graph
            dynamic: Whether to enable dynamic shapes
        """
        if not HAS_TORCH_COMPILE:
            warnings.warn("torch.compile not available. Upgrade to PyTorch 2.0+")
            return self
        
        self.model_optimizer.compilation_config = CompilationConfig(
            enabled=True,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
        
        self.model = self.model_optimizer._compile_model(self.model)
        self._is_compiled = True
        return self
    
    def enable_optimizations(
        self,
        enable_fp8: bool = False,
        enable_compilation: bool = False,
        compilation_mode: str = "reduce-overhead",
    ):
        """
        Enable all optimizations at once.
        
        Args:
            enable_fp8: Enable FP8 quantization
            enable_compilation: Enable torch.compile
            compilation_mode: Compilation mode for torch.compile
        """
        if enable_fp8:
            self.enable_fp8_quantization()
        
        if enable_compilation:
            self.compile_model(mode=compilation_mode)
        
        return self

    def _predict_policy_trajectory(
        self,
        text_embeddings: torch.Tensor,
        visual_embeddings: torch.Tensor,
        device: torch.device,
        total_steps: int,
    ) -> Optional[torch.Tensor]:
        """
        Predict coarse-stage velocity trajectory in one shot using the π-Flow policy network.
        """
        if self.policy_network is None or total_steps <= 0:
            return None
        # Keep policy network on the same device
        self.policy_network = self.policy_network.to(device=device, dtype=text_embeddings.dtype)
        time_grid = torch.linspace(0, 1, total_steps, device=device, dtype=text_embeddings.dtype)
        time_grid = torch.linspace(0, 1, total_steps, device=self.device, dtype=text_embeddings.dtype)
        return self.policy_network(
            text_embeddings=text_embeddings.detach(),
            visual_embeddings_list=visual_embeddings_list,
            timesteps=time_grid,
            latent_shape=latents.shape,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        input_images: Optional[List[Union[torch.Tensor, Any]]] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        num_frames: Optional[int] = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        use_optimized_inference: bool = True,
        use_pi_flow_policy: bool = False,
        **kwargs,
    ):
        # Use optimized inference context
        with optimized_inference_mode(
            enable_cudnn_benchmark=use_optimized_inference,
            enable_tf32=use_optimized_inference,
            enable_flash_sdp=use_optimized_inference,
        ):
            return self._forward_impl(
                prompt=prompt,
                input_images=input_images,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                image_guidance_scale=image_guidance_scale,
                negative_prompt=negative_prompt,
                eta=eta,
                generator=generator,
                latents=latents,
                output_type=output_type,
                return_dict=return_dict,
                callback=callback,
                callback_steps=callback_steps,
                use_pi_flow_policy=use_pi_flow_policy,
                **kwargs,
            )
    
    def _forward_impl(
        self,
        prompt: Union[str, List[str]] = None,
        input_images: Optional[List[Union[torch.Tensor, Any]]] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        num_frames: Optional[int] = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        use_pi_flow_policy: bool = False,
        **kwargs,
    ):
        # Validate and set default dimensions
        height = height or self.model.config.sample_size * self.vae_scale_factor
        width = width or self.model.config.sample_size * self.vae_scale_factor

        # Encode text prompts
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)
        
        text_inputs = self.tokenizer(
            prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.device))[0]

        # Encode visual conditions with preprocessing
        visual_embeddings_list = []
        if input_images:
            if not isinstance(input_images, list):
                input_images = [input_images]
            if len(input_images) > 3:
                raise ValueError("Maximum 3 reference images supported")
            
            for img in input_images:
                # Preprocess image
                if not isinstance(img, torch.Tensor):
                    img_tensor = self.image_processor.preprocess(img, return_tensors="pt")
                else:
                    img_tensor = img
                
                img_tensor = img_tensor.to(device=self.device, dtype=text_embeddings.dtype)
                
                # Encode with visual encoder
                if self.visual_encoder is not None:
                    vis_emb = self.visual_encoder(img_tensor).last_hidden_state
                else:
                    # Fallback: use VAE encoder + projection
                    with torch.no_grad():
                        latent_features = self.vae.encode(img_tensor * 2 - 1).latent_dist.mode()
                        B, C, H, W = latent_features.shape
                        # Flatten spatial dims and project
                        vis_emb = latent_features.flatten(2).transpose(1, 2)  # [B, H*W, C]
                        # Simple projection to visual_embed_dim
                        if vis_emb.shape[-1] != self.model.config.visual_embed_dim:
                            proj = nn.Linear(vis_emb.shape[-1], self.model.config.visual_embed_dim).to(self.device)
                            vis_emb = proj(vis_emb)
                
                visual_embeddings_list.append(vis_emb)

        # Prepare timesteps
        if isinstance(self.scheduler, OmniScheduler):
            # π-Flow / Flow Matching path
            self.scheduler.config.prediction_type = "velocity"
            self.scheduler.set_timesteps(num_inference_steps, device=self.device, use_karras_sigmas=True)
            total_steps = len(self.scheduler.timesteps) - 1
        else:
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
            timesteps = self.scheduler.timesteps
            total_steps = len(timesteps)

        # Initialize latent space
        num_channels_latents = self.model.config.in_channels
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if num_frames > 1:
            shape = (batch_size, num_channels_latents, num_frames, height // self.vae_scale_factor, width // self.vae_scale_factor)

        latents = torch.randn(shape, generator=generator, device=self.device, dtype=text_embeddings.dtype)
        latents = latents * self.scheduler.init_noise_sigma

        if isinstance(self.scheduler, OmniScheduler):
            policy_velocities = None
            if use_pi_flow_policy:
                policy_velocities = self._compute_policy_trajectory(
                    text_embeddings=text_embeddings,
                    visual_embeddings_list=visual_embeddings_list,
                    latents=latents,
                    total_steps=total_steps,
                )

            with self.progress_bar(total=total_steps) as progress_bar:
                for step_idx in range(total_steps):
                    t_val = self.scheduler.timesteps[step_idx]

                    use_policy_step = (
                        use_pi_flow_policy and policy_velocities is not None and step_idx < self.scheduler.coarse_steps
                    )

                    if use_policy_step:
                        model_output = policy_velocities[step_idx]
                        model_fn = None
                    else:
                        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t_val)

                        with self.model_optimizer.autocast_context():
                            noise_pred = self.model(
                                hidden_states=latent_model_input,
                                timestep=t_val,
                                encoder_hidden_states=torch.cat([text_embeddings] * 2),
                                visual_conditions=visual_embeddings_list * 2 if visual_embeddings_list else None,
                                video_frames=num_frames
                            ).sample

                        if guidance_scale > 1.0:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                        model_output = noise_pred
                        model_fn = None  # extendable to second eval for higher-order solvers

                    step_output = self.scheduler.step(
                        model_output=model_output,
                        timestep=step_idx,
                        sample=latents,
                        model_fn=model_fn,
                    )
                    latents = step_output.prev_sample if hasattr(step_output, "prev_sample") else step_output[0]

                    if callback is not None and step_idx % callback_steps == 0:
                        callback(step_idx, t_val, latents)

                    progress_bar.update()
        else:
            # Compatible with original DDIM/standard scheduler
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    
                    # Use mixed precision autocast
                    with self.model_optimizer.autocast_context():
                        noise_pred = self.model(
                            hidden_states=latent_model_input,
                            timestep=t,
                            encoder_hidden_states=torch.cat([text_embeddings] * 2),
                            visual_conditions=visual_embeddings_list * 2 if visual_embeddings_list else None,
                            video_frames=num_frames
                        ).sample

                    # Apply classifier-free guidance
                    if guidance_scale > 1.0:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                    latents = self.scheduler.step(noise_pred, t, latents, eta=eta).prev_sample
                    
                    # Call callback if provided
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
                    
                    progress_bar.update()

        # Decode latents with proper post-processing
        if output_type == "latent":
            output_images = latents
        else:
            # Decode latents to pixel space
            with torch.no_grad():
                if num_frames > 1:
                    # Video decoding: process frame by frame
                    B, C, T, H, W = latents.shape
                    latents_2d = latents.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
                    decoded = self.latent_processor.decode(latents_2d)
                    decoded = decoded.reshape(B, T, 3, H * 8, W * 8)
                    
                    # Convert to [0, 1] range
                    decoded = (decoded / 2 + 0.5).clamp(0, 1)
                    
                    # Post-process video
                    if output_type == "pil":
                        output_images = self.video_processor.postprocess_video(decoded, output_type="pil")
                    elif output_type == "np":
                        output_images = decoded.cpu().numpy()
                    else:
                        output_images = decoded
                else:
                    # Image decoding
                    decoded = self.latent_processor.decode(latents)
                    decoded = (decoded / 2 + 0.5).clamp(0, 1)
                    
                    # Post-process images
                    if output_type == "pil":
                        output_images = self.image_processor.postprocess(decoded, output_type="pil")
                    elif output_type == "np":
                        output_images = decoded.cpu().numpy()
                    else:
                        output_images = decoded

        if not return_dict:
            return (output_images,)

        return BaseOutput(images=output_images)

# -----------------------------------------------------------------------------
# 6. Advanced Multi-Modal Window Attention Block (Audio + Video + Image)
# -----------------------------------------------------------------------------

@dataclass
class MultiModalInput:
    """Container for multi-modal inputs"""
    image_embeds: Optional[torch.Tensor] = None      # [B, L_img, D]
    video_embeds: Optional[torch.Tensor] = None      # [B, T_video, L_vid, D]
    audio_embeds: Optional[torch.Tensor] = None      # [B, T_audio, L_aud, D]
    attention_mask: Optional[torch.Tensor] = None    # [B, total_length]


class TemporalWindowPartition(nn.Module):
    """
    Partition temporal sequences into windows for efficient attention.
    Supports both uniform and adaptive windowing strategies.
    """
    def __init__(
        self,
        window_size: int = 8,
        shift_size: int = 0,
        use_adaptive_window: bool = False,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_adaptive_window = use_adaptive_window
    
    def partition(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Partition sequence into windows.
        
        Args:
            x: Input tensor [B, T, L, D] or [B, L, D]
            
        Returns:
            windowed: [B * num_windows, window_size, L, D]
            info: Dictionary with partition information
        """
        if x.ndim == 3:  # Static input (image)
            return x, {"is_temporal": False, "original_shape": x.shape}
        
        B, T, L, D = x.shape
        
        # Apply temporal shift for shifted window attention (Swin-Transformer style)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=-self.shift_size, dims=1)
        
        # Pad if necessary
        pad_t = (self.window_size - T % self.window_size) % self.window_size
        if pad_t > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, pad_t))
        
        T_padded = T + pad_t
        num_windows = T_padded // self.window_size
        
        # Reshape into windows: [B, num_windows, window_size, L, D]
        x_windowed = x.view(B, num_windows, self.window_size, L, D)
        
        # Merge batch and window dims: [B * num_windows, window_size, L, D]
        x_windowed = x_windowed.view(B * num_windows, self.window_size, L, D)
        
        info = {
            "is_temporal": True,
            "original_shape": (B, T, L, D),
            "num_windows": num_windows,
            "pad_t": pad_t,
        }
        
        return x_windowed, info
    
    def merge(self, x_windowed: torch.Tensor, info: Dict[str, Any]) -> torch.Tensor:
        """
        Merge windows back to original sequence.
        
        Args:
            x_windowed: Windowed tensor [B * num_windows, window_size, L, D]
            info: Partition information from partition()
            
        Returns:
            x: Merged tensor [B, T, L, D] or [B, L, D]
        """
        if not info["is_temporal"]:
            return x_windowed
        
        B, T, L, D = info["original_shape"]
        num_windows = info["num_windows"]
        pad_t = info["pad_t"]
        
        # Reshape: [B * num_windows, window_size, L, D] -> [B, num_windows, window_size, L, D]
        x = x_windowed.view(B, num_windows, self.window_size, L, D)
        
        # Merge windows: [B, T_padded, L, D]
        x = x.view(B, num_windows * self.window_size, L, D)
        
        # Remove padding
        if pad_t > 0:
            x = x[:, :-pad_t, :, :]
        
        # Reverse temporal shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=self.shift_size, dims=1)
        
        return x


class WindowCrossAttention(nn.Module):
    """
    Window-based Cross Attention with support for temporal sequences.
    Performs attention within local windows for computational efficiency.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_relative_position_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        # QK Normalization for stability
        self.q_norm = OmniRMSNorm(self.head_dim)
        self.k_norm = OmniRMSNorm(self.head_dim)
        
        # Attention dropout
        self.attn_drop = nn.Dropout(attn_drop)
        
        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Relative position bias (for temporal coherence)
        self.use_relative_position_bias = use_relative_position_bias
        if use_relative_position_bias:
            # Temporal relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1), num_heads)
            )
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
            
            # Get relative position index
            coords = torch.arange(window_size)
            relative_coords = coords[:, None] - coords[None, :]  # [window_size, window_size]
            relative_coords += window_size - 1  # Shift to start from 0
            self.register_buffer("relative_position_index", relative_coords)
    
    def get_relative_position_bias(self, window_size: int) -> torch.Tensor:
        """Generate relative position bias for attention"""
        if not self.use_relative_position_bias:
            return None
        
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:window_size, :window_size].reshape(-1)
        ].reshape(window_size, window_size, -1)
        
        # Permute to [num_heads, window_size, window_size]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias
    
    def forward(
        self,
        query: torch.Tensor,      # [B, T_q, L_q, D] or [B, L_q, D]
        key: torch.Tensor,        # [B, T_k, L_k, D] or [B, L_k, D]
        value: torch.Tensor,      # [B, T_v, L_v, D] or [B, L_v, D]
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform windowed cross attention.
        
        Args:
            query: Query tensor
            key: Key tensor  
            value: Value tensor
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor with same shape as query
        """
        # Handle both temporal and non-temporal inputs
        is_temporal = query.ndim == 4
        
        if is_temporal:
            B, T_q, L_q, D = query.shape
            _, T_k, L_k, _ = key.shape
            
            # Flatten temporal and spatial dims for cross attention
            query_flat = query.reshape(B, T_q * L_q, D)
            key_flat = key.reshape(B, T_k * L_k, D)
            value_flat = value.reshape(B, T_k * L_k, D)
        else:
            B, L_q, D = query.shape
            _, L_k, _ = key.shape
            query_flat = query
            key_flat = key
            value_flat = value
        
        # Project to Q, K, V
        q = self.q_proj(query_flat)  # [B, N_q, D]
        k = self.k_proj(key_flat)    # [B, N_k, D]
        v = self.v_proj(value_flat)  # [B, N_v, D]
        
        # Reshape for multi-head attention
        q = q.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N_q, head_dim]
        k = k.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N_k, head_dim]
        v = v.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N_v, head_dim]
        
        # Apply QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N_q, N_k]
        
        # Add relative position bias if temporal
        if is_temporal and self.use_relative_position_bias:
            # Apply per-window bias
            rel_bias = self.get_relative_position_bias(min(T_q, self.window_size))
            if rel_bias is not None:
                # Broadcast bias across spatial dimensions
                attn = attn + rel_bias.unsqueeze(0).unsqueeze(2)
        
        # Apply attention mask
        if attention_mask is not None:
            attn = attn.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        
        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, -1, D)  # [B, N_q, D]
        
        # Output projection
        out = self.proj(out)
        out = self.proj_drop(out)
        
        # Reshape back to original shape
        if is_temporal:
            out = out.reshape(B, T_q, L_q, D)
        else:
            out = out.reshape(B, L_q, D)
        
        return out


class MultiModalFusionLayer(nn.Module):
    """
    Fuses multiple modalities (audio, video, image) with learnable fusion weights.
    """
    def __init__(
        self,
        dim: int,
        num_modalities: int = 3,
        fusion_type: str = "weighted",  # "weighted", "gated", "adaptive"
    ):
        super().__init__()
        self.dim = dim
        self.num_modalities = num_modalities
        self.fusion_type = fusion_type
        
        if fusion_type == "weighted":
            # Learnable fusion weights
            self.fusion_weights = nn.Parameter(torch.ones(num_modalities) / num_modalities)
        
        elif fusion_type == "gated":
            # Gated fusion with cross-modal interactions
            self.gate_proj = nn.Sequential(
                nn.Linear(dim * num_modalities, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, num_modalities),
                nn.Softmax(dim=-1)
            )
        
        elif fusion_type == "adaptive":
            # Adaptive fusion with per-token gating
            self.adaptive_gate = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.GELU(),
                nn.Linear(dim // 2, num_modalities),
                nn.Sigmoid()
            )
    
    def forward(self, modality_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse multiple modality features.
        
        Args:
            modality_features: List of [B, L, D] tensors for each modality
            
        Returns:
            fused: Fused features [B, L, D]
        """
        if self.fusion_type == "weighted":
            # Simple weighted sum
            weights = F.softmax(self.fusion_weights, dim=0)
            fused = sum(w * feat for w, feat in zip(weights, modality_features))
        
        elif self.fusion_type == "gated":
            # Concatenate and compute gates
            concat_features = torch.cat(modality_features, dim=-1)  # [B, L, D * num_modalities]
            gates = self.gate_proj(concat_features)  # [B, L, num_modalities]
            
            # Apply gates
            stacked = torch.stack(modality_features, dim=-1)  # [B, L, D, num_modalities]
            fused = (stacked * gates.unsqueeze(2)).sum(dim=-1)  # [B, L, D]
        
        elif self.fusion_type == "adaptive":
            # Adaptive per-token fusion
            fused_list = []
            for feat in modality_features:
                gate = self.adaptive_gate(feat)  # [B, L, num_modalities]
                fused_list.append(feat.unsqueeze(-1) * gate.unsqueeze(2))
            
            fused = torch.cat(fused_list, dim=-1).sum(dim=-1)  # [B, L, D]
        
        return fused


class FancyMultiModalWindowAttentionBlock(nn.Module):
    """
    🎯 Fancy Multi-Modal Window Attention Block
    
    A state-of-the-art block that processes audio, video, and image embeddings
    with temporal window-based cross-attention for efficient multi-modal fusion.
    
    Features:
    - ✨ Temporal windowing for audio and video (frame-by-frame processing)
    - 🪟 Shifted window attention for better temporal coherence (Swin-style)
    - 🔄 Cross-modal attention between all modality pairs
    - 🎭 Adaptive multi-modal fusion with learnable gates
    - 🚀 Efficient computation with window partitioning
    - 💎 QK normalization for training stability
    
    Architecture:
        1. Temporal Partitioning (audio/video frames → windows)
        2. Intra-Modal Self-Attention (within each modality)
        3. Inter-Modal Cross-Attention (audio ↔ video ↔ image)
        4. Multi-Modal Fusion (adaptive weighted combination)
        5. Feed-Forward Network (SwiGLU activation)
        6. Window Merging (reconstruct temporal sequences)
    """
    
    def __init__(
        self,
        dim: int = 1024,
        num_heads: int = 16,
        window_size: int = 8,
        shift_size: int = 4,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.1,
        use_relative_position_bias: bool = True,
        fusion_type: str = "adaptive",  # "weighted", "gated", "adaptive"
        use_shifted_window: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size if use_shifted_window else 0
        self.mlp_ratio = mlp_ratio
        
        # =============== Temporal Window Partitioning ===============
        self.window_partition = TemporalWindowPartition(
            window_size=window_size,
            shift_size=self.shift_size,
        )
        
        # =============== Intra-Modal Self-Attention ===============
        self.norm_audio_self = OmniRMSNorm(dim)
        self.norm_video_self = OmniRMSNorm(dim)
        self.norm_image_self = OmniRMSNorm(dim)
        
        self.audio_self_attn = WindowCrossAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_relative_position_bias=use_relative_position_bias,
        )
        
        self.video_self_attn = WindowCrossAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_relative_position_bias=use_relative_position_bias,
        )
        
        self.image_self_attn = WindowCrossAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_relative_position_bias=False,  # No temporal bias for static images
        )
        
        # =============== Inter-Modal Cross-Attention ===============
        # Audio → Video/Image
        self.norm_audio_cross = OmniRMSNorm(dim)
        self.audio_to_visual = WindowCrossAttention(
            dim=dim, num_heads=num_heads, window_size=window_size,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
        )
        
        # Video → Audio/Image
        self.norm_video_cross = OmniRMSNorm(dim)
        self.video_to_others = WindowCrossAttention(
            dim=dim, num_heads=num_heads, window_size=window_size,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
        )
        
        # Image → Audio/Video
        self.norm_image_cross = OmniRMSNorm(dim)
        self.image_to_temporal = WindowCrossAttention(
            dim=dim, num_heads=num_heads, window_size=window_size,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
        )
        
        # =============== Multi-Modal Fusion ===============
        self.multimodal_fusion = MultiModalFusionLayer(
            dim=dim,
            num_modalities=3,
            fusion_type=fusion_type,
        )
        
        # =============== Feed-Forward Network ===============
        self.norm_ffn = OmniRMSNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim, bias=False),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim, bias=False),
            nn.Dropout(drop),
        )
        
        # =============== Stochastic Depth (Drop Path) ===============
        self.drop_path = nn.Identity() if drop_path <= 0. else nn.Dropout(drop_path)
        
        # =============== Output Projections ===============
        self.output_projection = nn.ModuleDict({
            'audio': nn.Linear(dim, dim),
            'video': nn.Linear(dim, dim),
            'image': nn.Linear(dim, dim),
        })
    
    def forward(
        self,
        audio_embeds: Optional[torch.Tensor] = None,  # [B, T_audio, L_audio, D]
        video_embeds: Optional[torch.Tensor] = None,  # [B, T_video, L_video, D]
        image_embeds: Optional[torch.Tensor] = None,  # [B, L_image, D]
        attention_mask: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the Fancy Multi-Modal Window Attention Block.
        
        Args:
            audio_embeds: Audio embeddings [B, T_audio, L_audio, D]
                         T_audio: number of audio frames
                         L_audio: sequence length per frame
            video_embeds: Video embeddings [B, T_video, L_video, D]
                         T_video: number of video frames
                         L_video: sequence length per frame (e.g., patches)
            image_embeds: Image embeddings [B, L_image, D]
                         L_image: sequence length (e.g., image patches)
            attention_mask: Optional attention mask
            return_intermediates: Whether to return intermediate features
            
        Returns:
            outputs: Dictionary containing processed embeddings for each modality
                - 'audio': [B, T_audio, L_audio, D]
                - 'video': [B, T_video, L_video, D]
                - 'image': [B, L_image, D]
                - 'fused': [B, L_total, D] (optional)
        """
        intermediates = {} if return_intermediates else None
        
        # ========== Stage 1: Temporal Window Partitioning ==========
        partitioned_audio, audio_info = None, None
        partitioned_video, video_info = None, None
        
        if audio_embeds is not None:
            partitioned_audio, audio_info = self.window_partition.partition(audio_embeds)
            if return_intermediates:
                intermediates['audio_windows'] = partitioned_audio
        
        if video_embeds is not None:
            partitioned_video, video_info = self.window_partition.partition(video_embeds)
            if return_intermediates:
                intermediates['video_windows'] = partitioned_video
        
        # ========== Stage 2: Intra-Modal Self-Attention ==========
        audio_self_out, video_self_out, image_self_out = None, None, None
        
        if audio_embeds is not None:
            audio_normed = self.norm_audio_self(partitioned_audio)
            audio_self_out = self.audio_self_attn(audio_normed, audio_normed, audio_normed)
            audio_self_out = partitioned_audio + self.drop_path(audio_self_out)
        
        if video_embeds is not None:
            video_normed = self.norm_video_self(partitioned_video)
            video_self_out = self.video_self_attn(video_normed, video_normed, video_normed)
            video_self_out = partitioned_video + self.drop_path(video_self_out)
        
        if image_embeds is not None:
            image_normed = self.norm_image_self(image_embeds)
            image_self_out = self.image_self_attn(image_normed, image_normed, image_normed)
            image_self_out = image_embeds + self.drop_path(image_self_out)
        
        # ========== Stage 3: Inter-Modal Cross-Attention ==========
        audio_cross_out, video_cross_out, image_cross_out = None, None, None
        
        # Prepare context (merge windows temporarily for cross-attention)
        if audio_self_out is not None:
            audio_merged = self.window_partition.merge(audio_self_out, audio_info)
        if video_self_out is not None:
            video_merged = self.window_partition.merge(video_self_out, video_info)
        
        # Audio attends to Video and Image
        if audio_embeds is not None:
            audio_q = self.norm_audio_cross(audio_merged)
            
            # Create key-value context from other modalities
            kv_list = []
            if video_embeds is not None:
                kv_list.append(video_merged)
            if image_embeds is not None:
                # Expand image to match temporal dimension
                B, L_img, D = image_self_out.shape
                T_audio = audio_merged.shape[1]
                image_expanded = image_self_out.unsqueeze(1).expand(B, T_audio, L_img, D)
                kv_list.append(image_expanded)
            
            if kv_list:
                # Concatenate along sequence dimension
                kv_context = torch.cat([kv.flatten(1, 2) for kv in kv_list], dim=1)
                kv_context = kv_context.reshape(B, -1, D)
                
                audio_cross_out = self.audio_to_visual(
                    audio_q.flatten(1, 2),
                    kv_context,
                    kv_context,
                    attention_mask
                )
                audio_cross_out = audio_cross_out.reshape_as(audio_merged)
                audio_cross_out = audio_merged + self.drop_path(audio_cross_out)
            else:
                audio_cross_out = audio_merged
        
        # Video attends to Audio and Image
        if video_embeds is not None:
            video_q = self.norm_video_cross(video_merged)
            
            kv_list = []
            if audio_embeds is not None:
                kv_list.append(audio_merged if audio_cross_out is None else audio_cross_out)
            if image_embeds is not None:
                B, L_img, D = image_self_out.shape
                T_video = video_merged.shape[1]
                image_expanded = image_self_out.unsqueeze(1).expand(B, T_video, L_img, D)
                kv_list.append(image_expanded)
            
            if kv_list:
                kv_context = torch.cat([kv.flatten(1, 2) for kv in kv_list], dim=1)
                kv_context = kv_context.reshape(B, -1, D)
                
                video_cross_out = self.video_to_others(
                    video_q.flatten(1, 2),
                    kv_context,
                    kv_context,
                    attention_mask
                )
                video_cross_out = video_cross_out.reshape_as(video_merged)
                video_cross_out = video_merged + self.drop_path(video_cross_out)
            else:
                video_cross_out = video_merged
        
        # Image attends to Audio and Video
        if image_embeds is not None:
            image_q = self.norm_image_cross(image_self_out)
            
            kv_list = []
            if audio_embeds is not None:
                # Average pool audio over time for image
                audio_pooled = (audio_merged if audio_cross_out is None else audio_cross_out).mean(dim=1)
                kv_list.append(audio_pooled)
            if video_embeds is not None:
                # Average pool video over time for image
                video_pooled = (video_merged if video_cross_out is None else video_cross_out).mean(dim=1)
                kv_list.append(video_pooled)
            
            if kv_list:
                kv_context = torch.cat(kv_list, dim=1)
                
                image_cross_out = self.image_to_temporal(
                    image_q,
                    kv_context,
                    kv_context,
                    attention_mask
                )
                image_cross_out = image_self_out + self.drop_path(image_cross_out)
            else:
                image_cross_out = image_self_out
        
        # ========== Stage 4: Multi-Modal Fusion ==========
        # Collect features from all modalities for fusion
        fusion_features = []
        if audio_cross_out is not None:
            audio_flat = audio_cross_out.flatten(1, 2)  # [B, T*L, D]
            fusion_features.append(audio_flat)
        if video_cross_out is not None:
            video_flat = video_cross_out.flatten(1, 2)  # [B, T*L, D]
            fusion_features.append(video_flat)
        if image_cross_out is not None:
            fusion_features.append(image_cross_out)  # [B, L, D]
        
        # Pad/align sequence lengths for fusion
        if len(fusion_features) > 1:
            max_len = max(f.shape[1] for f in fusion_features)
            aligned_features = []
            for feat in fusion_features:
                if feat.shape[1] < max_len:
                    pad_len = max_len - feat.shape[1]
                    feat = F.pad(feat, (0, 0, 0, pad_len))
                aligned_features.append(feat)
            
            # Fuse modalities
            fused_features = self.multimodal_fusion(aligned_features)
        else:
            fused_features = fusion_features[0] if fusion_features else None
        
        # ========== Stage 5: Feed-Forward Network ==========
        if fused_features is not None:
            fused_normed = self.norm_ffn(fused_features)
            fused_ffn = self.ffn(fused_normed)
            fused_features = fused_features + self.drop_path(fused_ffn)
        
        # ========== Stage 6: Prepare Outputs ==========
        outputs = {}
        
        # Project back to original shapes
        if audio_embeds is not None and audio_cross_out is not None:
            # Partition again for consistency
            audio_final, _ = self.window_partition.partition(audio_cross_out)
            audio_final = self.output_projection['audio'](audio_final)
            audio_final = self.window_partition.merge(audio_final, audio_info)
            outputs['audio'] = audio_final
        
        if video_embeds is not None and video_cross_out is not None:
            video_final, _ = self.window_partition.partition(video_cross_out)
            video_final = self.output_projection['video'](video_final)
            video_final = self.window_partition.merge(video_final, video_info)
            outputs['video'] = video_final
        
        if image_embeds is not None and image_cross_out is not None:
            image_final = self.output_projection['image'](image_cross_out)
            outputs['image'] = image_final
        
        if fused_features is not None:
            outputs['fused'] = fused_features
        
        if return_intermediates:
            outputs['intermediates'] = intermediates
        
        return outputs


# -----------------------------------------------------------------------------
# 7. Optimization Utilities (FP8, Compilation, Mixed Precision)
# -----------------------------------------------------------------------------

@dataclass
class FP8Config:
    """Configuration for FP8 quantization"""
    enabled: bool = False
    margin: int = 0
    fp8_format: str = "hybrid"  # "e4m3", "e5m2", "hybrid"
    amax_history_len: int = 1024
    amax_compute_algo: str = "max"


@dataclass  
class CompilationConfig:
    """Configuration for torch.compile"""
    enabled: bool = False
    mode: str = "reduce-overhead"  # "default", "reduce-overhead", "max-autotune"
    fullgraph: bool = False
    dynamic: bool = True
    backend: str = "inductor"


@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision training/inference"""
    enabled: bool = True
    dtype: str = "bfloat16"  # "float16", "bfloat16"
    use_amp: bool = True


class ModelOptimizer:
    """
    Unified model optimizer supporting FP8 quantization, torch.compile, 
    and mixed precision inference.
    """
    def __init__(
        self,
        fp8_config: Optional[FP8Config] = None,
        compilation_config: Optional[CompilationConfig] = None,
        mixed_precision_config: Optional[MixedPrecisionConfig] = None,
    ):
        self.fp8_config = fp8_config or FP8Config()
        self.compilation_config = compilation_config or CompilationConfig()
        self.mixed_precision_config = mixed_precision_config or MixedPrecisionConfig()
        
        # Setup mixed precision
        self._setup_mixed_precision()
    
    def _setup_mixed_precision(self):
        """Setup mixed precision context"""
        if self.mixed_precision_config.enabled:
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            self.dtype = dtype_map.get(self.mixed_precision_config.dtype, torch.bfloat16)
        else:
            self.dtype = torch.float32
    
    @contextmanager
    def autocast_context(self):
        """Context manager for automatic mixed precision"""
        if self.mixed_precision_config.enabled and self.mixed_precision_config.use_amp:
            with torch.autocast(device_type='cuda', dtype=self.dtype):
                yield
        else:
            yield
    
    def _compile_model(self, model: nn.Module) -> nn.Module:
        """Compile model using torch.compile"""
        if not self.compilation_config.enabled or not HAS_TORCH_COMPILE:
            return model
        
        return torch.compile(
            model,
            mode=self.compilation_config.mode,
            fullgraph=self.compilation_config.fullgraph,
            dynamic=self.compilation_config.dynamic,
            backend=self.compilation_config.backend,
        )
    
    def _quantize_model_fp8(self, model: nn.Module) -> nn.Module:
        """Apply FP8 quantization using Transformer Engine"""
        if not self.fp8_config.enabled or not HAS_TRANSFORMER_ENGINE:
            return model
        
        # Convert compatible layers to FP8
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Replace with TE FP8 Linear
                fp8_linear = te.Linear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                )
                # Copy weights
                fp8_linear.weight.data.copy_(module.weight.data)
                if module.bias is not None:
                    fp8_linear.bias.data.copy_(module.bias.data)
                
                # Replace module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                if parent_name:
                    parent = dict(model.named_modules())[parent_name]
                    setattr(parent, child_name, fp8_linear)
        
        return model
    
    def optimize_model(
        self,
        model: nn.Module,
        apply_compilation: bool = True,
        apply_quantization: bool = True,
        apply_mixed_precision: bool = True,
    ) -> nn.Module:
        """
        Apply all optimizations to model.
        
        Args:
            model: Model to optimize
            apply_compilation: Whether to compile with torch.compile
            apply_quantization: Whether to apply FP8 quantization
            apply_mixed_precision: Whether to convert to mixed precision dtype
            
        Returns:
            Optimized model
        """
        # Apply FP8 quantization first
        if apply_quantization and self.fp8_config.enabled:
            model = self._quantize_model_fp8(model)
        
        # Convert to mixed precision dtype
        if apply_mixed_precision and self.mixed_precision_config.enabled:
            model = model.to(dtype=self.dtype)
        
        # Compile model last
        if apply_compilation and self.compilation_config.enabled:
            model = self._compile_model(model)
        
        return model


@contextmanager
def optimized_inference_mode(
    enable_cudnn_benchmark: bool = True,
    enable_tf32: bool = True,
    enable_flash_sdp: bool = True,
):
    """
    Context manager for optimized inference with various PyTorch optimizations.
    
    Args:
        enable_cudnn_benchmark: Enable cuDNN autotuner
        enable_tf32: Enable TF32 for faster matmul on Ampere+ GPUs
        enable_flash_sdp: Enable Flash Attention in scaled_dot_product_attention
    """
    # Save original states
    orig_benchmark = torch.backends.cudnn.benchmark
    orig_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    orig_tf32_cudnn = torch.backends.cudnn.allow_tf32
    orig_sdp_flash = torch.backends.cuda.flash_sdp_enabled()
    
    try:
        # Enable optimizations
        torch.backends.cudnn.benchmark = enable_cudnn_benchmark
        torch.backends.cuda.matmul.allow_tf32 = enable_tf32
        torch.backends.cudnn.allow_tf32 = enable_tf32
        
        if enable_flash_sdp:
            torch.backends.cuda.enable_flash_sdp(True)
        
        yield
        
    finally:
        # Restore original states
        torch.backends.cudnn.benchmark = orig_benchmark
        torch.backends.cuda.matmul.allow_tf32 = orig_tf32_matmul
        torch.backends.cudnn.allow_tf32 = orig_tf32_cudnn
        torch.backends.cuda.enable_flash_sdp(orig_sdp_flash)
