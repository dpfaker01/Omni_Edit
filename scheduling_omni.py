# Copyright 2025 OmniEdit Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
OmniScheduler: Unified Hybrid Flow-Diffusion Sampler

Key features:
1. Policy-driven velocity field inspired by π-Flow for few-step generation.
2. Multi-stage sampling (coarse → refine) to balance speed and detail.
3. High-order ODE solvers (RK2 / RK4) for faster convergence.
4. Hybrid flow-matching + diffusion sampling for stable trajectories.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from ..utils.torch_utils import randn_tensor
from .scheduling_utils import SchedulerMixin, SchedulerOutput


@dataclass
class OmniSchedulerOutput(BaseOutput):
    """
    Output container for OmniScheduler.

    Args:
        prev_sample (`torch.Tensor`):
            Sample at the previous timestep (x_{t-1}) for the next denoising step.
        prev_sample_mean (`torch.Tensor`):
            Mean estimate of the sample for inspection/debugging.
        velocity (`torch.Tensor`, *optional*):
            Policy-predicted velocity field for flow matching.
        stage (`int`):
            Current stage (0: coarse, 1: refine).
    """

    prev_sample: torch.Tensor
    prev_sample_mean: torch.Tensor
    velocity: Optional[torch.Tensor] = None
    stage: int = 0


class OmniScheduler(SchedulerMixin, ConfigMixin):
    """
    `OmniScheduler` - Unified Hybrid Flow-Diffusion Sampler

    Combines flow matching and high-order ODE solvers to achieve high-quality
    image/video generation in very few steps. Supports T2I, I2I, and T2V in one sampler.

    Key innovations:
    - Policy-driven velocity field: predicts the whole path in one forward pass.
    - Multi-stage sampling: coarse generation + optional refinement.
    - High-order ODE solvers (RK4/RK2) for faster convergence.
    - Hybrid flow/diffusion sampling for stable trajectories.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            Number of diffusion training steps.
        num_inference_steps (`int`, defaults to 4):
            Inference steps; supports few-step (4–8) generation.
        sigma_min (`float`, defaults to 0.002):
            Minimum noise level.
        sigma_max (`float`, defaults to 80.0):
            Maximum noise level.
        sigma_data (`float`, defaults to 0.5):
            Data std for preconditioning.
        rho (`float`, defaults to 7.0):
            Karras schedule parameter.
        solver_order (`int`, defaults to 2):
            Solver order (1: Euler, 2: RK2/Heun, 4: RK4).
        use_flow_matching (`bool`, defaults to True):
            Whether to use flow-matching mode.
        use_multi_stage (`bool`, defaults to True):
            Whether to use multi-stage sampling.
        coarse_ratio (`float`, defaults to 0.7):
            Fraction of steps allocated to coarse stage.
        snr (`float`, defaults to 0.15):
            SNR factor for correction step size.
        prediction_type (`str`, defaults to "velocity"):
            Prediction type ("velocity", "epsilon", "sample").
    """

    order = 2  # Default to 2nd-order solver

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 4,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        rho: float = 7.0,
        solver_order: int = 2,
        use_flow_matching: bool = True,
        use_multi_stage: bool = True,
        coarse_ratio: float = 0.7,
        snr: float = 0.15,
        prediction_type: str = "velocity",
    ):
        # Initial noise sigma
        self.init_noise_sigma = sigma_max

        # Mutable state
        self.timesteps = None
        self.sigmas = None
        self.discrete_sigmas = None
        self.num_inference_steps = num_inference_steps
        
        # Multi-stage state
        self.current_stage = 0
        self.coarse_steps = int(num_inference_steps * coarse_ratio)
        self.refine_steps = num_inference_steps - self.coarse_steps

        # Initialize timesteps and sigmas
        self._init_timesteps_and_sigmas()

    def _init_timesteps_and_sigmas(self):
        """Initialize timesteps and sigma values with Karras schedule."""
        num_steps = self.config.num_inference_steps
        sigma_min = self.config.sigma_min
        sigma_max = self.config.sigma_max
        rho = self.config.rho

        # Karras sigma schedule: σ_i = (σ_max^(1/ρ) + i/(N-1) * (σ_min^(1/ρ) - σ_max^(1/ρ)))^ρ
        ramp = torch.linspace(0, 1, num_steps + 1)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        self.sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        
        # Append final sigma = 0
        self.sigmas = torch.cat([self.sigmas, torch.zeros(1)])
        
        # Timesteps from 1 to 0
        self.timesteps = torch.linspace(1, 0, num_steps + 1)
        
        # Discrete sigmas for compatibility
        self.discrete_sigmas = self.sigmas[:-1]

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        """
        Precondition model input with sigma-dependent scaling.

        Args:
            sample (`torch.Tensor`): Input sample.
            timestep (`int`, *optional*): Current timestep.

        Returns:
            `torch.Tensor`: Scaled input sample.
        """
        if timestep is None:
            return sample
            
        # Fetch current sigma
        step_index = (self.timesteps == timestep).nonzero()
        if len(step_index) == 0:
            return sample
        sigma = self.sigmas[step_index[0]].to(sample.device)
        
        # Preconditioning scale: c_in = 1 / sqrt(σ² + σ_data²)
        sigma_data = self.config.sigma_data
        c_in = 1 / (sigma**2 + sigma_data**2).sqrt()
        
        return sample * c_in

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = None,
        use_karras_sigmas: bool = True,
    ):
        """
        Set inference timesteps.

        Args:
            num_inference_steps (`int`): Number of inference steps.
            device (`str` or `torch.device`, *optional*): Target device.
            use_karras_sigmas (`bool`): Whether to use Karras sigma schedule.
        """
        self.num_inference_steps = num_inference_steps
        self.coarse_steps = int(num_inference_steps * self.config.coarse_ratio)
        self.refine_steps = num_inference_steps - self.coarse_steps
        
        sigma_min = self.config.sigma_min
        sigma_max = self.config.sigma_max
        rho = self.config.rho

        if use_karras_sigmas:
            # Karras sigma schedule
            ramp = torch.linspace(0, 1, num_inference_steps + 1)
            min_inv_rho = sigma_min ** (1 / rho)
            max_inv_rho = sigma_max ** (1 / rho)
            self.sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        else:
            # Linear sigma schedule
            self.sigmas = torch.linspace(sigma_max, sigma_min, num_inference_steps + 1)

        self.sigmas = torch.cat([self.sigmas, torch.zeros(1)])
        self.timesteps = torch.linspace(1, 0, num_inference_steps + 1)
        self.discrete_sigmas = self.sigmas[:-1]

        if device is not None:
            self.sigmas = self.sigmas.to(device)
            self.timesteps = self.timesteps.to(device)
            self.discrete_sigmas = self.discrete_sigmas.to(device)

    def _get_velocity_from_prediction(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert model prediction to velocity field based on prediction type.

        Args:
            model_output: Model output.
            sample: Current sample.
            sigma: Current sigma.

        Returns:
            velocity: Velocity field (dx/dt).
        """
        sigma_data = self.config.sigma_data
        prediction_type = self.config.prediction_type
        
        # Ensure sigma has proper shape
        while len(sigma.shape) < len(sample.shape):
            sigma = sigma.unsqueeze(-1)

        if prediction_type == "velocity":
            # Direct velocity prediction
            velocity = model_output
        elif prediction_type == "epsilon":
            # From epsilon prediction: v = (x - σ*ε) / σ - x/σ = -ε
            # Flow matching: dx/dt ≈ -ε * σ
            velocity = -model_output * sigma
        elif prediction_type == "sample":
            # From sample prediction: v = (x_pred - x) / σ
            velocity = (model_output - sample) / sigma.clamp(min=1e-8)
        else:
            raise ValueError(f"Unknown prediction_type: {prediction_type}")
            
        return velocity

    def step_euler(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        dt: float,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Single Euler update (1st order).

        Args:
            model_output: Model output.
            timestep: Current timestep index.
            sample: Current sample.
            dt: Step size.
            generator: Random generator.

        Returns:
            Updated sample.
        """
        sigma = self.sigmas[timestep].to(sample.device)
        velocity = self._get_velocity_from_prediction(model_output, sample, sigma)
        
        # Euler update: x_{t+dt} = x_t + v * dt
        prev_sample = sample + velocity * dt
        
        return prev_sample, velocity

    def step_heun(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        dt: float,
        model_fn=None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Heun's method (improved Euler, 2nd order).

        Args:
            model_output: Initial model output.
            timestep: Current timestep index.
            sample: Current sample.
            dt: Step size.
            model_fn: Model function for second evaluation.
            generator: Random generator.

        Returns:
            Updated sample.
        """
        sigma = self.sigmas[timestep].to(sample.device)
        velocity_1 = self._get_velocity_from_prediction(model_output, sample, sigma)
        
        # Predictor step (Euler)
        sample_pred = sample + velocity_1 * dt
        
        if model_fn is not None and timestep + 1 < len(self.sigmas) - 1:
            # Corrector step
            sigma_next = self.sigmas[timestep + 1].to(sample.device)
            model_output_2 = model_fn(sample_pred, sigma_next)
            velocity_2 = self._get_velocity_from_prediction(model_output_2, sample_pred, sigma_next)
            
            # Heun update: x_{t+dt} = x_t + (v_1 + v_2) / 2 * dt
            prev_sample = sample + (velocity_1 + velocity_2) / 2 * dt
            velocity = (velocity_1 + velocity_2) / 2
        else:
            prev_sample = sample_pred
            velocity = velocity_1
        
        return prev_sample, velocity

    def step_rk4(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        dt: float,
        model_fn=None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Runge-Kutta 4th-order method.

        Args:
            model_output: Initial model output.
            timestep: Current timestep index.
            sample: Current sample.
            dt: Step size.
            model_fn: Model function for intermediate evaluations.
            generator: Random generator.

        Returns:
            Updated sample.
        """
        sigma = self.sigmas[timestep].to(sample.device)
        
        # k1
        k1 = self._get_velocity_from_prediction(model_output, sample, sigma)
        
        if model_fn is None:
            # Fallback to Euler if no model function
            return sample + k1 * dt, k1
        
        # Mid sigma values
        sigma_mid = (self.sigmas[timestep] + self.sigmas[min(timestep + 1, len(self.sigmas) - 2)]) / 2
        sigma_mid = sigma_mid.to(sample.device)
        sigma_next = self.sigmas[min(timestep + 1, len(self.sigmas) - 2)].to(sample.device)
        
        # k2
        sample_2 = sample + k1 * (dt / 2)
        model_output_2 = model_fn(sample_2, sigma_mid)
        k2 = self._get_velocity_from_prediction(model_output_2, sample_2, sigma_mid)
        
        # k3
        sample_3 = sample + k2 * (dt / 2)
        model_output_3 = model_fn(sample_3, sigma_mid)
        k3 = self._get_velocity_from_prediction(model_output_3, sample_3, sigma_mid)
        
        # k4
        sample_4 = sample + k3 * dt
        model_output_4 = model_fn(sample_4, sigma_next)
        k4 = self._get_velocity_from_prediction(model_output_4, sample_4, sigma_next)
        
        # RK4 update: x_{t+dt} = x_t + (k1 + 2*k2 + 2*k3 + k4) / 6 * dt
        velocity = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        prev_sample = sample + velocity * dt
        
        return prev_sample, velocity

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
        model_fn=None,
    ) -> Union[OmniSchedulerOutput, Tuple]:
        """
        Execute one sampling step, auto-selecting solver and strategy.

        Args:
            model_output (`torch.Tensor`): Model output.
            timestep (`int`): Current timestep index.
            sample (`torch.Tensor`): Current sample.
            generator (`torch.Generator`, *optional*): Random generator.
            return_dict (`bool`): Return dict format.
            model_fn: Model function for higher-order solvers.

        Returns:
            `OmniSchedulerOutput` or `tuple`.
        """
        if self.timesteps is None:
            raise ValueError(
                "`self.timesteps` is not set, you need to run 'set_timesteps' after creating the scheduler"
            )

        # Compute step size
        if timestep + 1 < len(self.sigmas):
            sigma_curr = self.sigmas[timestep]
            sigma_next = self.sigmas[timestep + 1]
            dt = sigma_next - sigma_curr
        else:
            dt = -self.sigmas[timestep]

        # Determine current stage
        if self.config.use_multi_stage:
            if timestep < self.coarse_steps:
                self.current_stage = 0  # coarse
            else:
                self.current_stage = 1  # refine

        # Choose solver order based on stage
        solver_order = self.config.solver_order
        
        # Coarse stage may use lower order to speed up
        if self.current_stage == 0 and self.config.use_multi_stage:
            effective_order = min(solver_order, 2)
        else:
            effective_order = solver_order

        if effective_order == 1:
            prev_sample, velocity = self.step_euler(
                model_output, timestep, sample, dt, generator
            )
        elif effective_order == 2:
            prev_sample, velocity = self.step_heun(
                model_output, timestep, sample, dt, model_fn, generator
            )
        elif effective_order >= 4:
            prev_sample, velocity = self.step_rk4(
                model_output, timestep, sample, dt, model_fn, generator
            )
        else:
            prev_sample, velocity = self.step_euler(
                model_output, timestep, sample, dt, generator
            )

        prev_sample_mean = prev_sample.clone()

        if not return_dict:
            return (prev_sample, prev_sample_mean, velocity, self.current_stage)

        return OmniSchedulerOutput(
            prev_sample=prev_sample,
            prev_sample_mean=prev_sample_mean,
            velocity=velocity,
            stage=self.current_stage,
        )

    def apply_policy_trajectory(
        self,
        policy_velocities: Optional[torch.Tensor],
        sample: torch.Tensor,
        return_all: bool = False,
    ):
        """
        Run inference directly using the velocity trajectory from a π-Flow policy network.

        Args:
            policy_velocities: Velocity field sequence of shape [S, *sample.shape].
            sample: Initial noise sample.
            return_all: Whether to return intermediate results for each step.
        """
        if policy_velocities is None:
            return (sample, []) if return_all else sample

        steps = min(policy_velocities.shape[0], len(self.sigmas) - 1)
        traj = []
        curr = sample
        for i in range(steps):
            out = self.step(policy_velocities[i], i, curr, return_dict=True)
            curr = out.prev_sample
            if return_all:
                traj.append(curr)

        return (curr, traj) if return_all else curr

    def step_correct(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Corrective step to improve sample quality.

        Args:
            model_output (`torch.Tensor`): Model output.
            sample (`torch.Tensor`): Current sample.
            generator (`torch.Generator`, *optional*): Random generator.
            return_dict (`bool`): Return dict format.

        Returns:
            `SchedulerOutput` or `tuple`.
        """
        if self.timesteps is None:
            raise ValueError(
                "`self.timesteps` is not set, you need to run 'set_timesteps' after creating the scheduler"
            )

        # Generate correction noise
        noise = randn_tensor(sample.shape, layout=sample.layout, generator=generator).to(sample.device)

        # Compute step size
        grad_norm = torch.norm(model_output.reshape(model_output.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        step_size = (self.config.snr * noise_norm / grad_norm.clamp(min=1e-8)) ** 2 * 2
        step_size = step_size * torch.ones(sample.shape[0]).to(sample.device)

        # Adjust shape
        step_size = step_size.flatten()
        while len(step_size.shape) < len(sample.shape):
            step_size = step_size.unsqueeze(-1)

        # Correction update
        prev_sample_mean = sample + step_size * model_output
        prev_sample = prev_sample_mean + ((step_size * 2) ** 0.5) * noise

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to original samples.

        Args:
            original_samples: Original samples.
            noise: Noise tensor.
            timesteps: Timestep indices.

        Returns:
            Noised samples.
        """
        timesteps = timesteps.to(original_samples.device)
        
        # Get corresponding sigma
        sigmas = self.discrete_sigmas.to(original_samples.device)[timesteps]
        
        # Adjust shape
        while len(sigmas.shape) < len(original_samples.shape):
            sigmas = sigmas.unsqueeze(-1)
        
        # Add noise
        if noise is not None:
            noisy_samples = original_samples + noise * sigmas
        else:
            noisy_samples = original_samples + torch.randn_like(original_samples) * sigmas
            
        return noisy_samples

    def get_flow_velocity(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute target velocity field for flow matching.

        Used to train the policy network.

        Args:
            x_0: Start sample (noise).
            x_1: Target sample (data).
            t: Time point [0, 1].

        Returns:
            Target velocity field.
        """
        # Linear interpolation path: x_t = (1-t) * x_0 + t * x_1
        # Velocity field: v = dx/dt = x_1 - x_0
        while len(t.shape) < len(x_0.shape):
            t = t.unsqueeze(-1)
            
        velocity = x_1 - x_0
        return velocity

    def get_interpolated_sample(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get interpolated sample on the flow-matching path.

        Args:
            x_0: Start sample (noise).
            x_1: Target sample (data).
            t: Time point [0, 1].

        Returns:
            Interpolated sample x_t.
        """
        while len(t.shape) < len(x_0.shape):
            t = t.unsqueeze(-1)
            
        x_t = (1 - t) * x_0 + t * x_1
        return x_t

    def __len__(self):
        return self.config.num_train_timesteps
