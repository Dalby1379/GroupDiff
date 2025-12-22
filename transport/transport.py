# Copyright © 2025, Adobe Inc. and its licensors. 
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Modified from https://github.com/bytetriper/RAE/blob/main/src/stage2/transport/transport.py


import torch as th
import numpy as np
import enum

from . import path
from .utils import mean_flat
from .integrators import ode, sde


class ModelType(enum.Enum):
    NOISE = enum.auto()  # predicts epsilon
    SCORE = enum.auto()  # predicts \nabla \log p(x)
    VELOCITY = enum.auto()  # predicts v(x)


class PathType(enum.Enum):
    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()


class WeightType(enum.Enum):
    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


def truncated_logitnormal_sample(shape, mu, sigma, low=0.0, high=1.0):
    """
    Samples X in (0,1) with Z = logit(X) ~ Normal(mu, sigma^2), truncated so X in [low, high].
    """
    mu, sigma = th.as_tensor(mu), th.as_tensor(sigma)
    low, high = th.as_tensor(low), th.as_tensor(high)

    # Standardize bounds (logit space)
    z_low, z_high = th.logit(low), th.logit(high)

    # Base distribution
    base = th.distributions.Normal(th.zeros_like(mu), th.ones_like(sigma))
    alpha = (z_low - mu) / sigma
    beta = (z_high - mu) / sigma

    # Inverse CDF sampling on truncated range
    cdf_alpha, cdf_beta = base.cdf(alpha), base.cdf(beta)

    out_shape = th.broadcast_shapes(shape, mu.shape, sigma.shape, low.shape, high.shape)
    U = th.rand(out_shape, device=mu.device, dtype=mu.dtype)
    U = cdf_alpha + (cdf_beta - cdf_alpha) * U.clamp_(0, 1)

    Z = mu + sigma * base.icdf(U)
    return th.sigmoid(Z).clamp(low, high)


class Transport:
    def __init__(
        self,
        *,
        model_type,
        path_type,
        loss_type,
        time_dist_type,
        time_dist_shift,
        train_eps,
        sample_eps,
    ):
        path_options = {
            PathType.LINEAR: path.ICPlan,
            PathType.GVP: path.GVPCPlan,
            PathType.VP: path.VPCPlan,
        }

        self.loss_type = loss_type
        self.model_type = model_type
        self.time_dist_type = time_dist_type
        self.time_dist_shift = time_dist_shift
        assert self.time_dist_shift >= 1.0, "time distribution shift must be >= 1.0."
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps

    def prior_logp(self, z):
        """Standard multivariate normal prior."""
        N = np.prod(z.shape[1:])
        return -N / 2.0 * np.log(2 * np.pi) - th.sum(z**2, dim=list(range(1, z.ndim))) / 2.0

    def check_interval(
        self,
        train_eps,
        sample_eps,
        *,
        diffusion_form="SBDM",
        sde=False,
        reverse=False,
        eval=False,
        last_step_size=0.0,
    ):
        eps = train_eps if not eval else sample_eps
        t0, t1 = 0.0, 1.0 - 1.0 / 1000

        # Handling numerical stabilities based on path type
        if isinstance(self.path_sampler, path.VPCPlan):
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
        elif isinstance(self.path_sampler, (path.ICPlan, path.GVPCPlan)):
            if self.model_type != ModelType.VELOCITY or sde:
                # Avoid numerical issue by taking a first semi-implicit step
                is_sbdm_sde = diffusion_form == "SBDM" and sde
                t0 = eps if is_sbdm_sde or self.model_type != ModelType.VELOCITY else 0
                t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        if reverse:
            return 1 - t0, 1 - t1
        return t0, t1

    def sample(self, x1):
        """Sample t and x0."""
        x0 = th.randn_like(x1)
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)

        # Time distribution sampling
        dist_options = self.time_dist_type.split("_")
        if dist_options[0] == "uniform":
            t = th.rand((x1.shape[0],), device=x1.device) * (t1 - t0) + t0
        elif dist_options[0] == "logit-normal":
            mu, sigma = float(dist_options[1]), float(dist_options[2])
            t = truncated_logitnormal_sample((x1.shape[0],), mu, sigma, t0, t1).to(x1.device)
        else:
            raise NotImplementedError(f"Unknown time distribution type {self.time_dist_type}")

        # Time shift
        t = self.time_dist_shift * t / (1 + (self.time_dist_shift - 1) * t)
        return t, x0, x1

    def resolve_prediction(self, model_output, x, t):
        """
        Centralized logic to convert raw model output to (ODE Drift, Score).
        Returns:
            ode_drift: The vector field for probability flow ODE.
            score: The score function \nabla log p_t(x).
        """
        # Determine coefficients based on path
        if self.model_type == ModelType.VELOCITY:
            # Velocity: Drift is raw output, Score needs conversion
            ode_drift = model_output
            score = self.path_sampler.get_score_from_velocity(model_output, x, t)

        elif self.model_type == ModelType.SCORE:
            # Score: Drift needs conversion, Score is raw output
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            ode_drift = -drift_mean + drift_var * model_output
            score = model_output

        elif self.model_type == ModelType.NOISE:
            # Noise: Both need conversion
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
            score = model_output / -sigma_t
            ode_drift = -drift_mean + drift_var * score

        else:
            raise NotImplementedError()

        return ode_drift, score

    def training_losses(self, model, x1, model_kwargs=None, t=None):
        """
        Compute training losses.
        
        Args:
            model: The model to evaluate loss on
            x1: Clean data tensor
            model_kwargs: Optional dict of extra keyword arguments to pass to the model
            t: Optional pre-sampled timesteps. If None, will sample internally.
               Shape should be (batch_size,) with values in [0, 1]
        
        Returns:
            Dict with keys "pred" and "loss"
        """
        model_kwargs = model_kwargs or {}
        
        # Sample or use provided timesteps
        if t is None:
            t, x0, x1 = self.sample(x1)
        else:
            x0 = th.randn_like(x1)
            # t is already provided, just validate shape
            assert t.shape[0] == x1.shape[0], f"Timestep batch size {t.shape[0]} must match data batch size {x1.shape[0]}"
        
        t, xt, ut = self.path_sampler.plan(t, x0, x1)

        model_output = model(xt, t, **model_kwargs)

        # Calculate Loss Weight
        if self.model_type == ModelType.VELOCITY:
            # Velocity loss is simple MSE against target velocity ut
            loss = mean_flat((model_output - ut) ** 2)
        else:
            # Weight computation for Score/Noise models
            _, drift_var = self.path_sampler.compute_drift(xt, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))

            if self.loss_type == WeightType.VELOCITY:
                weight = (drift_var / sigma_t) ** 2
            elif self.loss_type == WeightType.LIKELIHOOD:
                weight = drift_var / (sigma_t**2)
            else:  # WeightType.NONE
                weight = 1

            if self.model_type == ModelType.NOISE:
                loss = mean_flat(weight * ((model_output - x0) ** 2))
            else:  # SCORE
                loss = mean_flat(weight * ((model_output * sigma_t + x0) ** 2))

        return {"pred": model_output, "loss": loss}

    def get_drift(self):
        """Returns ODE drift function."""

        def drift_fn(x, t, model, **kwargs):
            model_output = model(x, t, **kwargs)
            drift, _ = self.resolve_prediction(model_output, x, t)
            return drift

        return drift_fn

    def get_score(self):
        """Returns Score function."""

        def score_fn(x, t, model, **kwargs):
            model_output = model(x, t, **kwargs)
            _, score = self.resolve_prediction(model_output, x, t)
            return score

        return score_fn


class Sampler:
    def __init__(self, transport):
        self.transport = transport
        # Keep public API for accessing drift/score directly if needed
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()

    def sample_sde(
        self,
        *,
        sampling_method="Euler",
        diffusion_form="SBDM",
        diffusion_norm=1.0,
        last_step="Mean",
        last_step_size=0.04,
        num_steps=250,
    ):
        if last_step is None:
            last_step_size = 0.0

        # --- OPTIMIZED SDE DRIFT (Runs model ONLY ONCE per step) ---
        def sde_diffusion_fn(x, t):
            return self.transport.path_sampler.compute_diffusion(x, t, form=diffusion_form, norm=diffusion_norm)

        def sde_drift_fn(x, t, model, **kwargs):
            # 1. Run Model Once
            model_output = model(x, t, **kwargs)
            # 2. Get both ODE drift and Score from single output
            ode_drift, score = self.transport.resolve_prediction(model_output, x, t)
            # 3. Combine: Drift_SDE = Drift_ODE - Diffusion * Score
            diffusion = sde_diffusion_fn(x, t)
            return ode_drift - diffusion * score

        # Setup integration intervals
        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )

        _sde = sde(
            sde_drift_fn,
            sde_diffusion_fn,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            sampler_type=sampling_method,
            time_dist_shift=self.transport.time_dist_shift,
        )

        # Last Step Logic
        def _sample(init, model, **model_kwargs):
            xs = _sde.sample(init, model, **model_kwargs)

            # Apply final correction step
            x_final = xs[-1]
            t_final = th.ones(init.size(0), device=init.device) * (1 - t1)

            if last_step == "Mean":
                drift = sde_drift_fn(x_final, t_final, model, **model_kwargs)
                x = x_final - drift * last_step_size
            elif last_step == "Euler":
                # Note: Uses ODE drift for pure Euler correction usually, matching original logic
                drift = self.drift(x_final, t_final, model, **model_kwargs)
                x = x_final - drift * last_step_size
            elif last_step == "Tweedie":
                alpha = self.transport.path_sampler.compute_alpha_t(t_final)[0][0]
                sigma = self.transport.path_sampler.compute_sigma_t(t_final)[0][0]
                score_val = self.score(x_final, t_final, model, **model_kwargs)
                x = x_final / alpha + (sigma**2) / alpha * score_val
            else:  # None
                x = x_final

            xs.append(x)
            assert len(xs) == num_steps, "Samples does not match the number of steps"
            return xs

        return _sample

    def sample_ode(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
    ):
        drift_fn = self.drift
        if reverse:
            drift_fn = lambda x, t, model, **kwargs: self.drift(x, th.ones_like(t) * (1 - t), model, **kwargs)

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=drift_fn,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
            time_dist_shift=self.transport.time_dist_shift,
        )

        return _ode.sample
