"""ComfyUI nodes for the MiniMax-H3 Turbo LoRA (4-step audio-video).

Drops into the stock MiniMax-H3 workflow (t2v and i2v):

  MiniMaxH3TurboLoRA    MODEL -> MODEL   applies the turbo LoRA
  MiniMaxH3TurboSampler       -> SAMPLER 4-step sampler for SamplerCustomAdvanced

The sampler steps the video and audio streams on their own flow schedules
(video shift 12, audio shift 3). A stock single-schedule sampler over-steps the
audio at 4 steps and the audio breaks; this one keeps each stream on its own
clock so 4 steps stays clean.
"""

import math

import torch

import comfy.samplers
import comfy.lora
import comfy.utils
import folder_paths

SHIFT_V, SHIFT_A = 12.0, 3.0


def _time_shift_sigma(sigma, fr, to):
    base = sigma / (fr + sigma * (1.0 - fr))
    return to * base / (1.0 + (to - 1.0) * base)


def _time_shift_slope(sigma, fr, to):
    base = sigma / (fr + sigma * (1.0 - fr))
    return (to * (1.0 + (fr - 1.0) * base) ** 2) / (fr * (1.0 + (to - 1.0) * base) ** 2)


def _audio_sigma(sv):
    return _time_shift_sigma(sv, SHIFT_V, SHIFT_A)


def _audio_slope(sv):
    return _time_shift_slope(sv, SHIFT_V, SHIFT_A)


def _latent_shapes(model):
    """[video_shape, audio_shape] the sampler is packing over — video latent is
    flattened first, then audio, so we need the split point."""
    guider = getattr(model, "inner_model", model)
    conds = getattr(guider, "conds", None)
    if conds:
        for cond_list in conds.values():
            for c in (cond_list or []):
                mc = c.get("model_conds", {}) if isinstance(c, dict) else {}
                if "latent_shapes" in mc:
                    return mc["latent_shapes"].cond
    return None


@torch.no_grad()
def _turbo_sampler(model, x, sigmas, extra_args=None, callback=None, disable=None,
                   **kwargs):
    extra_args = {} if extra_args is None else extra_args
    shapes = _latent_shapes(model)
    if not shapes or len(shapes) < 2:
        raise RuntimeError(
            "MiniMaxH3TurboSampler expects the MiniMax-H3 video+audio latent "
            "(the EmptyMiniMaxH3LatentAV / MiniMaxH3ImageToVideo output).")
    v_numel = math.prod(shapes[0][1:])           # flat pack is [video | audio]
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        sv, sv_n = float(sigmas[i]), float(sigmas[i + 1])
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        out = (x - denoised) / sigmas[i]
        xv, ov = x[..., :v_numel], out[..., :v_numel]
        xa, oa = x[..., v_numel:], out[..., v_numel:]
        xv = xv + (sv_n - sv) * ov               # video on its own sigma
        sl = _audio_slope(max(sv, 1e-6))
        xa = xa + (_audio_sigma(sv_n) - _audio_sigma(sv)) * (oa / sl)  # audio clock
        x = torch.cat([xv, xa], dim=-1)
        if callback is not None:
            callback({"i": i, "denoised": denoised, "x": x,
                      "sigma": sigmas[i], "sigma_hat": sigmas[i]})
    return x


class MiniMaxH3TurboSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "MiniMaxH3Turbo"
    DESCRIPTION = ("4-step sampler for the MiniMax-H3 Turbo LoRA. Feed into "
                   "SamplerCustomAdvanced and set the scheduler to 4 steps.")

    def get_sampler(self):
        return (comfy.samplers.KSAMPLER(_turbo_sampler),)


class MiniMaxH3TurboLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "lora_name": (folder_paths.get_filename_list("loras"),),
            "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0,
                                   "step": 0.01}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_lora"
    CATEGORY = "MiniMaxH3Turbo"
    DESCRIPTION = "Apply the MiniMax-H3 Turbo LoRA to the H3 diffusion model."

    def apply_lora(self, model, lora_name, strength):
        path = folder_paths.get_full_path("loras", lora_name)
        lora = comfy.utils.load_torch_file(path, safe_load=True)
        modules = sorted({k.rsplit(".lora_", 1)[0] for k in lora})
        to_load = {m: "diffusion_model.{}.weight".format(m) for m in modules}
        patches = comfy.lora.load_lora(lora, to_load, log_missing=True)
        new_model = model.clone()
        new_model.add_patches(patches, strength)
        return (new_model,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3TurboLoRA": MiniMaxH3TurboLoRA,
    "MiniMaxH3TurboSampler": MiniMaxH3TurboSampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3TurboLoRA": "MiniMax-H3 Turbo LoRA",
    "MiniMaxH3TurboSampler": "MiniMax-H3 Turbo Sampler (4-step)",
}
