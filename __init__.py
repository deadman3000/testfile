
"""SeedVR2 image-space multiplier bridge for ComfyUI.

Decode a source image latent, resize the decoded pixels in image space, then
encode with the SeedVR2 VAE. A user-entered short-side value sets the base
short-side resolution; the multiplier scales it. The long side is calculated
from the decoded image aspect ratio, and both dimensions are aligned to 16 px.
"""

import torch
import torch.nn.functional as F


class SeedVR2ImageSpaceMultiplierBridge:
    SCALE_METHODS = ["bilinear", "bicubic", "lanczos", "nearest", "nearest-exact", "area"]
    PIXEL_ALIGNMENT = 16

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "image_vae": ("VAE",),
                "seedvr2_vae": ("VAE",),
                # Keep the original widget order intact for saved workflow compatibility.
                "multiplier": ("FLOAT", {
                    "default": 1.0,
                    "min": 1.0,
                    "max": 4.0,
                    "step": 0.1,
                    "round": 0.1,
                }),
                "scaling_method": (cls.SCALE_METHODS, {"default": "lanczos"}),
                # New widget goes last so existing saved widget values keep their positions.
                "short_side": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 16384,
                    "step": 16,
                    "display": "number",
                    "tooltip": "Base short-side size in pixels. Set to 0 to use the decoded image's current short side.",
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING", "IMAGE")
    RETURN_NAMES = ("seedvr2_latent", "output_resolution", "reference_image")
    FUNCTION = "convert"
    CATEGORY = "latent/SeedVR2"
    DESCRIPTION = (
        "Decodes the source latent, resizes pixels using short_side × multiplier while "
        "preserving aspect ratio, aligns dimensions to 16 pixels, and encodes with the "
        "SeedVR2 VAE. The output_resolution string can be connected to a Show Text node. "
        "Set short_side to 0 to use the decoded image's original short side."
    )

    @classmethod
    def _target_dimension(cls, value, alignment=None):
        """Round a pixel dimension to the nearest aligned size."""
        if alignment is None:
            alignment = cls.PIXEL_ALIGNMENT
        return max(alignment, int(round(value / alignment)) * alignment)

    @classmethod
    def _target_size(cls, height, width, short_side, multiplier):
        """Calculate aligned output dimensions from short side and aspect ratio."""
        actual_short = min(height, width)
        actual_long = max(height, width)
        base_short = actual_short if int(short_side) == 0 else int(short_side)
        target_short = base_short * float(multiplier)
        target_long = target_short * (actual_long / actual_short)

        short_dim = cls._target_dimension(target_short)
        long_dim = cls._target_dimension(target_long)
        if height <= width:
            return short_dim, long_dim
        return long_dim, short_dim

    @staticmethod
    def _resize_image(image, target_h, target_w, method):
        # ComfyUI IMAGE layout is [B,H,W,C]. Interpolate in channel-first layout.
        original_dtype = image.dtype
        nchw = image[..., :3].permute(0, 3, 1, 2)
        work = nchw.float() if nchw.dtype in (torch.float16, torch.bfloat16) else nchw

        if method == "lanczos":
            def lanczos_weights(input_size, output_size, device):
                scale = input_size / output_size
                filter_scale = max(scale, 1.0)
                positions = (
                    (torch.arange(output_size, device=device, dtype=torch.float32) + 0.5)
                    * scale - 0.5
                )
                source = torch.arange(input_size, device=device, dtype=torch.float32)
                distance = (positions[:, None] - source[None, :]) / filter_scale
                weights = torch.sinc(distance) * torch.sinc(distance / 3.0)
                weights = torch.where(
                    distance.abs() < 3.0,
                    weights,
                    torch.zeros_like(weights),
                )
                weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
                return weights

            hw = lanczos_weights(work.shape[-2], target_h, work.device)
            ww = lanczos_weights(work.shape[-1], target_w, work.device)
            resized = torch.einsum("oh,bchw->bcow", hw, work)
            resized = torch.einsum("bcow,pw->bcop", resized, ww)
        else:
            kwargs = {"size": (target_h, target_w), "mode": method}
            if method in ("bilinear", "bicubic"):
                kwargs["align_corners"] = False
            resized = F.interpolate(work, **kwargs)

        return resized.to(original_dtype).permute(0, 2, 3, 1).contiguous()

    def convert(
        self,
        samples,
        image_vae,
        seedvr2_vae,
        multiplier=1.0,
        scaling_method="lanczos",
        short_side=0,
    ):
        if not isinstance(samples, dict) or "samples" not in samples:
            raise ValueError("Input must be a ComfyUI LATENT containing a 'samples' tensor.")

        image_latent = samples["samples"]
        if getattr(image_latent, "is_nested", False):
            nested = image_latent.unbind()
            if len(nested) != 1:
                raise ValueError("Nested latent input must contain exactly one image latent.")
            image_latent = nested[0]

        # Qwen Image VAE in this ComfyUI path expects video-style latent layout
        # [B,C,T,H,W], even when processing a single still image. Preserve an
        # existing T=1 dimension, or add it to a conventional 4-D image latent.
        if image_latent.ndim == 4:
            image_latent = image_latent.unsqueeze(2)
        elif image_latent.ndim == 5:
            if image_latent.shape[2] != 1:
                raise ValueError(
                    "This bridge supports still images only. Expected a 5-D latent with T=1, "
                    f"but received shape {tuple(image_latent.shape)}."
                )
        else:
            raise ValueError(
                "Expected a 4-D image latent [B,C,H,W] or 5-D single-frame latent [B,C,1,H,W], "
                f"but received shape {tuple(image_latent.shape)}."
            )

        factor = float(multiplier)
        if not 1.0 <= factor <= 4.0:
            raise ValueError(f"multiplier must be between 1.0 and 4.0, got {factor}.")
        if int(short_side) < 0 or int(short_side) > 16384:
            raise ValueError(f"short_side must be between 0 and 16384, got {short_side}.")
        if scaling_method not in self.SCALE_METHODS:
            raise ValueError(f"Unsupported scaling_method: {scaling_method}.")

        image = image_vae.decode(image_latent)

        # Qwen Image VAE may return a single-frame tensor in [B,T,H,W,C].
        # Remove the frame dimension only when T == 1.
        if image.ndim == 5:
            if image.shape[1] != 1:
                raise ValueError(
                    "Image VAE returned multiple frames where a still image was expected; "
                    f"received shape {tuple(image.shape)}."
                )
            image = image[:, 0, :, :, :]

        # Accept standard ComfyUI IMAGE tensors [B,H,W,C] after normalizing
        # the single-frame Qwen output above.
        if image.ndim != 4 or image.shape[-1] < 3:
            raise ValueError(
                "Image VAE did not return an IMAGE tensor in [B,H,W,C] layout "
                "or a single-frame [B,1,H,W,C] tensor; "
                f"received shape {tuple(image.shape)}."
            )

        image = image[..., :3]

        old_h, old_w = image.shape[1:3]
        target_h, target_w = self._target_size(
            old_h, old_w, short_side, factor
        )
        if (target_h, target_w) != (old_h, old_w):
            image = self._resize_image(image, target_h, target_w, scaling_method)

        output_resolution = f"{target_w} × {target_h}"
        # Match official SeedVR2 preprocessing before VAE encoding.
        # Keep `image` unchanged for the reference_image output.
        seedvr2_input = image.clamp(0.0, 1.0)
        seedvr2_latent = seedvr2_vae.encode(seedvr2_input)

        if seedvr2_latent.ndim != 5:
            raise ValueError(
                "SeedVR2 VAE encode should return a 5-D video latent [B,C,T,H,W], "
                f"but returned shape {tuple(seedvr2_latent.shape)}. Check that the "
                "selected VAE is the SeedVR2 VAE."
            )
        if seedvr2_latent.shape[1] != 16:
            raise ValueError(
                "SeedVR2 VAE encode returned an unexpected channel count: "
                f"shape {tuple(seedvr2_latent.shape)}. SeedVR2 expects 16 latent channels. "
                "Check the VAE connection and ComfyUI/SeedVR2 versions."
            )

        return ({"samples": seedvr2_latent}, output_resolution, image)


WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS = {
    "SeedVR2ImageSpaceMultiplierBridge": SeedVR2ImageSpaceMultiplierBridge
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVR2ImageSpaceMultiplierBridge": "SeedVR2 Image-Space Multiplier Bridge"
}