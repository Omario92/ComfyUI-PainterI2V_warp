import dataclasses
from collections.abc import Mapping

import torch
import comfy.model_management
import comfy.utils
import node_helpers
from comfy_api.latest import io, ComfyExtension
from typing_extensions import override


def _ensure_mapping(payload: object) -> dict:
    """Coerce the incoming WanVideoWrapper payload into a mutable dictionary."""

    if isinstance(payload, Mapping):
        return dict(payload)

    if hasattr(payload, "to_dict") and callable(payload.to_dict):
        return dict(payload.to_dict())

    if dataclasses.is_dataclass(payload):
        return dataclasses.asdict(payload)

    if hasattr(payload, "__dict__"):
        return dict(vars(payload))

    raise TypeError("Unsupported WanVideoWrapper payload type: expected mapping-like object")


def _get_first_available(data: Mapping, keys: tuple[str, ...], *, required: bool = True, default=None):
    """Return the first non-None value using a list of candidate keys."""

    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    if required:
        raise KeyError(f"Missing required value. Tried keys: {', '.join(keys)}")
    return default


def _painter_i2v_core(
    positive,
    negative,
    vae,
    width,
    height,
    length,
    batch_size,
    motion_amplitude=1.15,
    start_image=None,
    clip_vision_output=None,
):
    """Shared implementation that returns transformed conditioning and latent dictionary."""

    latent_samples = torch.zeros(
        [batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
        device=comfy.model_management.intermediate_device(),
    )

    if start_image is not None:
        start_image = start_image[:1]
        start_image = comfy.utils.common_upscale(
            start_image.movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)

        image = torch.ones(
            (length, height, width, start_image.shape[-1]),
            device=start_image.device,
            dtype=start_image.dtype,
        ) * 0.5
        image[0] = start_image[0]

        concat_latent_image = vae.encode(image[:, :, :, :3])

        mask = torch.ones(
            (1, 1, latent_samples.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]),
            device=start_image.device,
            dtype=start_image.dtype,
        )
        mask[:, :, 0] = 0.0

        if motion_amplitude > 1.0:
            base_latent = concat_latent_image[:, :, 0:1]
            gray_latent = concat_latent_image[:, :, 1:]

            diff = gray_latent - base_latent
            diff_mean = diff.mean(dim=(1, 3, 4), keepdim=True)
            diff_centered = diff - diff_mean
            scaled_latent = base_latent + diff_centered * motion_amplitude + diff_mean

            scaled_latent = torch.clamp(scaled_latent, -6, 6)
            concat_latent_image = torch.cat([base_latent, scaled_latent], dim=2)

        positive = node_helpers.conditioning_set_values(
            positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        )
        negative = node_helpers.conditioning_set_values(
            negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        )

        ref_latent = vae.encode(start_image[:, :, :, :3])
        positive = node_helpers.conditioning_set_values(
            positive, {"reference_latents": [ref_latent]}, append=True
        )
        negative = node_helpers.conditioning_set_values(
            negative, {"reference_latents": [torch.zeros_like(ref_latent)]}, append=True
        )

    if clip_vision_output is not None:
        positive = node_helpers.conditioning_set_values(
            positive, {"clip_vision_output": clip_vision_output}
        )
        negative = node_helpers.conditioning_set_values(
            negative, {"clip_vision_output": clip_vision_output}
        )

    return positive, negative, {"samples": latent_samples}

class PainterI2V(io.ComfyNode):
    """
    An enhanced Wan2.2 Image-to-Video node specifically designed to fix the slow-motion issue in 4-step LoRAs (like lightx2v).
    """
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PainterI2V",
            category="conditioning/video_models",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=4096, step=16),
                io.Int.Input("height", default=480, min=16, max=4096, step=16),
                io.Int.Input("length", default=81, min=1, max=4096, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Float.Input("motion_amplitude", default=1.15, min=1.0, max=2.0, step=0.05),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True),
                io.Image.Input("start_image", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ]
        )

    @classmethod
    def execute(
        cls,
        positive,
        negative,
        vae,
        width,
        height,
        length,
        batch_size,
        motion_amplitude=1.15,
        start_image=None,
        clip_vision_output=None,
    ) -> io.NodeOutput:
        positive, negative, latent_dict = _painter_i2v_core(
            positive,
            negative,
            vae,
            width,
            height,
            length,
            batch_size,
            motion_amplitude=motion_amplitude,
            start_image=start_image,
            clip_vision_output=clip_vision_output,
        )

        return io.NodeOutput(positive, negative, latent_dict)


class PainterI2VWanVideoWrapper:
    """Integrates PainterI2V improvements into WanVideoWrapper payload workflows."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "wan_wrapper_payload": ("ANY",),
                "motion_amplitude": ("FLOAT", {"default": 1.15, "min": 1.0, "max": 2.0, "step": 0.05}),
                "inject_cond_latent": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "start_image": ("IMAGE",),
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
            },
        }

    RETURN_TYPES = ("ANY", "CONDITIONING", "CONDITIONING", "LATENT", "ANY")
    RETURN_NAMES = (
        "wan_wrapper_payload",
        "positive",
        "negative",
        "latent",
        "cond_latent",
    )
    FUNCTION = "execute"
    CATEGORY = "conditioning/video_models"

    def execute(
        self,
        wan_wrapper_payload,
        motion_amplitude=1.15,
        inject_cond_latent=True,
        start_image=None,
        clip_vision_output=None,
    ):
        payload_dict = _ensure_mapping(wan_wrapper_payload)

        try:
            width = _get_first_available(payload_dict, ("width", "video_width", "latent_width"))
            height = _get_first_available(payload_dict, ("height", "video_height", "latent_height"))
            length = _get_first_available(payload_dict, ("length", "video_length", "frames", "frame_count"))
            batch_size = _get_first_available(
                payload_dict,
                ("batch_size", "video_batch_size", "latent_batch_size"),
                required=False,
                default=1,
            )

            positive = _get_first_available(
                payload_dict,
                ("positive", "positive_conditioning", "pos"),
            )
            negative = _get_first_available(
                payload_dict,
                ("negative", "negative_conditioning", "neg"),
            )
            vae = _get_first_available(payload_dict, ("vae", "video_vae"))
        except KeyError as exc:
            raise KeyError(
                "WanVideoWrapper payload is missing required Wan video fields"
            ) from exc

        payload_start_image = _get_first_available(
            payload_dict,
            ("start_image", "image", "initial_image"),
            required=False,
        )
        payload_clip_vision = _get_first_available(
            payload_dict,
            ("clip_vision_output", "clip_vision", "vision_output"),
            required=False,
        )

        effective_start_image = start_image if start_image is not None else payload_start_image
        effective_clip_vision = (
            clip_vision_output if clip_vision_output is not None else payload_clip_vision
        )

        positive, negative, latent_dict = _painter_i2v_core(
            positive,
            negative,
            vae,
            width,
            height,
            length,
            batch_size,
            motion_amplitude=motion_amplitude,
            start_image=effective_start_image,
            clip_vision_output=effective_clip_vision,
        )

        updated_payload = dict(payload_dict)
        updated_payload.update(
            {
                "positive": positive,
                "positive_conditioning": positive,
                "negative": negative,
                "negative_conditioning": negative,
                "latent": latent_dict,
                "latent_image": latent_dict,
                "vae": vae,
                "width": width,
                "height": height,
                "length": length,
                "batch_size": batch_size,
                "motion_amplitude": motion_amplitude,
            }
        )

        if effective_start_image is not None:
            updated_payload["start_image"] = effective_start_image

        if effective_clip_vision is not None:
            updated_payload["clip_vision_output"] = effective_clip_vision

        cond_latent = {
            "name": "PainterI2V",
            "positive": positive,
            "negative": negative,
            "latent": latent_dict,
            "latents": latent_dict,
            "motion_amplitude": motion_amplitude,
        }

        if inject_cond_latent:
            existing_cond_latents = payload_dict.get("add_cond_latents")
            if existing_cond_latents is None:
                cond_latent_list = []
            elif isinstance(existing_cond_latents, list):
                cond_latent_list = list(existing_cond_latents)
            else:
                cond_latent_list = [existing_cond_latents]

            cond_latent_list = [
                entry
                for entry in cond_latent_list
                if not isinstance(entry, Mapping) or entry.get("name") != "PainterI2V"
            ]
            cond_latent_list.append(cond_latent)
            updated_payload["add_cond_latents"] = cond_latent_list

        return updated_payload, positive, negative, latent_dict, cond_latent


class PainterI2VExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [PainterI2V]

async def comfy_entrypoint() -> PainterI2VExtension:
    return PainterI2VExtension()


# 节点注册映射
NODE_CLASS_MAPPINGS = {
    "PainterI2V": PainterI2V,
    "PainterI2VWanVideoWrapper": PainterI2VWanVideoWrapper,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterI2V": "PainterI2V (Wan2.2 Slow-Motion Fix)",
    "PainterI2VWanVideoWrapper": "PainterI2V WanVideoWrapper Bridge",
}

