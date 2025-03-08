"""
Jovi_Measure - Image Metrics
"""

import skimage.measure as skm

from comfy.utils import ProgressBar

from .. import JOVBaseNode
from . import deep_merge, tensor2cv

# ==============================================================================
# === CLASS ===
# ==============================================================================

class ShannonEntropyNode(JOVBaseNode):
    NAME = "SHANNON ENTROPY"
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("FLOAT",)
    OUTPUT_TOOLTIPS = (
        "The Shannon entropy value of the image.",
    )
    SORT = 50
    DESCRIPTION = """
Calculate the Shannon entropy of an image.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "required": {
                'image': ("IMAGE", {"default": None, "tooltip": "RGBA, RGB or Grayscale image"}),
            }
        })
        return d

    def run(self, image, **kw) -> float:
        vals = []
        images = [i for i in image]
        pbar = ProgressBar(len(images))
        for idx, image in enumerate(images):
            image = tensor2cv(image)
            val = skm.shannon_entropy(image)
            vals.append(val)
            pbar.update_absolute(idx)
        return vals,

class BlurEffectNode(JOVBaseNode):
    NAME = "BLUR EFFECT"
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("FLOAT",)
    OUTPUT_TOOLTIPS = (
        "The amount of blurriness (0->1.0) of the input image.",
    )
    SORT = 50
    DESCRIPTION = """
Calculate the blurriness of the input image.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "required": {
                'image': ("IMAGE", {"default": None, "tooltip": "RGBA, RGB or Grayscale image"}),
            },
            "optional": {
                'h_size': ("INT", {"default": 11, "tooltip": "Size of the re-blurring filter."}),
                # 'channel_axis': ("INT", {"default": 0, "min": 0, "max": 3}),
            }
        })
        return d

    def run(self, image, h_size, **kw) -> float:
        vals = []
        images = [i for i in image]
        pbar = ProgressBar(len(images))
        for idx, image in enumerate(images):
            image = tensor2cv(image)
            channel_axis = 2
            if len(hwc := image.shape) == 2 or hwc[2] == 1:
                channel_axis = None
            val = skm.blur_effect(image, h_size=h_size, channel_axis=channel_axis)
            vals.append(val)
            pbar.update_absolute(idx)
        return vals,