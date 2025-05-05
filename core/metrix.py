""" Jovi_Measure - Image Metrics """

import skimage.measure as skm

from comfy.utils import ProgressBar

from cozy_comfyui import \
    EnumConvertType, \
    deep_merge, parse_param, zip_longest_fill

from cozy_comfyui.lexicon import \
    Lexicon

from cozy_comfyui.node import \
    CozyBaseNode

from cozy_comfyui.image.convert import \
    tensor_to_cv

# ==============================================================================
# === CLASS ===
# ==============================================================================

class ShannonEntropyNode(CozyBaseNode):
    NAME = "SHANNON ENTROPY"
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("FLOAT",)
    OUTPUT_IS_LIST = (True,)
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
                Lexicon.IMAGE: ("IMAGE", {
                    "default": None,}),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> float:
        image = parse_param(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        vals = []
        pbar = ProgressBar(len(image))
        for idx, image in enumerate(image):
            image = tensor_to_cv(image)
            val = skm.shannon_entropy(image)
            vals.append(val)
            pbar.update_absolute(idx)
        return (vals,)

class BlurEffectNode(CozyBaseNode):
    NAME = "BLUR EFFECT"
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("FLOAT",)
    OUTPUT_IS_LIST = (True,)
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
                Lexicon.IMAGE: ("IMAGE", {
                    "default": None,}),
            },
            "optional": {
                Lexicon.BLUR: ("INT", {
                    "default": 11,
                    "tooltip": "Size of the re-blurring filter"}),
                # 'channel_axis': ("INT", {"default": 0, "min": 0, "max": 3}),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> float:
        image = parse_param(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        blur = parse_param(kw, Lexicon.BLUR, EnumConvertType.INT, None)

        vals = []
        params = list(zip_longest_fill(image, blur))
        pbar = ProgressBar(len(params))
        for idx, (image, blur) in enumerate(params):
            image = tensor_to_cv(image)
            channel_axis = 2
            if len(hwc := image.shape) == 2 or hwc[2] == 1:
                channel_axis = None
            val = skm.blur_effect(image, h_size=blur, channel_axis=channel_axis)
            vals.append(val)
            pbar.update_absolute(idx)
        return (vals,)
