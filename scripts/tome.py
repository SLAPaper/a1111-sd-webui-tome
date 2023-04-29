# Copyright 2023 SLAPaper
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

import sys

import gradio as gr
import torch

import launch
import modules.scripts as scripts
import modules.shared as shared
from modules.processing import Processed, StableDiffusionProcessing


class ToMe:
    """ToMe implementation"""

    def __init__(self):
        pass

    def patch_model(self, model: torch.nn.Module):
        if not launch.is_installed("tomesd"):
            print(
                "Cannot import tomesd, please install it manually following the instructions on https://github.com/dbolya/tomesd",
                file=sys.stderr)
            return

        ratio: float = shared.opts.data.get('tome_merging_ratio', 0.5)
        max_downsample: int = int(
            shared.opts.data.get('tome_maximum_down_sampling', 1))
        sx: int = shared.opts.data.get('tome_stride_x', 2)
        sy: int = shared.opts.data.get('tome_stride_y', 2)
        use_rand: bool = shared.opts.data.get('tome_random', True)
        merge_attn: bool = shared.opts.data.get('tome_merge_attention', True)
        merge_crossattn: bool = shared.opts.data.get(
            'tome_merge_cross_attention', False)
        merge_mlp: bool = shared.opts.data.get('tome_merge_mlp', False)

        import tomesd
        tomesd.apply_patch(model,
                           ratio=ratio,
                           max_downsample=max_downsample,
                           sx=sx,
                           sy=sy,
                           use_rand=use_rand,
                           merge_attn=merge_attn,
                           merge_crossattn=merge_crossattn,
                           merge_mlp=merge_mlp)

        print(
            f"Applying ToMe patch with ratio[{ratio}], "
            f"max_downsample[{max_downsample}], sx[{sx}], sy[{sy}], "
            f"use_rand[{use_rand}], merge_attn[{merge_attn}], "
            f"merge_crossattn[{merge_crossattn}], merge_mlp[{merge_mlp}]",
            file=sys.stderr)

    def on_ui_settings_callback(self):
        if not launch.is_installed("tomesd"):
            print(
                "Cannot import tomesd, please install it manually following the instructions on https://github.com/dbolya/tomesd",
                file=sys.stderr)
            return

        section = ('tome', 'ToMe Settings')
        shared.opts.add_option(
            "tome_merging_ratio",
            shared.OptionInfo(
                0.5,
                "ToMe merging ratio, higher the faster, it should not go over 1-(1/(sx*sy)), which is 0.75 by default",
                gr.Slider, {
                    "minimum": 0,
                    "maximum": 0.99,
                    "step": 0.01
                },
                section=section))
        shared.opts.add_option(
            "tome_min_x",
            shared.OptionInfo(
                768,
                "Only activate tome if image width reach this value",
                gr.Slider, {
                    "minimum": 512,
                    "maximum": 2048,
                    "step": 256
                },
                section=section))
        shared.opts.add_option(
            "tome_min_y",
            shared.OptionInfo(
                768,
                "Only activate tome if image height reach this value",
                gr.Slider, {
                    "minimum": 512,
                    "maximum": 2048,
                    "step": 256
                },
                section=section))
        shared.opts.add_option(
            "tome_random",
            shared.OptionInfo(
                False,
                "Use random perturbations - Disable if you see werid artifacts",
                gr.Checkbox,
                section=section))
        shared.opts.add_option(
            "tome_merge_attention",
            shared.OptionInfo(True,
                              "Merge attention (recommended)",
                              gr.Checkbox,
                              section=section))
        shared.opts.add_option(
            "tome_merge_cross_attention",
            shared.OptionInfo(False,
                              "Merge cross attention (not recommended)",
                              gr.Checkbox,
                              section=section))
        shared.opts.add_option(
            "tome_merge_mlp",
            shared.OptionInfo(False,
                              "Merge mlp (very not recommended)",
                              gr.Checkbox,
                              section=section))
        shared.opts.add_option(
            "tome_maximum_down_sampling",
            shared.OptionInfo("1",
                              "Maximum down sampling",
                              gr.Radio, {"choices": ["1", "2", "4", "8"]},
                              section=section))
        shared.opts.add_option(
            "tome_stride_x",
            shared.OptionInfo(2,
                              "Stride - X",
                              gr.Slider, {
                                  "minimum": 2,
                                  "maximum": 8,
                                  "step": 2
                              },
                              section=section))
        shared.opts.add_option(
            "tome_stride_y",
            shared.OptionInfo(2,
                              "Stride - Y",
                              gr.Slider, {
                                  "minimum": 2,
                                  "maximum": 8,
                                  "step": 2
                              },
                              section=section))


_tome_instance = ToMe()
scripts.script_callbacks.on_ui_settings(_tome_instance.on_ui_settings_callback)


class Script(scripts.Script):
    """Use script interface to inject into image generation loop"""

    def title(self):
        return "ToMe"

    def show(self, is_img2img: bool):
        return scripts.AlwaysVisible

    def ui(self, is_img2img: bool):
        if launch.is_installed("tomesd"):
            return [
                gr.Checkbox(
                    value=True,
                    label="Enable ToMe optimization",
                    info=
                    "Use tomesd to boost generation speed. Other settings in Settings Tab.",
                    interactive=True)
            ]
        else:
            return [
                gr.Checkbox(
                    value=False,
                    label="Enable ToMe optimization",
                    info=
                    "Import tomesd failed, please check your environment before using ToMe extension.",
                    interactive=False)
            ]

    def process(self, p: StableDiffusionProcessing, *args):
        # patch all, unload when postprocess
        if args and args[0]:
            tome_min_x = shared.opts.data.get('tome_min_x', 768)
            tome_min_y = shared.opts.data.get('tome_min_y', 768)

            if p.width >= tome_min_x and p.height >= tome_min_y:
                _tome_instance.patch_model(p.sd_model)
                return

            print(
                f'Image size [{p.width}*{p.height}] < '
                f'Threashold [{tome_min_x}*{tome_min_y}]'
                ', no need to apply ToMe patch',
                file=sys.stderr)

    def postprocess(self, p: StableDiffusionProcessing, processed: Processed,
                    *args):
        # remove patch all
        if args and args[0]:
            import tomesd
            tomesd.remove_patch(p.sd_model)
