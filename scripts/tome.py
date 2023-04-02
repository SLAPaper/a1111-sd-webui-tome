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

import modules.scripts as scripts
import modules.shared as shared
import launch
import gradio as gr
import tomesd
import sys
import torch


class ToMe:
    """ToMe implementation"""

    def __init__(self):
        self._has_tomesd = False
        if launch.is_installed("tomesd"):
            self._has_tomesd = True

    def on_model_loaded_callback(self, model: torch.nn.Module):
        if shared.opts.data.get('tome_enable', False):
            ratio: float = shared.opts.data.get('tome_merging_ratio', 0.5)
            max_downsample: int = int(
                shared.opts.data.get('tome_maximum_down_sampling', 1))
            sx: int = shared.opts.data.get('tome_stride_x', 2)
            sy: int = shared.opts.data.get('tome_stride_y', 2)
            use_rand: bool = shared.opts.data.get('tome_random', True)
            merge_attn: bool = shared.opts.data.get('tome_merge_attention',
                                                    True)
            merge_crossattn: bool = shared.opts.data.get(
                'tome_merge_cross_attention', False)
            merge_mlp: bool = shared.opts.data.get('tome_merge_mlp', False)

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
        else:
            tomesd.remove_patch(model)

            print(f"ToMe patch is not enabled", file=sys.stderr)

    def on_ui_settings_callback(self):
        section = ('tome', 'ToMe Settings')
        shared.opts.add_option(
            "tome_enable",
            shared.OptionInfo(
                False,
                "Enable ToMe optimization if you installed tomesd",
                gr.Checkbox, {"interactive": True},
                section=section))
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
            "tome_random",
            shared.OptionInfo(
                True,
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
scripts.script_callbacks.on_model_loaded(
    _tome_instance.on_model_loaded_callback)
scripts.script_callbacks.on_ui_settings(_tome_instance.on_ui_settings_callback)
