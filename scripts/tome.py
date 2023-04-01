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


class ToMe:
    """ToMe implementation"""

    def __init__(self):
        self._has_tomesd = False
        if launch.is_installed("tomesd"):
            self._has_tomesd = True

    def on_model_loaded_callback(self, model):
        if shared.opts.data.get('tome_enable', False):
            ratio: float = shared.opts.data.get('tome_merging_ratio', 0.5)
            tomesd.apply_patch(model, ratio=ratio)

            print(f"Applying ToMe patch with ratio[{ratio}]", file=sys.stderr)
        else:
            tomesd.remove_patch(model)

            print(f"ToMe patch is not enabled", file=sys.stderr)

    def on_ui_settings_callback(self):
        section = ('tome', 'ToMe Settings')
        shared.opts.add_option(
            "tome_enable",
            shared.OptionInfo(
                False,
                "Enable ToMe optimization if you installed tomesd [apply when loading checkpoints]",
                gr.Checkbox, {"interactive": True},
                section=section))
        shared.opts.add_option(
            "tome_merging_ratio",
            shared.OptionInfo(
                0.5,
                "ToMe merging ratio, higher the faster, at the cost of generation quality (slightly) [apply when loading checkpoint]",
                gr.Slider, {
                    "minimum": 0.1,
                    "maximum": 0.9,
                    "step": 0.1
                },
                section=section))


_tome_instance = ToMe()
scripts.script_callbacks.on_model_loaded(
    _tome_instance.on_model_loaded_callback)
scripts.script_callbacks.on_ui_settings(_tome_instance.on_ui_settings_callback)
