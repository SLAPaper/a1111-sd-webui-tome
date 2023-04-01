<!--
 Copyright 2023 SLAPaper
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

# ToMe extension for Stable Diffusion A1111 WebUI

Use [tomesd](https://github.com/dbolya/tomesd) to speed up generation

Related: [official PR of A1111 WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9256)

## Installation

Open a terminal, activate your webui environment (typically, execute the `venv/Scripts/activate` from webui installation path)

Do anything necessary needed if you have a fancy environment settings like me.

And then follow the instruction of [tomesd Installation](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9256)

## Settings

In `Settings` tab, you'll find a section called `ToMe Settings`, there are 2 options:

1. Enable ToMe: self explained
2. ToMe Merging Ratio: default is 0.5, higher the faster, at the cost of slightly generation quality

All the settings apply when you reload the sd model (checkpoint).

## Performance Showcase

Tested on RTX 4090, Python 3.10.9, PyTorch 2.0, CUDA 11.8, CuDNN 8.8.1.3, with `--skip-version-check --xformers --opt-sdp-attention --no-half-vae` enabled, batch count 5, same seed, use best result

Conclusion: seems works with batch size 8, and have little effect when batch size is 1 or 4

In batch 8, it can reach 6.57 it/s (ratio 0.5) or 6.73 it/s (ratio 0.9) which is over 52 it/s on single image, the speedup ratio is 13.47% or 16.23%

Generation Info|Disabled ToMe|ToMe:0.5|ToMe:0.9
---------------|-------------|--------|--------
Eular a, 512*512, batch 1, 30 steps|32.41 it/s|33.37 it/s|33.33 it/s
DPM++ 2M Karras, 512*512, batch 1, 30 steps|32.78 it/s|32.42 it/s|31.79 it/s
DPM++ 2M Karras, 512*512, batch 4, 30 steps|12.01 it/s|12.03 it/s|13.27 it/s
DPM++ 2M Karras, 512*512, batch 8, 30 steps|5.79 it/s|6.57 it/s|6.73 it/s
DPM++ 2M Karras, 512*512, batch 8, 150 steps|5.73 it/s|6.58 it/s|6.76 it/s
-|-|-|-
Eular a, 768*512, batch 1, 30 steps|27.78 it/s|26.98 it/s|27.93 it/s
DPM++ 2M Karras, 768*512, batch 1, 30 steps|28.29 it/s|27.19 it/s|27.44 it/s
-|-|-|-
Eular a, 768*768, batch 1, 30 steps|20.65 it/s|19.75 it/s|20.68 it/s
DPM++ 2M Karras, 768*768, batch 1, 30 steps|20.73 it/s|19.69 it/s|20.76 it/s
