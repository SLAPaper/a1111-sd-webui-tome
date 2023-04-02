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
2. ToMe Merging Ratio: default is 0.5, higher the faster, at the cost of (slightly) generation quality

All the settings apply when you reload the sd model (checkpoint).

## Performance

Tested on RTX 4090, Python 3.10.9, PyTorch 2.0, CUDA 11.8, CuDNN 8.8.1.3, with `--skip-version-check --xformers --opt-sdp-attention --no-half-vae` enabled, step 30, batch count 5, same seed, use best result

Generation Info|Disabled ToMe|ToMe:0.5|ToMe:0.9
---------------|-------------|--------|--------
Eular a, 512*512, batch 1|32.41 it/s|33.37 it/s|33.33 it/s
DPM++ 2M Karras, 512*512, batch 1|32.78 it/s|32.42 it/s|31.79 it/s
DPM++ 2M Karras, 512*512, batch 1, Hires fix 2x|7.74 it/s|9.82 it/s **(+26.87%)**|10.79 it/s **(+39.41%)**
DPM++ 2M Karras, 512*512, batch 1, Hires fix 4x|1.16 s/it|1.50 it/s **(+74.00%)**|1.83 it/s **(+112.28%)**
-|-|-|-
DPM++ 2M Karras, 512*512, batch 4|12.01 it/s|12.03 it/s|13.27 it/s **(+10.49%)**
DPM++ 2M Karras, 512*512, batch 4, Hires fix 2x|1.84 it/s|2.54 it/s **(+38.04%)**|2.83 it/s **(+53.80%)**
-|-|-|-
DPM++ 2M Karras, 512*512, batch 8|5.79 it/s|6.57 it/s **(+13.47%)**|6.73 it/s **(+16.23%)**
-|-|-|-
Eular a, 768*768, batch 1|20.65 it/s|19.75 it/s|20.68 it/s
DPM++ 2M Karras, 768*768, batch 1|20.73 it/s|19.69 it/s|20.76 it/s
-|-|-|-
DPM++ 2M Karras, 1024*1024, batch 1|7.72 it/s|9.88 it/s **(+27.98%)**|10.83 it/s **(+40.28%)**
-|-|-|-
DPM++ 2M Karras, 2048*2048, batch 1|1.15 s/it|1.52 it/s **(+74.80%)**|1.92 it/s **(+120.80%)**

### Conclusion

Works with big image size and big batch size, you will need total pixel of `4*512*512 = 1024*1024` or more to see a difference

The higher the total pixel there are, the more performance boost you'll get, on `2048*2048`, it could be over +100%
