# OS-Atlas: A Foundation Action Model For Generalist GUI Agents

<div align="center">

[\[🏠Homepage\]](https://osatlas.github.io) [\[💻Code\]](https://github.com/OS-Copilot/OS-Atlas) [\[🚀Quick Start\]](#quick-start) [\[📗Paper\]](https://arxiv.org/abs/2410.23218) [\[🤗Models\]](https://huggingface.co/collections/OS-Copilot/os-atlas-67246e44003a1dfcc5d0d045)[\[🤗Data\]](https://huggingface.co/datasets/OS-Copilot/OS-Atlas-data) [\[🤗ScreenSpot-v2\]](https://huggingface.co/datasets/OS-Copilot/ScreenSpot-v2) [📝Poster\]](https://github.com/OS-Copilot/OS-Atlas/assets/OS-Atlas-Poster.pdf)

</div>

## Overview

**OS-Atlas** paper is accepted by ICLR 2025.

![os-atlas](https://github.com/user-attachments/assets/cf2ee020-5e15-4087-9a7e-75cc43662494)

<!-- ## TODO List
- [] 
- [ ]
- [ ]
- [ ] -->

## Quick Start
OS-Atlas provides two base grounding models: [OS-Atlas-Base-4B](https://huggingface.co/OS-Copilot/OS-Atlas-Base-4B) and [OS-Atlas-Base-7B](https://huggingface.co/OS-Copilot/OS-Atlas-Base-7B). OS-Atlas-Base-4B is finetuned from [InternVL2-4B](https://huggingface.co/OpenGVLab/InternVL2-4B), and OS-Atlas-Base-7B is finetuned from [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct).

This section provides instructions on how to inference our pre-trained grounding models.

**Notes:** Our models accept images of any size as input. The model outputs are normalized to relative coordinates within a 0-1000 range (either a center point or a bounding box defined by top-left and bottom-right coordinates). For visualization, please remember to convert these relative coordinates back to the original image dimensions.

### OS-Atlas-Base-4B
First, install the `transformers` library:
```
pip install transformers
```
For additional dependencies, please refer to the [InternVL2 documentation](https://internvl.readthedocs.io/en/latest/get_started/installation.html)

Inference code example:
```python
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
path = 'OS-Copilot/OS-Atlas-Base-4B'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

# set the max number of tiles in `max_num`
pixel_values = load_image('./examples/images/web_dfacd48d-d2c2-492f-b94c-41e6a34ea99f.png', max_num=6).to(torch.bfloat16).cuda()
generation_config = dict(max_new_tokens=1024, do_sample=True)

question = "In the screenshot of this web page, please give me the coordinates of the element I want to click on according to my instructions(with point).\n\"'Champions League' link\""
response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')
```


### OS-Atlas-Base-7B
First, ensure that the necessary dependencies are installed:
```
pip install transformers
pip install qwen-vl-utils
```

Inference code example:
```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "OS-Copilot/OS-Atlas-Base-7B", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("OS-Copilot/OS-Atlas-Base-7B")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "./examples/images/web_6f93090a-81f6-489e-bb35-1a2838b18c01.png",
            },
            {"type": "text", "text": "In this UI screenshot, what is the position of the element corresponding to the command \"switch language of current page\" (with bbox)?"},
        ],
    }
]


# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)

generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
)
print(output_text)
# <|object_ref_start|>language switch<|object_ref_end|><|box_start|>(576,12),(592,42)<|box_end|><|im_end|>
```


## Citation
If you find this repository helpful, feel free to cite our paper:
```bibtex
@article{wu2024atlas,
        title={OS-ATLAS: A Foundation Action Model for Generalist GUI Agents},
        author={Wu, Zhiyong and Wu, Zhenyu and Xu, Fangzhi and Wang, Yian and Sun, Qiushi and Jia, Chengyou and Cheng, Kanzhi and Ding, Zichen and Chen, Liheng and Liang, Paul Pu and others},
        journal={arXiv preprint arXiv:2410.23218},
        year={2024}
      }
```
