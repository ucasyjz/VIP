import requests
import io
import json
import random
import os

from pathlib import Path
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T

def get_prompt_ids(prompt, tokenizer):
    prompt_ids = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
    ).input_ids
    return prompt_ids

class ImageDataset(Dataset):    
    def __init__(
        self,
        tokenizer = None,
        width: int = 256,
        height: int = 256,
        base_width: int = 256,
        base_height: int = 256,
        use_caption:     bool = False,
        image_dir: str = '',
        single_img_prompt: str = '',
        use_bucketing: bool = False,
        fallback_prompt: str = '',
        total_img_num: int = 50000,
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.img_types = (".png", ".jpg", ".jpeg", '.bmp')
        self.use_bucketing = use_bucketing
        self.total_img_num = total_img_num
        
        self.image_dir, self.len = self.get_images_list_online(image_dir)
        self.fallback_prompt = fallback_prompt

        self.use_caption = use_caption
        self.single_img_prompt = single_img_prompt

        self.width = width
        self.height = height

    def get_images_list_online(self, image_dir):
        full_img_dir = []
        if os.path.exists(image_dir):
            txt_path = os.path.join(image_dir,'enhance_file.txt')
            lines = open(txt_path).readlines()
            random.shuffle(lines)
            for line in lines:
                print(line)
                name = os.path.join(image_dir,'processed') + '/' + line.split('-')[0] + '.jpg'
                print(line.split('-'))
                prompt = line.split('-')[1]
                try:
                    center_prompt,surrounding_prompt = prompt.split(';')[0], prompt.split(';')[1][1:]
                except:
                    center_prompt,surrounding_prompt = 'Center:', 'Surrounding:'
                full_img_dir.append(
                    {
                        "name": name,
                        "prompt": prompt,
                        'center': center_prompt,
                        'surrounding': surrounding_prompt,
                    }
                )
            return full_img_dir, len(full_img_dir)
        return ['']

    def image_batch(self, index):
        train_data = self.image_dir[index]
        img_name = train_data['name']
        prompt = train_data['prompt']
        center_prompt = train_data['center']
        surrounding_prompt = train_data['surrounding']
        img = T.transforms.PILToTensor()(Image.open(img_name).convert("RGB"))
        width = self.width
        height = self.height

        if self.use_bucketing:
            _, h, w = img.shape
            width, height = self.width, self.height

        # v0.2b之后的处理方式
        original_h, original_w = img.shape[-2:]
        crop_scale = min(original_h/height, original_w/width)
        crop_size = (int(crop_scale*height), int(crop_scale*width))
        transform = T.Compose([
            # T.transforms.CenterCrop(crop_size),
            T.Resize([height,width])
        ]) # height, width 是目标 h w
        img = transform(img)
        # print(prompt)
        prompt_ids = get_prompt_ids(prompt, self.tokenizer)
        center_prompt_ids = get_prompt_ids(center_prompt, self.tokenizer)
        surrounding_prompt_ids = get_prompt_ids(surrounding_prompt, self.tokenizer)

        return img, prompt, prompt_ids,center_prompt_ids,surrounding_prompt_ids

    @staticmethod
    def __getname__(): return 'image'
    
    def __len__(self):
        # Image directory
        return self.len

    def __getitem__(self, index):
        try:
            img, prompt, prompt_ids,center_prompt_ids,surrounding_prompt_ids = self.image_batch(index)
        except:
            img, prompt, prompt_ids,center_prompt_ids,surrounding_prompt_ids = self.image_batch(0)
        img = img / 127.5 - 1.0

        ## original mask ###
        w, h = img.shape[1], img.shape[2]
        masked_img = torch.zeros_like(img)
        mask = torch.ones(1,w,h)
        p = 0.25
        mask[:,int(p*w):int((1-p)*w), int(p*h):int((1-p)*h)] = 0
        masked_img = (mask < 0.5) * img

        example = {
            "pixel_values": img,
            "masked_pixel_values": masked_img,
            "mask_values": mask,
            "prompt_ids": prompt_ids,
            "center_prompt_ids": center_prompt_ids,
            "surrounding_prompt_ids": surrounding_prompt_ids,
            "text_prompt": prompt, 
            'dataset': self.__getname__()
        }

        return example
