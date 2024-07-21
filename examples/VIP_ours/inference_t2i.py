import torch
from build.lib.diffusers.pipelines import StableDiffusionInpaintPipeline
from diffusers.utils import load_image, make_image_grid
import os 
path = "" ####### weight path ########
pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    path, torch_dtype=torch.float32, low_cpu_mem_usage=False, safety_checker = None,
    requires_safety_checker = False
)
path_img = "" ######## input img path #########
prompts = ['Center:xxx; Surrounding:xxx']
for prompt in prompts:
    for i in os.listdir(path_img):
        print(i)
        if not i.endswith('jpg') and not i.endswith('png'):
            continue
        size = 512
        masked_image = os.path.join(path_img,i)
        mask = os.path.join(path_img,i.replace('.jpg', '_mask.jpg'))
        init_image = load_image(masked_image).resize((size, size))
        mask_image = load_image(mask).resize((size, size))
        pipeline = pipeline.to('cuda')
        generator = torch.Generator("cuda").manual_seed(42)
        image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, height=size,width=size, guidance_scale=7.5,generator=generator).images[0]
        save_path = "" ######## output img path #########
        os.makedirs(save_path, exist_ok=True)
        image.save(save_path + i)

