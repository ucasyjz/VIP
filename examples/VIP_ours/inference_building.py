import torch
from build.lib.diffusers.pipelines import StableDiffusionInpaintPipeline
from diffusers.utils import load_image, make_image_grid
import os 
from tqdm import tqdm

path = "" ######## weights path #########
pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    path, torch_dtype=torch.float32, low_cpu_mem_usage=False,safety_checker = None,
    requires_safety_checker = False
)
root = "./test_data/building/"
path_img = root + "masked"
prompt_list = ['Center:; Surrounding:'] * len(os.listdir(path_img))
n = 0
for i in tqdm(os.listdir(path_img)):
    if not i.endswith('jpg') and not i.endswith('png'):
        continue
    size = 192
    masked_image = os.path.join(path_img,i)
    init_image = load_image(masked_image).resize((size, size))
    mask_image = load_image(root + "mask.jpg").resize((size, size))
    pipeline = pipeline.to('cuda')
    prompt = prompt_list[n]
    generator = torch.Generator("cuda").manual_seed(42)
    image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, height=size,width=size, guidance_scale=7.5,generator=generator).images[0]
    save_path = root + "inference_results/"
    os.makedirs(save_path, exist_ok=True)
    image.save(save_path + '/' + i)
    n += 1


