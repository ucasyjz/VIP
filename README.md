# VIP-Versatile-Image-Outpainting-Empowered-by-Multimodal-Large-Language-Model
This repository is the official implementation of VIP: Versatile Image Outpainting Empowered by Multimodal Large Language Model
## 📜 News
🚀 [2024/7/22] The training and inference code are released!

🚀 [2024/6/3] The [paper](https://arxiv.org/abs/2406.01059) is released!

## 🛠️ Usage
### Requirements
```shell
- torch==1.13.1
- torchvision==0.14.1
- transformers==4.39.3
```
Note that in out method, there are some changes of **UNet2DConditionModel** in diffusers, please don't download the official **diffusers** dependency package.

### For training
```shell
cd examples/VIP_ours/
bash train_on_enhanced_prompt.sh
```
### For inference
```shell
cd examples/VIP_ours/
python3 inference_*.py
```

