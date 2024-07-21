export MODEL_NAME="./stable-diffusion-inpainting"
accelerate launch --mixed_precision="fp16"  train_on_enhanced_prompt.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir='./enhanced_yjz_t2I' \
  --resolution=256 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=10000 \
  --learning_rate=5e-06 \
  --max_grad_norm=1 \
  --checkpointing_steps=1500 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --logging_dir="logs" \
  --output_dir="./C_T_S/" \