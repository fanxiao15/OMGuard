python OMGuard_finetune.py \
  --model_path xxx/Qwen3-VL-8B-Instruct \
  --train_json datas/train_subset_finetune.json \
  --output_dir ./output/Qwen3-VL-8B \
  --task_name Misleading_Detection \
  --val_ratio 0.1 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4