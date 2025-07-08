
"""
pip install 'ms-swift[llm]' -U

new_train.jsonl:
{"instruction": "Translate to French: 'Hello world'", "output": "Bonjour le monde"}



swift sft \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset ModelRuns/datasets_for_FT/dataset_balanced.jsonl \
  --train_type full \
  --torch_dtype bfloat16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --max_length 2048 \
  --output_dir output/qwen2.5-finetuned \
  --logging_steps 10 \
  --save_strategy epoch \
  --save_total_limit 2 \
  --eval_steps 50 \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 4


"""
