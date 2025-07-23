
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


  #--train_type lora \

  source qwen-env/bin/activate  # activate the environment


  swift sft \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset MBPP/ModelRuns/datasets_for_FT/dataset_hard_heavy.jsonl \
  --output_dir FineTunedModels/DoRA_Finetuned/qwen2.5-hard-heavy \
  --train_type dora \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --max_length 2048 \
  --torch_dtype bfloat16 \
  --save_strategy epoch \
  --save_total_limit 2 \
  --eval_steps 50 \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 4

V1

swift sft \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset MBPP/ModelRuns/datasets_for_FT/dataset_balanced.jsonl \
  --output_dir FineTunedModels/DoRA_Finetuned_V1/qwen2.5-balanced \
  --train_type dora \
  --lora_rank 16 \
  --num_train_epochs 5 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --max_grad_norm 0.3 \
  --max_length 2048 \
  --torch_dtype bfloat16 \
  --save_strategy epoch \
  --save_total_limit 2 \
  --eval_steps 50 \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 4

"""
