
"""
# hugging face token
## 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

# fresh venv (optional)
python -m venv qwen-env 
source qwen-env/bin/activate

# install PyTorch that matches your CUDA
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision

# install the latest vLLM with serving extras
pip install "vllm[serve]>=0.8.5"  # pulls in ray, starlette, etc. :contentReference[oaicite:0]{index=0}

###############################################################################################################################################################################
# HF_TOKEN is only needed if the machine is rate-limited by the Hub

export HUGGING_FACE_HUB_TOKEN=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

tmux new -s vllm
source qwen-env/bin/activate
ss -ltnp | grep ':18000' || echo "✓ 18000 looks free"


## for GPU  machines

# --- start vLLM -------------------------------------------------

vllm serve "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --dtype float16 \
    --port 18001 \
    --host 0.0.0.0 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.70 \
    --swap-space 0 \
    --tokenizer-pool-type none \
    2>&1 | tee ~/qwen7b_fp16.log
  

vllm serve "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --dtype float16 \
    --port 18001 \
    --host 0.0.0.0 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.88 \
    --enforce-eager \
    --swap-space 0 \
    --tokenizer-pool-type none


  
  
####### QWEN 2.5 0.5B

## for smaller machines
# --- start vLLM -------------------------------------------------

vllm serve "Qwen/Qwen2.5-0.5B" \
    --dtype float16 \
    --port 18000 \
    --host 0.0.0.0 \
    --tokenizer-pool-type none \
    --swap-space 0 \
    --uvicorn-log-level info \
    2>&1 | tee ~/vllm.log

# ----------------------------------------------------------------

## run it

curl http://localhost:18000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"Qwen/Qwen2.5-0.5B",
          "messages":[{"role":"user","content":"Give me a one-line Python lambda that reverses a string."}],
          "temperature":0.2,"max_tokens":64}'




####### QWEN 2.5 1.5B Instruct

## for smaller machines
# --- start vLLM -------------------------------------------------

vllm serve "Qwen/Qwen2.5-0.5B" \
    --dtype float16 \
    --port 18000 \
    --host 0.0.0.0 \
    --tokenizer-pool-type none \
    --swap-space 0 \
    --uvicorn-log-level info \
    2>&1 | tee ~/vllm.log

# ----------------------------------------------------------------

## run it

curl http://localhost:18000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"Qwen/Qwen2.5-0.5B",
          "messages":[{"role":"user","content":"Give me a one-line Python lambda that reverses a string."}],
          "temperature":0.2,"max_tokens":64}'




####### 7B

## for smaller machines
# --- start vLLM -------------------------------------------------

vllm serve "meta-llama/CodeLlama-7b-Python-hf" \
    --dtype float16 \
    --port 18000 \
    --host 0.0.0.0 \
    --tokenizer-pool-type none \
    --swap-space 0 \
    --uvicorn-log-level info \
    2>&1 | tee ~/vllm.log

##FF
vllm serve "meta-llama/CodeLlama-7b-Python-hf" \
  --dtype float16 \
  --port 18000 \
  --host 0.0.0.0 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --tokenizer-pool-type none \
  --swap-space 0 \
  --uvicorn-log-level info \
  2>&1 | tee ~/vllm.log
    
# ----------------------------------------------------------------

## run it

# Call the server using curl:
curl -X POST "http://localhost:18000/v1/completions" \
	-H "Content-Type: application/json" \
	--data '{
		"model": "meta-llama/CodeLlama-7b-Python-hf",
		"prompt": "Once upon a time,",
		"max_tokens": 512,
		"temperature": 0.5
	}'
          


"""