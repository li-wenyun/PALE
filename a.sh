export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=0 python hal_det_llama.py --dataset_name triviaqa --model_name llama2_chat_7B --most_likely 0 --use_rouge 0 --generate_gt 1