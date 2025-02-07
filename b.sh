export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=2 python hal_det_llama.py --dataset_name triviaqa --model_name llama2_chat_7B --use_rouge 0 --most_likely 1 --weighted_svd 1 --feat_loc_svd 3
