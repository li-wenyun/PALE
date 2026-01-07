# PALE: Prompt-guided data Augmented haLlucination dEtection

This repository contains the source code for the AAAI'26 paper [**Bolster Hallucination Detection via Prompt-Guided Data Augmentation**](https://arxiv.org/abs/2510.15977) by Wenyun Li, Zheng Zhang, Dongmei Jiang, Xiangyuan Lan.

## Abstract

Large language models (LLMs) have garnered significant interest in AI community. Despite their impressive generation capabilities, they have been found to produce misleading or fabricated information, a phenomenon known as hallucinations. Consequently, hallucination detection has become critical to ensure the reliability of LLM-generated content. One primary challenge in hallucination detection is the scarcity of well-labeled datasets containing both truthful and hallucinated outputs. To address this issue, we introduce Prompt-guided data Augmented haLlucination dEtection (PALE), a novel framework that leverages prompt-guided responses from LLMs as data augmentation for hallucination detection. This strategy can generate both truthful and hallucinated data under prompt guidance at a relatively low cost. To more effectively evaluate the truthfulness of the sparse intermediate embeddings produced by LLMs, we introduce an estimation metric called the Contrastive Mahalanobis Score (CM Score). This score is based on modeling the distributions of truthful and hallucinated data in the activation space. CM Score employs a matrix decomposition approach to more accurately capture the underlying structure of these distributions. Importantly, our framework does not require additional human annotations, offering strong generalizability and practicality for real-world applications. Extensive experiments demonstrate that PALE achieves superior hallucination detection performance, outperforming the competitive baseline by a significant margin of 6.55%.

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (for model inference)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/PALE.git
cd PALE
```

2. Create and activate conda environment:
```bash
conda env create -f env.yml
conda activate pale
```

3. Install additional dependencies:
```bash
pip install -r requirements.txt
```

## Model Preparation

Download the required models:

1. **LLaMA-2 models** (7B/13B): Download from [HuggingFace](https://huggingface.co/meta-llama) and place in the `models/` directory:
```bash
mkdir models
# Place downloaded LLaMA-2 models in models/
```

2. **OPT models** (6.7B/13B): Download from [HuggingFace](https://huggingface.co/facebook/opt-6.7b) and place in the `models/` directory.

3. **BLEURT model**: For ground truth evaluation, download the BLEURT model:
```bash
# The code will automatically download BLEURT-20 when needed
```

## Quick Start

### 0. Prompt-Guided data augmentation
For prompt-guided data augmentation, reference [**HaluEval**](https://arxiv.org/abs/2510.15977).


### 1. Generate LLM Responses

First, generate responses from LLMs for your dataset:

```bash
# For TruthfulQA with LLaMA-2 7B
CUDA_VISIBLE_DEVICES=0 python hal_generate.py --dataset_name tqa --model_name step-1-8k --most_likely True --num_gene 1

# For TriviaQA
CUDA_VISIBLE_DEVICES=0 python hal_generate.py --dataset_name triviaqa --model_name step-1-8k --most_likely True --num_gene 1
```

**Parameters:**
- `--dataset_name`: Dataset name (`tqa`, `triviaqa`, `tydiqa`, `coqa`)
- `--model_name`: Model name (`step-1-8k`, `moonshot-v1-8k`, etc.)
- `--most_likely`: Whether to generate most likely answers (True) or sample multiple answers (False)
- `--num_gene`: Number of generations per question

### 2. Generate Ground Truth Labels

Generate ground truth labels using ROUGE or BLEURT:

```bash
# Using BLEURT (recommended)
CUDA_VISIBLE_DEVICES=0 python hal_gt.py --dataset_name tqa --model_name step-1-8k --most_likely True --use_rouge False

# Using ROUGE
CUDA_VISIBLE_DEVICES=0 python hal_gt.py --dataset_name tqa --model_name step-1-8k --most_likely True --use_rouge True
```

### 3. Run Hallucination Detection

Perform hallucination detection using PALE framework:

```bash
# For TruthfulQA with LLaMA-2 7B
CUDA_VISIBLE_DEVICES=0 python hal_det_llama.py --dataset_name tqa --model_name step-1-8k --use_rouge False --most_likely True --weighted_svd 1 --feat_loc_svd 3


```

**Key Parameters:**
- `--feat_loc_svd`: Feature location in transformer block (1: attention head output, 2: MLP output, 3: block output)
- `--wild_ratio`: Ratio of wild samples for training
- `--thres_gt`: Threshold for ground truth labeling

## Dataset Support

PALE supports the following datasets:

1. **TruthfulQA (tqa)**: Truthful Question Answering benchmark
2. **TriviaQA**: Reading comprehension dataset
3. **TyDi QA**: Typologically diverse question answering
4. **CoQA**: Conversational Question Answering dataset





## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{li2025bolster,
  title={Bolster Hallucination Detection via Prompt-Guided Data Augmentation},
  author={Li, Wenyun and Zhang, Zheng and Jiang, Dongmei and Lan, Xiangyuan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.

## Acknowledgments

This research was partially supported by the National Natural Science Foundation of China (Grant
Nos. 62372132, 62402252, and 62536003), the Shenzhen Science and Technology Program (Grant No. RCYX20221008092852077) and Guangdong High-Level Talent Programme (Grant No. 2024TQ08X283). The authors would also like to thank Huawei Ascend Cloud Ecological Development Project for providing high-performance Ascend 910 processors.