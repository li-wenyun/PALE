import os
import torch
import torch.nn.functional as F
import evaluate
from datasets import load_metric
from datasets import load_dataset
import datasets
from tqdm import tqdm
import numpy as np
import pickle
import llama_iti
import pickle
import argparse
import matplotlib.pyplot as plt
from pprint import pprint
from baukit import Trace, TraceDict
from metric_utils import get_measures, print_measures
import re
from torch.autograd import Variable
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer



def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

HF_NAMES = {
    'llama_7B': 'baffo32/decapoda-research-llama-7B-hf',
    'honest_llama_7B': 'validation/results_dump/llama_7B_seed_42_top_48_heads_alpha_15',
    'alpaca_7B': 'circulus/alpaca-7b',
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b',
    'llama2_chat_7B': 'models/Llama-2-7b-chat-hf',
    'llama2_chat_13B': 'models/Llama-2-13b-chat-hf',
    'llama2_chat_70B': 'meta-llama/Llama-2-70b-chat-hf',
}


def main():


    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='llama2_chat_7B')
    parser.add_argument('--model_name', type=str, default='moonshot-v1-8k')
    parser.add_argument('--dataset_name', type=str, default='tqa')
    parser.add_argument('--num_gene', type=int, default=1)
    parser.add_argument('--use_api', type=bool, default=False)
    parser.add_argument('--most_likely', type=bool, default=False)
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    parser.add_argument("--instruction", type=str, default=None, help='local directory of instruction file.')
    parser.add_argument('--use_rouge', type=bool, default=True)
    parser.add_argument('--thres_gt', type=float, default=0.5)

    # parser.add_argument('--model_name', type=str, default='llama2_chat_7B')
    # parser.add_argument('--dataset_name', type=str, default='triviaqa')
    # parser.add_argument('--num_gene', type=int, default=1)
    # parser.add_argument('--gene', type=int, default=0)
    # parser.add_argument('--generate_gt', type=int, default=0)
    # parser.add_argument('--use_rouge', type=int, default=0)
    # parser.add_argument('--weighted_svd', type=int, default=0)
    # parser.add_argument('--feat_loc_svd', type=int, default=0)
    # parser.add_argument('--wild_ratio', type=float, default=0.75)
    # parser.add_argument('--thres_gt', type=float, default=0.5)
    # parser.add_argument('--most_likely', type=int, default=0)

    # parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    args = parser.parse_args()

    MODEL = HF_NAMES[args.model] if not args.model_dir else args.model_dir




    if args.dataset_name == "tqa":
        dataset = load_dataset("truthful_qa", 'generation')['validation']
    elif args.dataset_name == 'triviaqa':
        dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")
        id_mem = set()

        def remove_dups(batch):
            if batch['question_id'][0] in id_mem:
                return {_: [] for _ in batch.keys()}
            id_mem.add(batch['question_id'][0])
            return batch

        dataset = dataset.map(remove_dups, batch_size=1, batched=True, load_from_cache_file=False)
    elif args.dataset_name == 'tydiqa':
        dataset = datasets.load_dataset("tydiqa", "secondary_task", split="train")
        used_indices = []
        for i in range(len(dataset)):
            if 'english' in dataset[i]['id']:
                used_indices.append(i)
    elif args.dataset_name == 'coqa':
        import json
        import pandas as pd
        from datasets import Dataset

        def _save_dataset():
            # https://github.com/lorenzkuhn/semantic_uncertainty/blob/main/code/parse_coqa.py
            save_path = f'./coqa_dataset'
            if not os.path.exists(save_path):
                # https://downloads.cs.stanford.edu/nlp/data/coqa/coqa-dev-v1.0.json
                with open(f'./coqa-dev-v1.0.json', 'r') as infile:
                    data = json.load(infile)['data']

                dataset = {}

                dataset['story'] = []
                dataset['question'] = []
                dataset['answer'] = []
                dataset['additional_answers'] = []
                dataset['id'] = []

                for sample_id, sample in enumerate(data):
                    story = sample['story']
                    questions = sample['questions']
                    answers = sample['answers']
                    additional_answers = sample['additional_answers']
                    for question_index, question in enumerate(questions):
                        dataset['story'].append(story)
                        dataset['question'].append(question['input_text'])
                        dataset['answer'].append({
                            'text': answers[question_index]['input_text'],
                            'answer_start': answers[question_index]['span_start']
                        })
                        dataset['id'].append(sample['id'] + '_' + str(question_index))
                        additional_answers_list = []

                        for i in range(3):
                            additional_answers_list.append(additional_answers[str(i)][question_index]['input_text'])

                        dataset['additional_answers'].append(additional_answers_list)
                        story = story + ' Q: ' + question['input_text'] + ' A: ' + answers[question_index]['input_text']
                        if not story[-1] == '.':
                            story = story + '.'

                dataset_df = pd.DataFrame.from_dict(dataset)

                dataset = Dataset.from_pandas(dataset_df)

                dataset.save_to_disk(save_path)
            return save_path

        # dataset = datasets.load_from_disk(_save_dataset())
        def get_dataset(tokenizer, split='validation'):
            # from https://github.com/lorenzkuhn/semantic_uncertainty/blob/main/code/parse_coqa.py
            dataset = datasets.load_from_disk(_save_dataset())
            id_to_question_mapping = dict(zip(dataset['id'], dataset['question']))

            def encode_coqa(example):
                example['answer'] = [example['answer']['text']] + example['additional_answers']
                example['prompt'] = prompt = example['story'] + ' Q: ' + example['question'] + ' A:'
                return tokenizer(prompt, truncation=False, padding=False)

            dataset = dataset.map(encode_coqa, batched=False, load_from_cache_file=False)
            dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)
            return dataset

        dataset = get_dataset(llama_iti.LlamaTokenizer.from_pretrained(MODEL, trust_remote_code=True))
    else:
        raise ValueError("Invalid dataset name")

    model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20').cuda()
    tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20')
    model.eval()


    # elif args.generate_gt:
        # from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer

        

    rouge = evaluate.load('rouge')
    gts = np.zeros(0)
    if args.dataset_name == 'tydiqa':
        length = len(used_indices)
    else:
        length = len(dataset)
    for i in range(length):
        if args.dataset_name == 'tqa':
            best_answer = dataset[i]['best_answer']
            correct_answer = dataset[i]['correct_answers']
            all_answers = [best_answer] + correct_answer
        elif args.dataset_name == 'triviaqa':
            all_answers = dataset[i]['answer']['aliases']
        elif args.dataset_name == 'coqa':
            all_answers = dataset[i]['answer']
        elif args.dataset_name == 'tydiqa':
            all_answers = dataset[int(used_indices[i])]['answers']['text']

        if args.most_likely:
            answers = np.load(
                f'./save_for_eval/{args.dataset_name}/{args.model_name}_hal_det/answers/most_likely_hal_det_{args.model_name}_{args.dataset_name}_answers_index_{i}.npy')
        else:
            answers = np.load(
                f'./save_for_eval/{args.dataset_name}/{args.model_name}_hal_det/answers/batch_generations_hal_det_{args.model_name}_{args.dataset_name}_answers_index_{i}.npy')
            # get the gt.
        if args.use_rouge:

                predictions = answers
                all_results = np.zeros((len(all_answers), len(predictions)))
                all_results1 = np.zeros((len(all_answers), len(predictions)))
                all_results2 = np.zeros((len(all_answers), len(predictions)))
                for anw in range(len(all_answers)):
                    results = rouge.compute(predictions=predictions,
                                            references=[all_answers[anw]] * len(predictions),
                                            use_aggregator=False)
                    all_results[anw] = results['rougeL']
                    all_results1[anw] = results['rouge1']
                    all_results2[anw] = results['rouge2']

                # breakpoint()
                gts = np.concatenate([gts, np.max(all_results, axis=0)], 0)

                if i % 50 == 0:
                    print("samples passed: ", i)
        else:

                predictions = answers
                all_results = np.zeros((len(all_answers), len(predictions)))
                with torch.no_grad():
                    for anw in range(len(all_answers)):
                        inputs = tokenizer(predictions.tolist(), [all_answers[anw]] * len(predictions),
                                           padding='longest', return_tensors='pt')
                        for key in list(inputs.keys()):
                            inputs[key] = inputs[key].cuda()
                        res = np.asarray(model(**inputs).logits.flatten().tolist())
                        all_results[anw] = res
                gts = np.concatenate([gts, np.max(all_results, axis=0)], 0)
                if i % 10 == 0:
                    print("samples passed: ", i)
        # breakpoint()
    if args.most_likely:
        if args.use_rouge:
            np.save(f'./ml_{args.dataset_name}_{args.model_name}_rouge_score.npy', gts)
        else:
            np.save(f'./ml_{args.dataset_name}_{args.model_name}_bleurt_score.npy', gts)
    else:
        if args.use_rouge:
            np.save(f'./bg_{args.dataset_name}_{args.model_name}_rouge_score.npy', gts)
        else:
            np.save(f'./bg_{args.dataset_name}_{args.model_name}_bleurt_score.npy', gts)

    



        




        



if __name__ == '__main__':
    seed_everything(42)
    main()