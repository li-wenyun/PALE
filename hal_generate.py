import os
import time
import torch
import torch.nn.functional as F
import evaluate
from datasets import load_metric
from datasets import load_dataset
import datasets
from tqdm import tqdm
import numpy as np
import pickle
# from utils import get_llama_activations_bau, tokenized_tqa, tokenized_tqa_gen, tokenized_tqa_gen_end_q
from utils import get_hal_prompt, get_qa_prompt, get_truth_prompt
import llama_iti
import pickle
import argparse
import matplotlib.pyplot as plt
from pprint import pprint
from baukit import Trace, TraceDict
from metric_utils import get_measures, print_measures
import re
from torch.autograd import Variable
from openai import OpenAI
import openai

API={
    'gpt-3.5-turbo':{'base_url':"https://api.agicto.cn/v1",'key':''},
    'deepseek-chat':{'base_url':"https://api.deepseek.com/v1",'key':'sk-5f06261529bb44df86d9b2fdbae1a6b5'},
    'qwen-plus':{'base_url':"https://dashscope.aliyuncs.com/compatible-mode/v1",'key':'sk-5be20597fa574155a9e56d7df1acfc7f'},
    'step-1-8k':{'base_url':"https://api.stepfun.com/v1",'key':'2hqEtnMCWe5cugi1mAVWRZat5hydLFG8tEJWPRW5XnxglpWxRBp5W0M0dvPAFXhC3'},
    'moonshot-v1-8k':{'base_url':"https://api.moonshot.cn/v1",'key':'sk-8zjQm3CMAI7qQUWYLgFxSCCQxCOkVfuSkRcs6kNxUZY2L4aV'},
    'ERNIE-3.5-8K':{'base_url':"https://api.agicto.cn/v1",'key':'sk-BmLsx7BClpqtmIwxLNB5pH5lJ36WJ7GxiV3nV5PiwF7Iwauf'},

}

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
    parser.add_argument('--model_name', type=str, default='step-1-8k')
    parser.add_argument('--dataset_name', type=str, default='triviaqa')
    parser.add_argument('--num_gene', type=int, default=1)
    parser.add_argument('--use_api', type=bool, default=True)
    parser.add_argument('--most_likely', type=bool, default=True)
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    parser.add_argument("--instruction", type=str, default='/home/liwenyun/code/haloscope/generation/qa/qa_one-turn_instruction.txt', help='local directory of instruction file.')
    args = parser.parse_args()

    
    if args.use_api:
        # openai.api_base=API[args.model_name]['base_url']
        # openai.api_key=API[args.model_name]['key']
        client = OpenAI(
            api_key = API[args.model_name]['key'],
            base_url = API[args.model_name]['base_url'],
        )

    else:
        MODEL = HF_NAMES[args.model_name] if not args.model_dir else args.model_dir




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
    f = open(args.instruction, 'r', encoding="utf-8")
    instruction = f.read()
    error_output='No output'

    if args.use_api:
        begin_index = 0
        if args.dataset_name == 'tydiqa':
            end_index = len(used_indices)
        else:
            end_index = len(dataset)

        if not os.path.exists(f'./save_for_eval/{args.dataset_name}/'):
            os.mkdir(f'./save_for_eval/{args.dataset_name}/')

        if not os.path.exists(f'./save_for_eval/{args.dataset_name}/{args.model_name}_hal_det/'):
            os.mkdir(f'./save_for_eval/{args.dataset_name}/{args.model_name}_hal_det/')


        if not os.path.exists(f'./save_for_eval/{args.dataset_name}/{args.model_name}_hal_det/answers'):
            os.mkdir(f'./save_for_eval/{args.dataset_name}/{args.model_name}_hal_det/answers')

        if not os.path.exists(f'./save_for_eval/{args.dataset_name}/{args.model_name}_hal_det/hallucinations'):
            os.mkdir(f'./save_for_eval/{args.dataset_name}/{args.model_name}_hal_det/hallucinations')

        if not os.path.exists(f'./save_for_eval/{args.dataset_name}/{args.model_name}_hal_det/truths'):
            os.mkdir(f'./save_for_eval/{args.dataset_name}/{args.model_name}_hal_det/truths')


        for i in range(begin_index, end_index):
            answers = [None] * args.num_gene
            hallucinations= [None] * args.num_gene
            truths = [None] * args.num_gene
            if args.dataset_name == 'tydiqa':
                question = dataset[int(used_indices[i])]['question']
                prompt = get_qa_prompt(dataset[int(used_indices[i])]['context'],question)
                hallucination_prompt=get_hal_prompt(dataset[int(used_indices[i])]['context'],question,instruction)
                truth_prompt=get_truth_prompt(dataset[int(used_indices[i])]['context'],question)
            elif args.dataset_name == 'triviaqa':
                prompt = get_qa_prompt("None",dataset[i]['question'])
                question= dataset[i]['question']
                hallucination_prompt=get_hal_prompt("None",dataset[i]['question'],instruction)
                truth_prompt=get_truth_prompt("None",question)
            elif args.dataset_name == 'coqa':
                prompt = get_qa_prompt("None",dataset[i]['prompt'])
                hallucination_prompt=get_hal_prompt("None",dataset[i]['prompt'],instruction)
            else:
                question = dataset[i]['question']
                prompt = get_qa_prompt("None",question)
                hallucination_prompt=get_hal_prompt("None",question,instruction)

            for gen_iter in range(args.num_gene):
                if args.most_likely:
                    try:
                        response = client.chat.completions.create(
                    model = args.model_name,
                    messages = prompt,
                    max_tokens=256,
                    top_p=1,
                    temperature = 1,
                    )
                        decoded=response.choices[0].message.content
                    except openai.APIStatusError as e:
                        print("error occured!"+str(gen_iter)+"responce {e}")
                        decoded = error_output
                    try:
                        hallucination_response = client.chat.completions.create(
                    model = args.model_name,
                    messages = hallucination_prompt,
                    max_tokens=256,
                    top_p=1,
                    temperature = 1,
                    )
                        hallucination_decoded=hallucination_response.choices[0].message.content
                    except openai.APIStatusError as e:
                        print("error occured!"+str(gen_iter)+"hallucination_responce {e}")
                        hallucination_decoded = error_output
                    if args.dataset_name == 'tydiqa' or args.dataset_name == 'tydiqa':
                        try:
                            truth_response=client.chat.completions.create(
                        model = args.model_name,
                        messages = truth_prompt,
                    max_tokens=256,
                        top_p=1,
                        temperature=1 
                    )
                            truth_decoded=truth_response.choices[0].message.content
                        except openai.APIStatusError as e:
                            print("error occured!"+str(gen_iter)+"truth_responce {e}")
                            truth_decoded =error_output
                    
                    
                    
                else:
                    response = client.chat.completions.create(
                    model = args.model_name,
                    messages = prompt,
                    max_tokens=256,
                    n=1,
                    # best_of=1,
                    top_p=0.5,
                    temperature = 0.5,
                    )
                    
                    hallucination_response = client.chat.completions.create(
                    model = args.model_name,
                    messages = hallucination_prompt,
                    max_tokens=256,
                    n=1,
                    # best_of=1,
                    top_p=0.5,
                    temperature = 0.5,
                    )
                    if args.dataset_name == 'tydiqa' or args.dataset_name == 'tydiqa':
                        truth_response=client.chat.completions.create(
                        model = args.model_name,
                        messages = truth_prompt,
                        top_p=0.5,
                    temperature = 0.5, 
                    )
                    truth_decoded=truth_response.choices[0].message.content
                    decoded=response.choices[0].message.content
                    hallucination_decoded=hallucination_response.choices[0].message.content
                time.sleep(40)


                # decoded = tokenizer.decode(generated[0, prompt.shape[-1]:],
                #                            skip_special_tokens=True)
                if args.dataset_name == 'tqa' or args.dataset_name == 'triviaqa':
                    # corner case.
                    if 'Answer the question concisely' in decoded:
                        decoded = decoded.split('Answer the question concisely')[0]
                    if 'Answer the question concisely' in hallucination_decoded:
                        hallucination_decoded = hallucination_decoded.split('Answer the question concisely')[0]
                if args.dataset_name == 'coqa':
                    if 'Q:' in decoded:
                        decoded = decoded.split('Q:')[0]
                    if 'Q:' in hallucination_decoded:
                        hallucination_decoded = hallucination_decoded.split('Q:')[0]
                answers[gen_iter] = decoded
                hallucinations[gen_iter]=hallucination_decoded
                if args.dataset_name == 'tydiqa' or args.dataset_name == 'tydiqa':
                    truths[gen_iter]=truth_decoded

            
            if args.dataset_name == 'coqa':
                truths=[dataset[i]['answer']]+dataset[i]['additional_answers']
                truths=truths[:args.num_gene]
            elif args.dataset_name == 'tqa':
                truths=[dataset[i]['best_answer']]+dataset[i]['correct_answers']
                truths=truths[:args.num_gene]
            else:
                assert 'Not supported dataset!'

            print('sample: ', i)
            if args.most_likely:
                info = 'most_likely_'
            else:
                info = 'batch_generations_'
            print("Saving answers")
            np.save(f'./save_for_eval/{args.dataset_name}/{args.model_name}_hal_det/answers/' + info + f'hal_det_{args.model_name}_{args.dataset_name}_answers_index_{i}.npy',
                    answers)
            print("Saving hallucinations")
            np.save(f'./save_for_eval/{args.dataset_name}/{args.model_name}_hal_det/hallucinations/' + info + f'hal_det_{args.model_name}_{args.dataset_name}_hallucinations_index_{i}.npy',
                    hallucinations)
            print("Saving truths")
            np.save(f'./save_for_eval/{args.dataset_name}/{args.model_name}_hal_det/truths/' + info + f'hal_det_{args.model_name}_{args.dataset_name}_truths_index_{i}.npy',
                    truths)

    else:
        tokenizer = llama_iti.LlamaTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        model = llama_iti.LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16,
                                                           device_map="auto").cuda()

        begin_index = 0
        if args.dataset_name == 'tydiqa':
            end_index = len(used_indices)
        else:
            end_index = len(dataset)

        if not os.path.exists(f'./save_for_eval/{args.dataset_name}/{args.model_name}_hal_det/'):
            os.mkdir(f'./save_for_eval/{args.dataset_name}/{args.model_name}_hal_det/')


        if not os.path.exists(f'./save_for_eval/{args.dataset_name}/{args.model_name}_hal_det/answers'):
            os.mkdir(f'./save_for_eval/{args.dataset_name}/{args.model_name}_hal_det/answers')

        if not os.path.exists(f'./save_for_eval/{args.dataset_name}/{args.model_name}_hal_det/hallucinations'):
            os.mkdir(f'./save_for_eval/{args.dataset_name}/{args.model_name}_hal_det/hallucinations')

        period_token_id = [tokenizer(_)['input_ids'][-1] for _ in ['\n']]
        period_token_id += [tokenizer.eos_token_id]

        for i in range(begin_index, end_index):
            answers = [None] * args.num_gene
            hallucinations= [None] * args.num_gene
            if args.dataset_name == 'tydiqa':
                question = dataset[int(used_indices[i])]['question']
                prompt = tokenizer(
                    "Concisely answer the following question based on the information in the given passage: \n" + \
                    " Passage: " + dataset[int(used_indices[i])]['context'] + " \n Q: " + question + " \n A:",
                    return_tensors='pt').input_ids.cuda()
                hallucination_prompt=tokenizer(
                    get_hal_prompt(dataset[int(used_indices[i])]['context'],question,instruction), return_tensors='pt'
                ).input_ids.cuda()
            elif args.dataset_name == 'coqa':
                prompt = tokenizer(
                    dataset[i]['prompt'], return_tensors='pt').input_ids.cuda()
                # hallucination_prompt=get_hal_prompt("None",dataset[i]['prompt'],instruction)
                hallucination_prompt=tokenizer(
                   get_hal_prompt("None",dataset[i]['prompt'],instruction) , return_tensors='pt'
                ).input_ids.cuda()
            else:
                question = dataset[i]['question']
                prompt = tokenizer(f"Answer the question concisely. Q: {question}" + " A:", return_tensors='pt').input_ids.cuda()
                # hallucination_prompt=get_hal_prompt("None",question,instruction)
                hallucination_prompt=tokenizer(
                    get_hal_prompt("None",question,instruction), return_tensors='pt'
                ).input_ids.cuda()
            for gen_iter in range(args.num_gene):
                if args.most_likely:
                    generated = model.generate(prompt,
                                                num_beams=5,
                                                num_return_sequences=1,
                                                do_sample=False,
                                                max_new_tokens=128,
                                               )
                    hallucination_generated=model.generate(hallucination_prompt,
                                                num_beams=5,
                                                num_return_sequences=1,
                                                do_sample=False,
                                                max_new_tokens=128,
                                               )
                else:
                    generated = model.generate(prompt,
                                                do_sample=True,
                                                num_return_sequences=1,
                                                num_beams=1,
                                                max_new_tokens=128,
                                                temperature=0.5,
                                                top_p=1.0)
                    hallucination_generated=model.generate(hallucination_prompt,
                                                do_sample=True,
                                                num_return_sequences=1,
                                                num_beams=1,
                                                max_new_tokens=128,
                                                temperature=0.5,
                                                top_p=1.0)

                decoded = tokenizer.decode(generated[0, prompt.shape[-1]:],
                                           skip_special_tokens=True)
                hallucination_decoded=tokenizer.decode(hallucination_generated[0, prompt.shape[-1]:],
                                           skip_special_tokens=True)
                if args.dataset_name == 'tqa' or args.dataset_name == 'triviaqa':
                    # corner case.
                    if 'Answer the question concisely' in decoded:
                        decoded = decoded.split('Answer the question concisely')[0]
                    if 'Answer the question concisely' in hallucination_decoded:
                        hallucination_decoded = hallucination_decoded.split('Answer the question concisely')[0]
                if args.dataset_name == 'coqa':
                    if 'Q:' in decoded:
                        decoded = decoded.split('Q:')[0]
                    if 'Q:' in hallucination_decoded:
                        hallucination_decoded = hallucination_decoded.split('Q:')[0]
                answers[gen_iter] = decoded
                hallucinations[gen_iter]=hallucination_decoded


            print('sample: ', i)
            if args.most_likely:
                info = 'most_likely_'
            else:
                info = 'batch_generations_'
            print("Saving answers")
            np.save(f'./save_for_eval/{args.dataset_name}_hal_det/answers/' + info + f'hal_det_{args.model_name}_{args.dataset_name}_answers_index_{i}.npy',
                    answers)
            print("Saving hallucinations")
            np.save(f'./save_for_eval/{args.dataset_name}_hal_det/hallucinations/' + info + f'hal_det_{args.model_name}_{args.dataset_name}_hallucinations_index_{i}.npy',
                    hallucinations)


        # get the split and label (true or false) of the unlabeled data and the test data.
        



if __name__ == '__main__':
    seed_everything(42)
    main()