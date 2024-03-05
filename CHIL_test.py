import os
import csv
import json
import time
from datetime import datetime, timedelta
import requests
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import AutoModel

import random
import numpy as np

from transformers import pipeline

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def json_reader(file_name):
    f = open(file_name)
    data = json.load(f)
    f.close()
    return data

def get_response1(question, seed):
    set_seed(seed)
    model_input = tokenizer(question, return_tensors="pt").to(device)

    model.eval()
    with torch.no_grad():
        ans = tokenizer.decode(model.generate(**model_input, max_new_tokens=100, repetition_penalty=1.15, do_sample=True, temperature=0.7)[0], skip_special_tokens=True)

    return ans

def get_response2(model, checkpoint_path, question, seed=0):
    set_seed(seed)
    pl = pipeline("text-generation", model=checkpoint_path, tokenizer="medalpaca/medalpaca-13b")
    response = pl(f"Question: {question}\n\nAnswer: ", max_length=256)

    return response


model_name = 'biomistral' #'healthalpaca-13b' #'biomedgpt'

if model_name == "biomedgpt":
    tokenizer = AutoTokenizer.from_pretrained("PharMolix/BioMedGPT-LM-7B")
    model = AutoModelForCausalLM.from_pretrained("PharMolix/BioMedGPT-LM-7B").to(device)

elif model_name == 'healthalpaca-13b':
    model = 'healthalpaca-13b'
    checkpoint_path = "./healthalpaca-13b"

elif model_name == 'biomistral':
    model = AutoModelForCausalLM.from_pretrained(
        "BioMistral/BioMistral-7B-DARE",
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("BioMistral/BioMistral-7B-DARE", add_bos_token=True, trust_remote_code=True)


data_task_dict = {'pmdata':['fatigue', 'stress', 'readiness', 'sleep_quality'], 'awfb':['activity', 'calories'], 'lifesnaps':['stress_resilience', 'sleep_disorder'], 'globem': ['anxiety', 'depression']}
for k,v in data_task_dict.items():
    print("|__Dataset:", k)
    for task in v:
        print("|___Subtask:", task)

        for seed in [0,1,2]:
            print("|____Seed:", seed)

            if os.path.exists("output/{}_{}_{}_sd{}.json".format(model_name, k, task, seed)):
                print("[INFO] skipping ...")
                continue

            data = json_reader('data/{}_{}/step1_ut_final.json'.format(k, task))
            
            res = []
            num_samples = 0
            for _data in tqdm(data):
                if num_samples > 50:
                    break

                try:
                    question = _data['question']
                    answer = _data['answer'] 
                except:
                    question = _data['input']
                    answer = _data['output'] 

                label = _data['answer']

                try:
                    if model_name in ['biomedgpt', 'biomistral']:
                        response = get_response1(question, seed)
                    else:
                        response = get_response2(model, checkpoint_path, question)
                except Exception as e:
                    print(e)
                    continue

                print("question:", question)
                print("response:", response)
                print()
                print()
                res.append({"question": question, "answer": response, 'label': label})
                num_samples += 1
                
            json_object = json.dumps(res, indent=4)
        
            with open("output/{}_{}_{}_sd{}.json".format(model_name, k, task, seed), "w") as outfile:
                outfile.write(json_object)