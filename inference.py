import os
import random
import numpy as np
import json
from tqdm import tqdm

import openai
import google.generativeai as genai

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModel
from transformers import pipeline
from transformers import AutoModelForQuestionAnswering


model = args.model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def json_reader(file_name):
    f = open(file_name)
    data = json.load(f)
    f.close()
    return data

def get_response(model, checkpoint_path, question, seed=0):
    pl = pipeline("text-generation", model=checkpoint_path, tokenizer="medalpaca/medalpaca-13b")
    response = pl(f"Question: {question}\n\nAnswer: ", max_length=256)
    
    return response


model = 'health-alpaca-13b'
checkpoint_path = "medalpaca-13b/tmp-checkpoint-27"
question = "You are a health assistant. Your mission is to read the following input health query and return your prediction.\n"
response = get_response(model, checkpoint_path, question)


print("question:", question)
print("response:", response)
print("label:", answer)
print()
