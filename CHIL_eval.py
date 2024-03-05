import os
import csv
import json
import time
from datetime import datetime, timedelta
import requests
from tqdm import tqdm

import torch

import random
import numpy as np
from sklearn.metrics import mean_absolute_error


def json_reader(file_name):
    f = open(file_name)
    data = json.load(f)
    f.close()
    return data

model_name = "biomedgpt" #"healthalpaca-13b"

data_task_dict = {'pmdata':['fatigue', 'stress', 'readiness', 'sleep_quality'], 'awfb':['activity', 'calories'], 'lifesnaps':['stress_resilience', 'sleep_disorder'], 'globem': ['anxiety', 'depression']}
for k,v in data_task_dict.items():
    print("|__Dataset:", k)
    for task in v:
        print("|___Subtask:", task)
        results = []
        for seed in [0,1,2]:

            data = json_reader("output/{}_{}_{}_sd{}.json".format(model_name, k, task, seed))

            num_total = 0
            num_correct = 0  
            y_trues = []
            y_preds = []      
            for d in data:            
                label = d['label']
                if model_name == 'healthalpaca-13b':
                    response = d['answer'][0]['generated_text']
                else:
                    response = d['answer']

                # Accuracy
                if task in ['fatigue', 'sleep_disorder', 'activity']:

                    # if task == 'fatigue':
                    if model_name == "healthalpaca-13b":
                        y_hat = response.split(':')[-1].strip()
                        if y_hat.endswith('.'):
                            y_hat = y_hat[:-1]
                    else:
                        y_hat = response.split('?')[-1].strip()


                    if task == 'fatigue':
                        if y_hat not in ['1', '2', '3', '4', '5']:
                            continue
                    
                    elif task == 'sleep_disorder':
                        if y_hat not in ['0', '1']:
                            continue

                    if label in y_hat:
                        num_correct +=1
                    
                    num_total +=1
                    
                    print("y_hat:", y_hat)
                    print("y:", label)
                    print() 
                
                # MAE
                else:

                    if task == 'calories':
                        y_hat = response.split(':')[-1].strip().split()[0].strip()

                        if y_hat.endswith('.'):
                            y_hat = y_hat[:-1]

                    elif task in ['anxiety', 'depression']:
                        y_hat = response.split(')')[-1].strip()

                    else:
                        y_hat = response.split(':')[-1].strip()


                    # check
                    if task in ['stress', 'sleep_quality']:
                        if y_hat not in ['0', '1', '2', '3', '4', '5']:
                            continue
                    
                    elif task == 'readiness':
                        if y_hat not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
                            continue

                    elif task in ['anxiety', 'depression']:
                        if y_hat not in ['0', '1', '2', '3', '4']:
                            continue
                    
                    try:
                        y_preds.append(float(y_hat))
                        y_trues.append(float(label.strip()))
                    
                    except Exception as e:
                        # print(e)
                        continue


            if task in ['fatigue', 'sleep_disorder', 'activity']:
                acc = num_correct / num_total * 100
                # print("acc:", acc)
                results.append(acc)
            else:            
                if len(y_preds) == 0 or len(y_trues) != len(y_preds):
                    # print("[ERROR] len(y_trues): {}, len(y_preds): {}".format(len(y_trues), len(y_preds)))
                    continue

                mae = mean_absolute_error(y_trues, y_preds)
                # print("mae:", mae)
                results.append(mae)

            # print()

            # exit(0)

        
        try:
            tmp = np.array(results)
            mean = np.mean(tmp)
            std = np.std(tmp, ddof=1)

            print("{:.2f} \textpm \ {:.1f}".format(mean, std))
            print()
        except:
            continue

        exit(0)