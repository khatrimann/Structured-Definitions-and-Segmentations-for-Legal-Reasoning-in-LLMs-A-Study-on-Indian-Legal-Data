#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:06:55 2024

@author: mann
"""

import json
from copy import deepcopy
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report
from llm_client_v3 import LLMClient
import gc
from torch.cuda import empty_cache

# tempr = 0.
# dec_tempr = 0.
# suffix = "_preamble_plaintiff"

dset = "test"
with open(f"{dset}.json", 'r') as dev:
    dev = json.load(dev)
    
with open("analysis_without_defs_without_rr.txt", 'r') as analysis_prompt:
    analysis_prompt = analysis_prompt.read()
    
petitioners = pd.read_csv("plaintiff.csv")
petitioners = [pet if corr == "TRUE" else corr for pet, corr in zip(petitioners.plaintiff.tolist(), petitioners.is_correct.tolist())]
    
for tempr in [1e-22]:
    for dec_tempr in [1e-22]:
        if tempr < 0.01 :
            do_sample_tempr = False
        else:
            do_sample_tempr = True

        if dec_tempr < 0.01:
            do_sample_dec = False
        else:
            do_sample_dec = True

        for suffix in ["_preamble_plaintiff"]:

            EXCLUDE = ["ANALYSIS", "STA", "RPC", "RATIO"]
            if "preamble" not in suffix:
                EXCLUDE.append("PREAMBLE")

            seg = []
            lens = []
            cases = []
            for case_idx, case_ in enumerate(dev):
                filtered = case_['data']['text']
                for result in case_["annotations"][0]["result"]:
                    if result["value"]["labels"][0] in EXCLUDE:
                        filtered = filtered.replace(result["value"]["text"], "")
                dev[case_idx]['fitered'] = {'text': filtered}
                cases.append(analysis_prompt.format(filtered))
                
            model_name = "mistral"
            run_gpt = 1
            if run_gpt:
                outputs = {}
                for run in [1]:
                    for model in ["microsoft/Phi-4-mini-instruct"]:
                    # for model in ["mistralai/Mistral-7B-Instruct-v0.3"]:
                #         # print(f"## {model}")
                        model_name = "MODEL"
                        if "mistral" in model.lower():
                            model_name = "mistral"
                        elif "phi-3" in model.lower():
                            model_name = "phi3"
                        elif "deepseek" in model.lower():
                            model_name = "deepseek-llama"
                        elif "llama" in model.lower():
                            model_name = "llama"
                        elif "phi-4" in model.lower():
                            model_name = "phi4"
                        
                        
                        client = LLMClient(model)
                        for prompt_idx, prompt in tqdm(enumerate(cases), total=len(cases)):
                            prompt = prompt.replace("You are a judge of the Indian Supreme Court. ", "")
                            if prompt not in outputs:
                                # import sys
                                # sys.exit(0)
                                completion = client.create(
                                    # model=model,
                                    messages=[
                                        {"role": "system", "content": "You are an Indian Supreme Court Judge"},
                                        {"role": "user", "content": prompt}
                                    ],
                                    do_sample=do_sample_tempr,
                                    temperature=tempr
                                )
                                
                                prompt = prompt[:prompt.find("What are views of the court. View includes c")].strip()
                                prompt = prompt + "\n\n**ANALYSIS**\n" + completion + f"\n\nBased upon the **ANALYSIS** provided above, ouput YES if the case was in favour of {petitioners[prompt_idx]}, else NO"
                                completion = client.create(
                                    # model=model,
                                    messages=[
                                        {"role": "system", "content": "You are an Indian Supreme Court Judge"},
                                        {"role": "user", "content": prompt},
                                    ],
                                    do_sample=do_sample_dec,
                                    temperature=dec_tempr
                                )
                            
                                outputs[prompt] = completion
                                # import sys
                                # sys.exit(0)

                        del client
                        gc.collect()
                        empty_cache()
                
                with open(f"ljp_{model_name}_without_defs_without_rr_without_chain{suffix}_{tempr}_{dec_tempr}.json", 'w') as jsonf:
                    json.dump(outputs, jsonf)
                    
            with open(f"ljp_{model_name}_without_defs_without_rr_without_chain{suffix}_{tempr}_{dec_tempr}.json", 'r') as jsonf:
                outputs = json.load(jsonf)
                
            output = {}
            labs = []
            for prompt_idx, prompt in tqdm(enumerate(outputs)):
                output[prompt_idx] = {"y": None, "n": None}
                
                output[prompt_idx]["n"] = outputs[prompt].count("NO")
                output[prompt_idx]["y"] = outputs[prompt].count("YES")
                
                if output[prompt_idx]["y"] == 0:
                    labs.append(0)
                else:
                    labs.append(1)
                
                    
            labels = labs
            # labels = list(map(lambda x: 1 if x == "YES" else 0, labels))
            ground_col_name = "Ground"
            grounds = pd.read_excel("grounds_ljp_chain.ods", engine='odf')[ground_col_name].tolist()

            idx = 0
            while len(grounds) != idx:
                if grounds[idx] == -1:
                    del grounds[idx]
                    del labels[idx]
                else:
                    idx += 1
                    
            report = classification_report(grounds, labels)
            print(report)
            diff = np.logical_and(grounds, labels)
