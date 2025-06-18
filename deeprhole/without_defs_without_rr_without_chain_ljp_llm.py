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
import os

# tempr = 0.
# dec_tempr = 0.
# suffix = "_preamble_plaintiff"
    
with open("analysis_without_defs_without_rr.txt", 'r') as analysis_prompt:
    analysis_prompt = analysis_prompt.read()
        
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

        EXCLUDE = ["STA", "RPC", "Ratio"]

        root = "MARRO_Rhetorical-Role-Labeling/dataset/IN-dataset"
        petitioners = pd.read_csv(f"ground_truths_{'uk' if 'UK' in root else 'in'}.csv").Plaintiff.tolist()

        files = sorted(os.listdir(root))

        cases = {}
        labels = set()

        seg = []
        lens = []
        for file in files:
            file_name = file
            
            cases[file_name] = {}
            file = os.path.join(root, file)
            with open(file, 'r') as file:
                file = file.readlines()

            for line_label in file:
                line, label = line_label.split("\t")
                label = label.strip()
                
                if label not in cases[file_name]:
                    cases[file_name][label] = []
                    
                cases[file_name][label].append(line)
                labels.add(label)

            seg.append({"RR": {rr: '\n'.join(cases[file_name][rr]) for rr in cases[file_name]}, "text": '\n'.join([line.split("\t")[0] for line in file])})
            lens.append(len(seg[-1]))
            
        cases = []
        
        
        for case_idx, case_ in enumerate(files):

            file_name = file
            
            file = os.path.join(root, case_)
            with open(file, 'r') as file:
                file = file.readlines()

            sequence = set()
            filtered = '\n'.join([line.split("\t")[0] for line in file if line.split("\t")[1].strip() not in EXCLUDE])
            tmp_case = ""
            for line in file:
                line, label = line.split("\t")
                label = label.strip()
                line = line.strip()
                
                if label not in EXCLUDE:
                    
                    sequence.add(label)
            
            cases.append(analysis_prompt.format(filtered))
            # cases.append(analysis_prompt.format(', '.join(sequence), tmp_case))
            
        model_name = "mistral"
        run_gpt = 1
        if run_gpt:
            outputs = {}
            for run in [1]:
                for model in ["microsoft/Phi-4-mini-instruct"]:
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
                            
                            prompt = prompt[:prompt.find("Start with established laws referred")].strip()
                            prompt = prompt + "\n\n**STA**\n" + completion + f"\n\nBased upon the **STA** provided above, ouput YES if the case was in favour of {petitioners[prompt_idx]}, else NO"
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

                            # break

                    del client
                    gc.collect()
                    empty_cache()
            
            with open(f"ljp_{model_name}_without_defs_without_rr_without_chain_{tempr}_{dec_tempr}_{'uk' if 'UK' in root else 'in'}.json", 'w') as jsonf:
                json.dump(outputs, jsonf)

            # import sys
            # sys.exit(0)
                
        with open(f"ljp_{model_name}_without_defs_without_rr_without_chain_{tempr}_{dec_tempr}_{'uk' if 'UK' in root else 'in'}.json", 'r') as jsonf:
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
