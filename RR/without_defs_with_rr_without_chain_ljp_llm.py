#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 01:48:20 2024

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

            # if tempr == 0. and dec_tempr == 0. and suffix == "_preamble_plaintiff":
            #     continue
            
            dset = "test"
            with open(f"{dset}.json", 'r') as dev:
                dev = json.load(dev)
                
            with open("analysis_without_defs_with_rr.txt", 'r') as analysis_prompt:
                analysis_prompt = analysis_prompt.read()
                
            petitioners = pd.read_csv("plaintiff.csv")
            petitioners = [pet if corr == "TRUE" else corr for pet, corr in zip(petitioners.plaintiff.tolist(), petitioners.is_correct.tolist())]
                
            EXCLUDE = ["ANALYSIS", "STA", "RPC", "RATIO"]
            ORDER = ["PREAMBLE", "FAC", "RLC", "ISSUE", "PRE_RELIED", "ARG_PETITIONER", "ARG_RESPONDENT", "ANALYSIS", "RATIO", "RPC"]
            
            if "preamble" not in suffix:
                EXCLUDE.append("PREAMBLE")
                del ORDER[0]
            
            seg = []
            lens = []
            for case_ in dev:
                tmp_seg = {"RR": {}, "text": case_["data"]["text"]}
                for result in case_["annotations"][0]["result"]:
                    if result["value"]["labels"][0] not in tmp_seg["RR"]:
                        tmp_seg["RR"][result["value"]["labels"][0]] = ""
                    tmp_seg["RR"][result["value"]["labels"][0]] += case_["data"]["text"][result["value"]["start"]:result["value"]["end"]] + "\n"
                lens.append(len(tmp_seg["RR"]))
                seg.append(tmp_seg)
                
            cases = []
            
            
            for case_idx, case_ in enumerate(dev):
                sequence = set()
                filtered = case_['data']['text']
                tmp_case = ""
                for result in case_["annotations"][0]["result"]:
                    if result["value"]["labels"][0] in EXCLUDE:
                        filtered = filtered.replace(result["value"]["text"], "")
                    else:
                        if result["value"]["labels"][0] not in ["NONE", "PRE_NOT_RELIED"]:
                            sequence.add(result["value"]["labels"][0])
                index_map = {value: index for index, value in enumerate(ORDER)}
                sequence = sorted(sequence, key=lambda x: index_map[x])
                for seq in sequence:
                    tmp_case += f"**{seq}**\n{seg[case_idx]['RR'][seq]}\n\n"
                tmp_case = tmp_case.strip()
                dev[case_idx]['fitered'] = {'text': filtered}
                # cases.append(analysis_prompt.format(', '.join(sequence), filtered))
                cases.append(analysis_prompt.format(', '.join(sequence), tmp_case))
                
            # import sys
            # sys.exit(0)
                
            run_gpt = 1
            tmp = []
            if run_gpt:
                outputs = {}
                for run in [1]:
                    # for model in ["gpt-3.5-turbo", "gpt-4-1106-preview", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0125", "ft:gpt-3.5-turbo-0125:personal:ljp-rr:9XxGCTvB:ckpt-step-484"]:
                    # for model in ["microsoft/Phi-3-mini-128k-instruct"]:
                    # for model in ["mistralai/Mistral-7B-Instruct-v0.3"]:
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
                                # print(prompt)
                                # import sys
                                # sys.exit(0)

                        del client
                        gc.collect()
                        empty_cache()
                                
                                
            
                
                with open(f"ljp_{model_name}_without_defs_with_rr_without_chain{suffix}_{tempr}_{dec_tempr}.json", 'w') as jsonf:
                    json.dump(outputs, jsonf)
                    
            print(f"ljp_{model_name}_without_defs_with_rr_without_chain{suffix}_{tempr}_{dec_tempr}.json")
            with open(f"ljp_{model_name}_without_defs_with_rr_without_chain{suffix}_{tempr}_{dec_tempr}.json", 'r') as jsonf:
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
            # import sys
            # sys.exit(0)