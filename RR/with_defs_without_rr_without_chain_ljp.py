#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 13:42:17 2024

@author: mann
"""

import json
from copy import deepcopy
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report
import gc
from openai import OpenAI

# tempr = 0.
# dec_tempr = 0.
# suffix = "_preamble_plaintiff"

for tempr in [0.]:
    for dec_tempr in [0.]:
        if tempr < 0.01 :
            do_sample_tempr = False
        else:
            do_sample_tempr = True

        if dec_tempr < 0.01:
            do_sample_dec = False
        else:
            do_sample_dec = True

        # for suffix in ["_plaintiff"]:
        for suffix in ["_preamble_plaintiff"]:

            dset = "test"
            with open(f"{dset}.json", 'r') as dev:
                dev = json.load(dev)
                
            with open("analysis_with_defs_without_rr.txt", 'r') as analysis_prompt:
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
            cases = []

            for case_idx, case_ in enumerate(dev):
                sequence = set()
                filtered = case_['data']['text']
                for result in case_["annotations"][0]["result"]:
                    if result["value"]["labels"][0] in EXCLUDE:
                        filtered = filtered.replace(result["value"]["text"], "")
                    else:
                        if result["value"]["labels"][0] not in ["NONE", "PRE_NOT_RELIED"]:
                            sequence.add(result["value"]["labels"][0])
                index_map = {value: index for index, value in enumerate(ORDER)}
                sequence = sorted(sequence, key=lambda x: index_map[x])
                dev[case_idx]['fitered'] = {'text': filtered}
                cases.append(analysis_prompt.format(filtered))
                # cases.append(analysis_prompt.format(', '.join(sequence), filtered))
                
            run_gpt = 1
            if run_gpt:
                outputs = {}
                for run in [1]:
                    for model in ["gpt-4.1-2025-04-14"]:
                    
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
                        elif "o3" in model.lower():
                            model_name = "GPTo3"
                        elif "gpt-4.1" in model.lower():
                            model_name = "GPT4.1"
                        client = OpenAI(
                            api_key=""
                            )

                        for prompt_idx, prompt in tqdm(enumerate(cases), total=len(cases)):
                            prompt = prompt.replace("You are a judge of the Indian Supreme Court. ", "")
                            if prompt not in outputs:
                                # import sys
                                # sys.exit(0)
                                completion = client.chat.completions.create(
                                    model=model,
                                    messages=[
                                        {"role": "system", "content": "You are an Indian Supreme Court Judge"},
                                        {"role": "user", "content": prompt}
                                    ],
                                    # do_sample=do_sample_tempr,
                                    temperature=tempr
                                )
                                
                                completion = completion.choices[0].message.content
                                print(completion)

                                prompt = prompt.replace("What are views of the court. View includes courts' discussion on the evidence, facts presented, prior cases, and statutes. Discussions on how the law is applicable or not applicable to the current case. Observations (non-binding) from the court.", "")
                                prompt = prompt.strip()
                                
                                prompt += "\n\n**ANALYSIS**\n" + completion + f"\n\nGiven the ANALYSIS. Output YES if the case will be in the favour of {petitioners[prompt_idx]}, else NO"
                                # prompt = prompt[:prompt.find("What are views of the court. View includes c")].strip()
                                # prompt = prompt + "\n\n**ANALYSIS**\n" + completion + f"\n\nBased upon the **ANALYSIS** provided above, ouput YES if the case was in favour of {petitioners[prompt_idx]}, else NO"
                                completion = client.chat.completions.create(
                                    model=model,
                                    messages=[
                                        {"role": "system", "content": "You are an Indian Supreme Court Judge"},
                                        {"role": "user", "content": prompt},
                                    ],
                                    # do_sample=do_sample_dec,
                                    temperature=dec_tempr
                                )

                                completion = completion.choices[0].message.content
                                print(completion)
                            
                                outputs[prompt] = completion
                                print(prompt)
                        del client
                        gc.collect()

                
                with open(f"ljp_{model_name}_with_defs_without_rr_without_chain{suffix}_{tempr}_{dec_tempr}.json", 'w') as jsonf:
                    json.dump(outputs, jsonf)
                    
            with open(f"ljp_{model_name}_with_defs_without_rr_without_chain{suffix}_{tempr}_{dec_tempr}.json", 'r') as jsonf:
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
