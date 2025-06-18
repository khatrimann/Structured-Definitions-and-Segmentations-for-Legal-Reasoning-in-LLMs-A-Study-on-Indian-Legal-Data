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

        for suffix in ["_preamble_plaintiff"]:

            # if tempr == 0. and dec_tempr == 0. and suffix == "_preamble_plaintiff":
            #     continue
            
            dset = "test"
            with open(f"{dset}.json", 'r') as dev:
                dev = json.load(dev)
                
            with open("analysis_with_defs_with_rr.txt", 'r') as analysis_prompt:
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
                                tmp.append(prompt)
                                completion = client.chat.completions.create(
                                    model=model,
                                    messages=[
                                        {"role": "system", "content": "You are an Indian Supreme Court Judge"},
                                        {"role": "user", "content": prompt}
                                    ],
                                )
                                
                                completion = completion.choices[0].message.content
                                print(completion)
                                
                                prompt = prompt.replace("What are views of the court. View includes courts' discussion on the evidence, facts presented, prior cases, and statutes. Discussions on how the law is applicable or not applicable to the current case. Observations (non-binding) from the court.", "")
                                prompt = prompt.replace("ANALYSIS", "RATIO", 1)
                                prompt = prompt.strip()
                                
                                prompt += "\n\n**ANALYSIS**\n" + completion + "\n\nTell your RATIO according to the definition of the given ongoing case. The output should be within 200 words."
                                # prompt = prompt[:prompt.find("What are views of the court. View includes c")].strip()
                                # prompt = prompt + "\n\n**ANALYSIS**\n" + completion + f"\n\nBased upon the **ANALYSIS** provided above, ouput YES if the case was in favour of {petitioners[prompt_idx]}, else NO"
                                tmp.append(prompt)
                                completion = client.chat.completions.create(
                                    model=model,
                                    messages=[
                                        {"role": "system", "content": "You are an Indian Supreme Court Judge"},
                                        {"role": "user", "content": prompt},
                                    ],
                                )

                                completion = completion.choices[0].message.content
                                print(completion)
                                
                                prompt = prompt.replace("Tell your RATIO according to the definition of the given ongoing case. The output should be within 200 words.", "")
                                prompt = prompt.replace("RATIO", "RPC", 1)
                                prompt = prompt.strip()
                                
                                prompt += "\n\n**RATIO**\n" + completion + "\n\nTell your RPC (the conclusion) according to the definition of the given ongoing case within 150 words."
                                tmp.append(prompt)
                                completion = client.chat.completions.create(
                                    model=model,
                                    messages=[
                                        {"role": "system", "content": "You are an Indian Supreme Court Judge"},
                                        {"role": "user", "content": prompt},
                                    ],
                                )

                                completion = completion.choices[0].message.content
                                print(completion)
                                
                                prompt = prompt.replace("Tell your RPC (the conclusion) according to the definition of the given ongoing case within 150 words.", "")
                                prompt = prompt.replace("RPC", "Decision", 1)
                                prompt = prompt.strip()
                                
                                prompt += "\n\n**RPC**\n" + completion + f"\n\nOutput YES if the decision was in the favor of {petitioners[prompt_idx]}, else NO"
                                tmp.append(prompt)
                                completion = client.chat.completions.create(
                                    model=model,
                                    messages=[
                                        {"role": "system", "content": "You are an Indian Supreme Court Judge"},
                                        {"role": "user", "content": prompt},
                                    ],
                                )

                                completion = completion.choices[0].message.content
                                print(completion)
                                
                                outputs[prompt] = completion
                                print(prompt)
                                print("*"*10)
                                print(completion)
                                # import sys
                                # sys.exit(0)
                                
                        del client
                        gc.collect()
                
                with open(f"ljp_{model_name}_with_defs_with_rr_with_chain{suffix}_{tempr}_{dec_tempr}.json", 'w') as jsonf:
                    json.dump(outputs, jsonf)

                # import sys
                # sys.exit(0)
                    
            print(f"ljp_{model_name}_with_defs_with_rr_with_chain{suffix}_{tempr}_{dec_tempr}.json")
            with open(f"ljp_{model_name}_with_defs_with_rr_with_chain{suffix}_{tempr}_{dec_tempr}.json", 'r') as jsonf:
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

