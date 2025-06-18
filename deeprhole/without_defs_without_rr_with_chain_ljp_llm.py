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
import os

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
            
        with open("analysis_without_defs_without_rr.txt", 'r') as analysis_prompt:
            analysis_prompt = analysis_prompt.read()
            
        EXCLUDE = ["STA", "RPC", "Ratio"]
        ORDER = ["FAC", "RLC", "PRE", "ARG", "ANALYSIS", "Ratio", "RPC"]
        
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
            index_map = {value: index for index, value in enumerate(ORDER)}
            sequence = sorted(sequence, key=lambda x: index_map[x])
            for seq in sequence:
                tmp_case += f"**{seq}**\n{seg[case_idx]['RR'][seq]}\n\n"
            tmp_case = tmp_case.strip()
            
            cases.append(analysis_prompt.format(filtered))
            # cases.append(analysis_prompt.format(', '.join(sequence), tmp_case))
            
        # import sys
        # sys.exit(0)
            
        run_gpt = 1
        tmp = []
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
                            tmp.append(prompt)
                            completion = client.create(
                                # model=model,
                                messages=[
                                    {"role": "system", "content": "You are an Indian Supreme Court Judge"},
                                    {"role": "user", "content": prompt}
                                ],
                                do_sample=do_sample_tempr,
                                temperature=tempr
                            )
                            
                            
                            prompt = prompt.replace("Start with established laws referred to by the present court, which can come from a mixture of sources – Acts , Sections, Articles, Rules, Order, Notices, Notifications, Quotations directly from the bare act, and so on.", "")
                            prompt = prompt.replace("STA", "ratio", 1)
                            prompt = prompt.strip()
                            
                            prompt += "\n\n**STA**\n" + completion + "\n\nA court's ratio includes the main reason given for the application of any legal principle to the legal issue. It is the result of the analysis by the court. It typically appears right before the final decision. It is not the same as \"ratio Decidendi\" taught in the legal academic curriculum. Tell your ratio according to the definition of the given ongoing case. The output should be within 200 words."
                            # prompt = prompt[:prompt.find("What are views of the court. View includes c")].strip()
                            # prompt = prompt + "\n\n**ANALYSIS**\n" + completion + f"\n\nBased upon the **ANALYSIS** provided above, ouput YES if the case was in favour of {petitioners[prompt_idx]}, else NO"
                            tmp.append(prompt)
                            completion = client.create(
                                # model=model,
                                messages=[
                                    {"role": "system", "content": "You are an Indian Supreme Court Judge"},
                                    {"role": "user", "content": prompt},
                                ],
                                do_sample=do_sample_tempr,
                                temperature=tempr
                            )
                            
                            prompt = prompt.replace("A court's ratio includes the main reason given for the application of any legal principle to the legal issue. It is the result of the analysis by the court. It typically appears right before the final decision. It is not the same as \"ratio Decidendi\" taught in the legal academic curriculum. Tell your ratio according to the definition of the given ongoing case. The output should be within 200 words.", "")
                            prompt = prompt.replace("ratio", "RPC", 1)
                            prompt = prompt.strip()
                            
                            prompt += "\n\n**ratio**\n" + completion + "\n\nRPC is Final decision + conclusion + order of the Court following from the natural/logical outcome of the rationale from ratio. Tell your RPC (the conclusion) according to the definition of the given ongoing case within 150 words."
                            tmp.append(prompt)
                            completion = client.create(
                                # model=model,
                                messages=[
                                    {"role": "system", "content": "You are an Indian Supreme Court Judge"},
                                    {"role": "user", "content": prompt},
                                ],
                                do_sample=do_sample_tempr,
                                temperature=tempr
                            )
                            
                            prompt = prompt.replace("RPC is Final decision + conclusion + order of the Court following from the natural/logical outcome of the rationale from ratio. Tell your RPC (the conclusion) according to the definition of the given ongoing case within 150 words.", "")
                            prompt = prompt.replace("RPC", "Decision", 1)
                            prompt = prompt.strip()
                            
                            prompt += "\n\n**RPC**\n" + completion + f"\n\nOutput YES if the decision was in the favor of {petitioners[prompt_idx]}, else NO"
                            tmp.append(prompt)
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

                            # break

                    del client
                    gc.collect()
                    empty_cache()
            
            with open(f"ljp_{model_name}_without_defs_without_rr_with_chain_{tempr}_{dec_tempr}_{'uk' if 'UK' in root else 'in'}.json", 'w') as jsonf:
                json.dump(outputs, jsonf)

            # import sys
            # sys.exit(0)
                
        print(f"ljp_{model_name}_without_defs_without_rr_with_chain_{tempr}_{dec_tempr}_{'uk' if 'UK' in root else 'in'}.json")
        with open(f"ljp_{model_name}_without_defs_without_rr_with_chain_{tempr}_{dec_tempr}_{'uk' if 'UK' in root else 'in'}.json", 'r') as jsonf:
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
