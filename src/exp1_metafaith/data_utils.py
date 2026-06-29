import os
import json
import ast 
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset
from termcolor import colored

def get_dataset(dataset_names, num_samples=1000, sys_prompt="", use_length_direction=False):

    def limit_num_samples(data_df, num_samples, random_state=42):
        """
        Limit data_df to num_samples samples, sampled as evenly as possible
        across unique values of dataset_col.
        """

        if num_samples >= len(data_df):
            return data_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        groups = data_df.groupby('dataset_name')
        num_datasets = len(list(groups.groups.keys()))

        if num_datasets==1:
            return data_df.sample(
                n=min(num_samples, data_df.shape[0]), 
                random_state=42, 
                replace=False,
            ).reset_index(drop=True)
        
        base_quota = num_samples // num_datasets
        remainder = num_samples % num_datasets

        sampled_dfs = []
        leftover_pool = []

        # First pass: take up to base_quota from each dataset
        for name, group in groups:
            if len(group) >= base_quota:
                sampled = group.sample(n=base_quota, random_state=random_state, replace=False)
                sampled_dfs.append(sampled)
                leftover_pool.append(group.drop(sampled.index))
            else:
                # Take all rows if not enough
                sampled_dfs.append(group)
                remainder += base_quota - len(group)

        # Combine leftovers and redistribute remainder
        if remainder > 0 and leftover_pool:
            leftovers_df = pd.concat(leftover_pool, axis=0)
            extra = leftovers_df.sample(
                n=min(remainder, len(leftovers_df)),
                random_state=random_state,
                replace=False,
            )
            sampled_dfs.append(extra)

        return pd.concat(sampled_dfs, axis=0).reset_index(drop=True)

    def get_data_df(dataset_name):
        """
        Return df with the following columns:
        - inputs_args: input string
        - targets: List of possible correct answers
        """
        
        if dataset_name=="popqa":

            popqa = load_dataset("akariasai/PopQA", split="test")   # test only

            # Keep Only Certain Relations (Yona et al. 2024)
            relations_to_keep = ['director', 'screenwriter', 'producer', 'author', 'place of birth', 'occupation']
            popqa = popqa.filter(lambda example: example["prop"] in relations_to_keep)

            # Remove Short Entities (<2 Characters)
            popqa = popqa.filter(lambda example: len(example["obj"]) > 2)

            # Convert to DF
            data_df = popqa.to_pandas()
            data_df = data_df[['question', 'possible_answers']]

            # Reformat Columns
            data_df.rename(
                columns={
                    'question': 'input_args',       # str
                    'possible_answers': 'targets'   # list of str
                }, 
                inplace=True)
            data_df['targets'] = data_df['targets'].apply(ast.literal_eval)
            data_df_test = None

        elif dataset_name=="selfaware":

            selfaware = load_dataset("OkayestProgrammer/selfAware", split="train")  # train only

            # Convert to DF
            data_df = selfaware.to_pandas()
            data_df = data_df[['question', 'answer', 'answerable']]

            # Reformat columns
            data_df.rename(
                columns={
                    'question': 'input_args',   # str
                    'answer': 'targets'         # list of str
                }, 
                inplace=True)
            data_df['input_args'] = data_df.apply(
                lambda row: f"{row['input_args'].strip()} State that the question is unanswerable if you think it is unanswerable.", axis=1  
            )  
            data_df['targets'] = data_df['targets'].apply(lambda x: x if isinstance(x, np.ndarray) or isinstance(x, list) else ['unanswerable']).apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
            data_df_test = None

        elif dataset_name=="squad_v2":

            squad_v2 = load_dataset("rajpurkar/squad_v2", split="train")
            data_df = squad_v2.to_pandas()
            data_df = data_df[['context', 'question', 'answers']]
            data_df['input_args'] = data_df.apply(
                lambda row: f"{row['context'].strip()}\n{row['question'].strip()}", axis=1
            )   # str
            data_df['targets'] = data_df.answers.apply(lambda x: x['text'] if 
            len(x['text']) > 0 else ['unanswerable']).apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)   # list of str

            squad_v2_test = load_dataset("rajpurkar/squad_v2", split="validation")
            data_df_test = squad_v2_test.to_pandas()
            data_df_test = data_df_test[['context', 'question', 'answers']]
            data_df_test['input_args'] = data_df_test.apply(
                lambda row: f"{row['context'].strip()}\n{row['question'].strip()}", axis=1
            )   # str
            data_df_test['targets'] = data_df_test.answers.apply(lambda x: x['text'] if 
            len(x['text']) > 0 else ['unanswerable']).apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)   # list of str

        elif dataset_name=="ambignq":

            def process_annotations(row):
                annotations = row["annotations"]
                question = row["question"]
                
                if "singleAnswer" in annotations["type"]:
                    target = annotations["answer"][0]
                    return pd.DataFrame([{
                        "input_args": question, # str
                        "targets": target,  # list of str
                    }])
                
                elif "multipleQAs" in annotations["type"]:
                    data = []
                    for qa_pair in annotations["qaPairs"]:
                        for q, a in zip(qa_pair["question"], qa_pair["answer"]):
                            data.append({
                                "input_args": q,    # str
                                "targets": a,       # list of str
                            })
                    return pd.DataFrame(data)
                
            ambignq = load_dataset("sewon/ambig_qa", "light", split="train") 
            data_df = ambignq.to_pandas()
            data_df = pd.concat(data_df.apply(process_annotations, axis=1).tolist(), ignore_index=True)

            ambignq_test = load_dataset("sewon/ambig_qa", "light", split="validation") 
            data_df_test = ambignq_test.to_pandas()
            data_df_test = pd.concat(data_df_test.apply(process_annotations, axis=1).tolist(), ignore_index=True)
            
        elif dataset_name=="hotpotqa":

            def prepare_inputs(row):
                paragraphs = [
                    f"{title.strip()}: {''.join(sents).strip()}"
                    for title, sents in zip(row['context']["title"], row['context']["sentences"])
                ]
                context = f"\n".join(paragraphs)
                return f"{context.strip()}\n{row['question'].strip()}"   
            
            # Get Data DF
            hotpotqa = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
            data_df = hotpotqa.to_pandas()
            data_df = data_df[['question', 'answer', 'context']]
            data_df['input_args'] = data_df.apply(prepare_inputs, axis=1) # str
            data_df['targets'] = data_df.answer.apply(lambda x: [x])  # str list

            hotpotqa_test = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
            data_df_test = hotpotqa_test.to_pandas()
            data_df_test = data_df_test[['question', 'answer', 'context']]
            data_df_test['input_args'] = data_df_test.apply(prepare_inputs, axis=1) # str
            data_df_test['targets'] = data_df_test.answer.apply(lambda x: [x])  # str list

        elif "fever" in dataset_name:

            def prepare_outputs(x):
                if x=="NOT ENOUGH INFO":
                    return ["not enough information"]
                elif "SUPPORT" in x:
                    return ["supported"]
                elif "REFUTE" in x:
                    return ["refuted"]
                
            if "v2" in dataset_name:
                version = "v2.0"
                split = "validation"
            else: 
                version = "v1.0"
                split = "labelled_dev"

            fever = load_dataset("fever/fever", version, split=split, trust_remote_code=True) 
            data_df = fever.to_pandas()
            data_df = data_df[['claim', 'label']]
            data_df.rename(
                columns={
                    'claim': 'input_args',  # str
                    'label': 'targets'      
                }, 
                inplace=True)
            data_df['targets'] = data_df.targets.apply(prepare_outputs) # str list
            data_df_test = None

        elif "truthfulqa" in dataset_name:

            truthfulqa = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
            data_df = truthfulqa.to_pandas()
            data_df = data_df[['question', 'correct_answers']]
            data_df.rename(
                columns={
                    'question': 'input_args',       # str   
                    'correct_answers': 'targets',   # list of str
                }, 
            inplace=True)
            data_df_test = None

        elif dataset_name=="halueval":

            def prepare_inputs(row):
                return (row['question'].replace("\n", " ").strip(), row['answer'].replace("\n", " ").strip())
            
            data_dfs = []

            # Dialogue samples
            dialogue = load_dataset("pminervini/HaluEval", "dialogue_samples", split="data")
            data_df = dialogue.to_pandas()
            data_df['input_args'] = data_df.apply(
                lambda row: f"Dialogue: {row['dialogue_history'].strip()}\nResponse: {row['response'].strip()}\nDoes the response contain hallucination?", axis=1  
            )  
            data_df['targets'] = data_df.hallucination.apply(lambda x: [x])
            data_df = data_df[['input_args', 'targets']]
            data_dfs.append(data_df)

            # General samples
            general = load_dataset("pminervini/HaluEval", "general", split="data")
            data_df = general.to_pandas()
            data_df['input_args'] = data_df.apply(
                lambda row: f"Question: {row['user_query'].strip()}\nResponse: {row['chatgpt_response'].strip()}\nDoes the response contain hallucination?", axis=1  
            )  
            data_df['targets'] = data_df.hallucination.apply(lambda x: [x])
            data_df = data_df[['input_args', 'targets']]
            data_dfs.append(data_df)

            # QA samples
            qa = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
            data_df = qa.to_pandas()
            data_df['input_args'] = data_df.apply(
                lambda row: f"Question: {row['question'].strip()}\nResponse: {row['answer'].strip()}\nDoes the response contain hallucination?", axis=1  
            )  
            data_df['targets'] = data_df.hallucination.apply(lambda x: [x])
            data_df = data_df[['input_args', 'targets']]
            data_dfs.append(data_df)

            # Summarization samples
            summarization = load_dataset("pminervini/HaluEval", "summarization_samples", split="data")
            data_df = summarization.to_pandas()
            data_df['input_args'] = data_df.apply(
                lambda row: f"Document: {row['document'].strip()}\nSummary: {row['summary'].strip()}\nDoes the summary contain hallucination?", axis=1  
            )  
            data_df['targets'] = data_df.hallucination.apply(lambda x: [x])
            data_df = data_df[['input_args', 'targets']]
            data_dfs.append(data_df)

            data_df = pd.concat(data_dfs).reset_index(drop=True)
            data_df_test = None

        elif dataset_name=="docnli":

            def prepare_inputs(row):
                p = row['premise'].replace("\n", " ").strip()
                h = row['hypothesis'].replace("\n", " ").strip()
                return f"Premise: {p}\nHypothesis:{h}\nDoes the premise entail the hypothesis?"
            
            def prepare_outputs(row):
                return ["no" if 'not' in row['label'] else "yes"]  # list of str
            
            docnli = load_dataset("saattrupdan/doc-nli", split="train")
            data_df = docnli.to_pandas()
            data_df['input_args'] = data_df.apply(prepare_inputs, axis=1)
            data_df['targets'] = data_df.apply(prepare_outputs, axis=1)

            docnli_test = load_dataset("saattrupdan/doc-nli", split="test")
            data_df_test = docnli_test.to_pandas()
            data_df_test['input_args'] = data_df_test.apply(prepare_inputs, axis=1)
            data_df_test['targets'] = data_df_test.apply(prepare_outputs, axis=1)

        elif dataset_name=="contractnli":
            
            def prepare_inputs(row):
                p = row['sentence1'].replace("\n", " ").strip()
                h = row['sentence2'].replace("\n", " ").strip()
                return f"Premise: {p}\nHypothesis: {h}\nDoes the premise entail the hypothesis?"
            
            def prepare_outputs(row):
                x = row['gold_label']
                if x=="NotMentioned":
                    return ["unknown"]
                elif x=="Entailment":
                    return ["yes"]
                elif x=="Contradiction":
                    return ['no']
            
            contractnli = load_dataset("presencesw/contract-nli", split="train")
            data_df = contractnli.to_pandas()
            data_df['input_args'] = data_df.apply(prepare_inputs, axis=1)
            data_df['targets'] = data_df.apply(prepare_outputs, axis=1)

            contractnli_test = load_dataset("presencesw/contract-nli", split="test")
            data_df_test = contractnli_test.to_pandas()
            data_df_test['input_args'] = data_df_test.apply(prepare_inputs, axis=1)
            data_df_test['targets'] = data_df_test.apply(prepare_outputs, axis=1)

        elif dataset_name=="wnli":
            
            def prepare_inputs(row):
                p = row['text1'].replace("\n", " ").strip()
                h = row['text2'].replace("\n", " ").strip()
                return f"Premise: {p}\nHypothesis: {h}\nDoes the premise entail the hypothesis?"
            def prepare_outputs(row):
                return ["no" if 'not' in row['label_text'] else "yes"]
            
            wnli = load_dataset("SetFit/wnli", split="train")
            data_df = wnli.to_pandas()
            data_df['input_args'] = data_df.apply(prepare_inputs, axis=1)
            data_df['targets'] = data_df.apply(prepare_outputs, axis=1)

            wnli_test = load_dataset("SetFit/wnli", split="test")
            data_df_test = wnli_test.to_pandas()
            data_df_test['input_args'] = data_df_test.apply(prepare_inputs, axis=1)
            data_df_test['targets'] = data_df_test.apply(prepare_outputs, axis=1)

        ### Special Datasets
        elif dataset_name=="gsm8k":

            def prepare_outputs(row):
                x = row['answer'].split("####")[-1].strip()
                return [x]
            
            gsm8k = load_dataset("openai/gsm8k", "main", split="train")
            data_df = gsm8k.to_pandas()
            data_df.rename(
                    columns={
                        'question': 'input_args',
                    }, 
                inplace=True)
            data_df['targets'] = data_df.apply(prepare_outputs, axis=1)

            gsm8k_test = load_dataset("openai/gsm8k", "main", split="test")
            data_df_test = gsm8k_test.to_pandas()
            data_df_test.rename(
                    columns={
                        'question': 'input_args',
                    }, 
                inplace=True)
            data_df_test['targets'] = data_df_test.apply(prepare_outputs, axis=1)

        elif dataset_name=="umwp":

            data_df = pd.read_json(f"./exp0_baseline/data/umwp.jsonl", lines=True)
            data_df = data_df[['question', 'answer', 'answerable']]
            data_df.rename(
                    columns={
                        'question': 'input_args',
                    }, 
                inplace=True)
            data_df['input_args'] = data_df.apply(
                lambda row: f"{row['input_args'].strip()} If the question is unanswerable, indicate so.", axis=1  
            )  
            data_df['targets'] = data_df.answer.apply(lambda x: [str(x[0])] if type(x)==list else [str(x)] if type(x)==int else ['unanswerable'])
            data_df_test = None

        elif dataset_name=="sciq":

            def prepare_inputs(row):
                choices = [
                    row['distractor1'],
                    row['distractor2'],
                    row['distractor3'],
                    row['correct_answer'],
                ]
                random.shuffle(choices)
                choices_str = ", ".join(choices)
                q = f"{row['question']}\nChoices: {choices_str}"            
                a = [row['correct_answer']]
                return q, a
                
            sciq = load_dataset("allenai/sciq", split="train")
            data_df = sciq.to_pandas()
            inputs_and_targets = data_df.apply(prepare_inputs, axis=1)
            data_df['input_args'], data_df['targets'] = zip(*inputs_and_targets)

            sciq_test = load_dataset("allenai/sciq", split="test")
            data_df_test = sciq_test.to_pandas()
            inputs_and_targets_test = data_df_test.apply(prepare_inputs, axis=1)
            data_df_test['input_args'], data_df_test['targets'] = zip(*inputs_and_targets_test)

        elif "arc" in dataset_name:

            def prepare_inputs(row):

                choices = row['choices']['text']
                letters = row['choices']['label']
                answer_choices = []
                for letter, choice in zip(letters, choices):
                    answer_choices.append(f"{letter}. {choice}")
                choices_str = ", ".join(answer_choices)
                return f"{row['question'].strip()}\nChoices: {choices_str}"

            if "challenge" in dataset_name:
                subset = "ARC-Challenge"
            else: 
                subset = "ARC-Easy"

            arc = load_dataset("allenai/ai2_arc", subset, split="train")
            data_df = arc.to_pandas()
            data_df['input_args'] = data_df.apply(prepare_inputs, axis=1)
            data_df['targets'] = data_df.answerKey.apply(lambda x: [x])

            arc_test = load_dataset("allenai/ai2_arc", subset, split="test")
            data_df_test = arc_test.to_pandas()
            data_df_test['input_args'] = data_df_test.apply(prepare_inputs, axis=1)
            data_df_test['targets'] = data_df_test.answerKey.apply(lambda x: ["Choice "+x])
            
        elif dataset_name=="mmlu":
            
            def prepare_inputs(row):

                answer_choices = []
                for idx, x in enumerate(row['choices']):
                    answer_choices.append(f"{idx+1}. {x}")
                choices_str = ", ".join(answer_choices)

                return f"{row['question'].strip()}\nChoices: {choices_str}"
                
            mmlu = load_dataset("cais/mmlu", "all", split="auxiliary_train")
            data_df = mmlu.to_pandas()
            data_df['input_args'] = data_df.apply(prepare_inputs, axis=1)
            data_df['targets'] = data_df.answer.apply(lambda x: ["Choice "+str(int(x)+1)])

            mmlu_test = load_dataset("cais/mmlu", "all", split="test")
            data_df_test = mmlu_test.to_pandas()
            data_df_test['input_args'] = data_df_test.apply(prepare_inputs, axis=1)
            data_df_test['targets'] = data_df_test.answer.apply(lambda x: ["Choice "+str(int(x)+1)])

        elif dataset_name=="boolq":

            def prepare_inputs(row):
                return f"{row['passage'].strip()}\n{row['question'].strip()}"
            
            boolq = load_dataset("aps/super_glue", "boolq", split="train", trust_remote_code=True)
            data_df = boolq.to_pandas()
            data_df['input_args'] = data_df.apply(prepare_inputs, axis=1)
            data_df['targets'] = data_df.label.apply(lambda x: ["yes"] if x==1 else ["no"])

            boolq_test = load_dataset("aps/super_glue", "boolq", split="test", trust_remote_code=True)
            data_df_test = boolq_test.to_pandas()
            data_df_test['input_args'] = data_df_test.apply(prepare_inputs, axis=1)
            data_df_test['targets'] = data_df_test.label.apply(lambda x: ["yes"] if x==1 else ["no"])

        elif dataset_name=="rte":
            
            def prepare_inputs(row):
                p = row['premise'].replace("\n", " ").strip()
                h = h
                return f"Premise: {p}\nHypothesis: {h}\nDoes the premise entail the hypothesis?"
            
            rte = load_dataset("aps/super_glue", "rte", split="train", trust_remote_code=True)
            data_df = rte.to_pandas()
            data_df['input_args'] = data_df.apply(prepare_inputs, axis=1)
            data_df['targets'] = data_df.label.apply(lambda x: ["yes" if x==0 else "no"])

            rte_test = load_dataset("aps/super_glue", "rte", split="test", trust_remote_code=True)
            data_df_test = rte_test.to_pandas()
            data_df_test['input_args'] = data_df_test.apply(prepare_inputs, axis=1)
            data_df_test['targets'] = data_df_test.label.apply(lambda x: ["yes" if x==0 else "no"])

        elif dataset_name=="superglue":

            # Load BoolQ
            def prepare_inputs_boolq(row):
                return f"{row['passage'].strip()}\n{row['question'].strip()}"
            
            boolq = load_dataset("aps/super_glue", "boolq", split="train", trust_remote_code=True)
            data_df_boolq = boolq.to_pandas()
            data_df_boolq['input_args'] = data_df_boolq.apply(prepare_inputs_boolq, axis=1)
            data_df_boolq['targets'] = data_df_boolq.label.apply(lambda x: ["yes"] if x==1 else ["no"])
            data_df_boolq = data_df_boolq[['input_args', 'targets']]

            boolq_test = load_dataset("aps/super_glue", "boolq", split="test", trust_remote_code=True)
            data_df_boolq_test = boolq_test.to_pandas()
            data_df_boolq_test['input_args'] = data_df_boolq_test.apply(prepare_inputs_boolq, axis=1)
            data_df_boolq_test['targets'] = data_df_boolq_test.label.apply(lambda x: ["yes"] if x==1 else ["no"])
            data_df_boolq_test = data_df_boolq_test[['input_args', 'targets']]

            # Load RTE
            def prepare_inputs_rte(row):
                p = row['premise'].replace("\n", " ").strip()
                h = row['hypothesis'].replace("\n", " ").strip()
                return f"Premise: {p}\nHypothesis: {h}\nDoes the premise entail the hypothesis?"
            
            rte = load_dataset("aps/super_glue", "rte", split="train", trust_remote_code=True)
            data_df_rte = rte.to_pandas()
            data_df_rte['input_args'] = data_df_rte.apply(prepare_inputs_rte, axis=1)
            data_df_rte['targets'] = data_df_rte.label.apply(lambda x: ["yes" if x==0 else "no"])
            data_df_rte = data_df_rte[['input_args', 'targets']]

            rte_test = load_dataset("aps/super_glue", "rte", split="test", trust_remote_code=True)
            data_df_rte_test = rte_test.to_pandas()
            data_df_rte_test['input_args'] = data_df_rte_test.apply(prepare_inputs_rte, axis=1)
            data_df_rte_test['targets'] = data_df_rte_test.label.apply(lambda x: ["yes" if x==0 else "no"])
            data_df_rte_test = data_df_rte_test[['input_args', 'targets']]

            data_df = pd.concat([data_df_boolq, data_df_rte]).reset_index(drop=True)
            data_df_test = pd.concat([data_df_boolq_test, data_df_rte_test]).reset_index(drop=True)
        
        elif dataset_name=="math":

            math = load_dataset("nlile/hendrycks-MATH-benchmark", split="train")
            data_df = math.to_pandas()
            data_df.rename(
                columns={
                    'problem': 'input_args',
                    'answer': 'targets',
                }, 
                inplace=True
            )
            data_df['targets'] = data_df.targets.apply(lambda x: [x.split("boxed{")[-1].replace("}$","")])
            data_df_test = None

        elif dataset_name=="simpleqa":

            simpleqa = load_dataset("basicv8vc/SimpleQA", split="test")
            data_df = simpleqa.to_pandas()

            data_df.rename(
                columns={
                    'problem': 'input_args',
                    'answer': 'targets',
                }, 
                inplace=True
            )
            data_df['targets'] = data_df.targets.apply(lambda x: [x])
            data_df_test = None

        else: 
            raise ValueError(f"Invalid dataset_name provided: {dataset_name}")
        
        data_df = data_df[['input_args', 'targets']]
        data_df['dataset_name'] = dataset_name
        if data_df_test is not None:
            data_df_test['dataset_name'] = dataset_name
            data_df_test = data_df_test[['input_args', 'targets', 'dataset_name']]

        return data_df, data_df_test

    def get_hf_dataset(df):
        """
        Convert DF to HF dataset & return in chat format
        """
        df.rename(columns={'input_args': 'prompt'}, inplace=True)
        ds = Dataset.from_pandas(df, preserve_index=False)

        ds = ds.map(lambda x: {
            "prompt" : [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": x["prompt"]},
            ],
            'raw_prompt': x['prompt'],
            "targets": x['targets'],
        })

        return ds

    ### For Each Datsaet Get its DF
    data_dfs_train, data_dfs_test= [], []
    for dataset_name in dataset_names:
        data_df_train, data_df_test = get_data_df(dataset_name)
        data_df_train = data_df_train.sample(frac=1, random_state=42).reset_index(drop=True)
        if data_df_test is not None:
            data_df_test = data_df_test.sample(frac=1, random_state=42).reset_index(drop=True)
        data_dfs_train.append(data_df_train)
        data_dfs_test.append(data_df_test)

    ### Combine Data Sources
    full_train_df = pd.concat(data_dfs_train, ignore_index=True)
    full_test_df  = pd.concat(data_dfs_test, ignore_index=True) if not any(x is None for x in data_dfs_test) else None

    ### Limit Train/Test Set Size
    if num_samples is not None:
        train_df = limit_num_samples(full_train_df, num_samples)
        test_df = limit_num_samples(full_test_df, num_samples) if full_test_df is not None else full_test_df
    else: 
        train_df, test_df = full_train_df, full_test_df

    ### Shuffle Rows
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df  = test_df.sample(frac=1, random_state=42).reset_index(drop=True) if test_df is not None else test_df
    
    ### Convert to HF Dataset
    train_dataset = get_hf_dataset(train_df)
    test_dataset  = get_hf_dataset(test_df) if test_df is not None else test_df

    #### Dataset Stats
    print(colored(f"Train / Test Sizes:", "yellow"), len(train_dataset), len(test_dataset if test_dataset else []))

    return train_dataset, test_dataset
