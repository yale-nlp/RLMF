import pandas as pd
import ast 
import numpy as np
import json 
import random

from datasets import load_dataset

from ._task import Task
from src.exp0_baseline.utilities.utils import score_qa, sanitize_text


class QA(Task):

    def __init__(self, args):

        super().__init__(args)      # initialize dataset name, data_df
        self.tupleify = True
        self.task_type = "qa"

        if self.dataset_name in [
            "umwp",
        ]:
            self.numerical_scoring = True
        else: 
            self.numerical_scoring = False
    
    def get_data_df(self):
        """
        inputs: tuple of input values ready for formatting in templates in input_prompts.py
        targets: List of possible correct answers
        """
        
        if self.dataset_name=="popqa":

            popqa = load_dataset("akariasai/PopQA", split="test")

            # Keep Only Certain Relations (Yona et al. 2024)
            relations_to_keep = ['director', 'screenwriter', 'producer', 'author', 'place of birth', 'occupation']
            popqa = popqa.filter(lambda example: example["prop"] in relations_to_keep)

            # Remove Short Entities (<2 Characters)
            popqa = popqa.filter(lambda example: len(example["obj"]) > 2)

            # Convert to DF
            data_df = popqa.to_pandas()
            data_df = data_df[['question', 'possible_answers']]

            # Reformat columns
            data_df.rename(
                columns={
                    'question': 'input_args', 
                    'possible_answers': 'targets'
                }, 
                inplace=True)
            data_df['targets'] = data_df['targets'].apply(ast.literal_eval)


        elif self.dataset_name=="selfaware":

            selfaware = load_dataset("JesusCrist/selfAware", split="train")

            # Convert to DF
            data_df = selfaware.to_pandas()
            data_df = data_df[['question', 'answer', 'answerable']]

            # Reformat columns
            data_df.rename(
                columns={
                    'question': 'input_args', 
                    'answer': 'targets'
                }, 
                inplace=True)
            data_df['targets'] = data_df['targets'].apply(lambda x: x if isinstance(x, np.ndarray) else ['unanswerable']).apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)


        elif self.dataset_name=="halueval":

            def prepare_inputs(row):
                return (row['question'].replace("\n", " ").strip(), row['answer'].replace("\n", " ").strip())
            
            data_dfs = []

            # Dialogue samples
            dialogue = load_dataset("pminervini/HaluEval", "dialogue_samples", split="data")
            data_df = dialogue.to_pandas()
            data_df.rename(
                    columns={
                        'dialogue_history': 'question',
                        'response': 'answer',
                        'hallucination': 'targets',
                    }, 
                inplace=True)
            data_df = data_df[['question', 'answer', 'targets']]
            data_df['task'] = ['halueval_dialogue']*data_df.shape[0]
            data_dfs.append(data_df)

            # General samples
            general = load_dataset("pminervini/HaluEval", "general", split="data")
            data_df = general.to_pandas()
            data_df.rename(
                    columns={
                        'user_query': 'question',
                        'chatgpt_response': 'answer',
                        'hallucination': 'targets',
                    }, 
                inplace=True)
            data_df = data_df[['question', 'answer', 'targets']]
            data_df['task'] = ['halueval_general']*data_df.shape[0]
            data_dfs.append(data_df)

            # QA samples
            qa = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
            data_df = qa.to_pandas()
            data_df.rename(
                    columns={
                        'hallucination': 'targets',
                    }, 
                inplace=True)
            data_df = data_df[['question', 'answer', 'targets']]
            data_df['task'] = ['halueval_qa']*data_df.shape[0]
            data_dfs.append(data_df)

            # Summarization samples
            summarization = load_dataset("pminervini/HaluEval", "summarization_samples", split="data")
            data_df = summarization.to_pandas()
            data_df.rename(
                    columns={
                        'document': 'question',
                        'summary': 'answer',
                        'hallucination': 'targets',
                    }, 
                inplace=True)
            data_df = data_df[['question', 'answer', 'targets']]
            data_df.question = data_df.question.apply(lambda x: f"Summarize: {x}")
            data_df['task'] = ['halueval_summarization']*data_df.shape[0]
            data_dfs.append(data_df)

            # Combine all subsets & format input
            data_df = pd.concat(data_dfs).reset_index(drop=True)
            data_df['input_args'] = data_df.apply(prepare_inputs, axis=1)
            data_df['targets'] = data_df.targets.apply(lambda x: [x])

            self.tupleify = False


        elif self.dataset_name=="umwp":

            data_df = pd.read_json("./exp0_baseline/data/umwp.jsonl", lines=True)
            data_df = data_df[['question', 'answer', 'answerable']]

            # Process inputs
            data_df.rename(
                    columns={
                        'question': 'input_args',
                    }, 
                inplace=True)
            data_df['targets'] = data_df.answer.apply(lambda x: x if x else ['unanswerable'])


        elif self.dataset_name=="sciq":

            def prepare_inputs(row):

                choices = [
                    row['distractor1'],
                    row['distractor2'],
                    row['distractor3'],
                    row['correct_answer'],
                ]
                random.shuffle(choices)

                answer_choices = ""
                for idx, x in enumerate(choices):
                    answer_choices += f"{idx+1}. {x}\n"

                correct_idx = choices.index(row['correct_answer']) +1

                return (row['question'].strip(), answer_choices.strip()), [correct_idx]
                
            sciq = load_dataset("allenai/sciq", split="test")
            data_df = sciq.to_pandas()

            # Process inputs
            inputs_and_targets= data_df.apply(prepare_inputs, axis=1)
            data_df['input_args'], data_df['targets'] = zip(*inputs_and_targets)

            self.tupleify = False


        elif "arc" in self.dataset_name:

            def prepare_inputs(row):

                choices = row['choices']['text']
                letters = row['choices']['label']
                answer_choices = ""
                for letter, choice in zip(letters, choices):
                    answer_choices += f"{letter}. {choice}\n"

                return (row['question'].strip(), answer_choices.strip())

            if "challenge" in self.dataset_name:
                subset = "ARC-Challenge"
            else: 
                subset = "ARC-Easy"

            arc = load_dataset("allenai/ai2_arc", subset, split="test")
            data_df = arc.to_pandas()

            # Process input
            data_df['input_args'] = data_df.apply(prepare_inputs, axis=1)
            data_df['targets'] = data_df.answerKey.apply(lambda x: [x])
            
            self.tupleify = False


        elif self.dataset_name=="mmlu":
            
            def prepare_inputs(row):

                answer_choices = ""
                for idx, x in enumerate(row['choices']):
                    answer_choices += f"{idx+1}. {x}\n"

                return (row['question'].strip(), answer_choices.strip())
                
            mmlu = load_dataset("cais/mmlu", "all", split="test")
            data_df = mmlu.to_pandas()

            # Process inputs
            data_df['input_args'] = data_df.apply(prepare_inputs, axis=1)
            data_df['targets'] = data_df.answer.apply(lambda x: [x+1])

            self.tupleify = False


        elif self.dataset_name=="superglue":

            # Load BoolQ
            def prepare_boolq_inputs(row):
                context = row['passage'].strip()
                question = row['question'].strip()
                return f"""{context}\n{question}"""
            
            boolq = load_dataset("aps/super_glue", "boolq", split="validation", trust_remote_code=True).to_pandas().sample(n=250, random_state=42)

            boolq['input_args'] = boolq.apply(prepare_boolq_inputs, axis=1)
            boolq['targets'] = boolq.label.apply(lambda x: ["yes"] if x==1 else ["no"])
            boolq['subset'] = 'boolq'
            boolq = boolq[['input_args', 'targets', 'subset']]

            # Load Copa
            def prepare_copa_inputs(row):
                context = row['premise'].strip()
                question = f"What’s the {row['question'].strip()} for this?"
                answer_choices = f"1. {row['choice1'].strip()}\n2. {row['choice2'].strip()}"
                return f"""{context}\n{question}\nAnswer Choices:\n{answer_choices}"""
            
            train = load_dataset("aps/super_glue", "copa", split="train", trust_remote_code=True).to_pandas()
            val = load_dataset("aps/super_glue", "copa", split="validation", trust_remote_code=True).to_pandas()
            copa = pd.concat([train, val]).reset_index(drop=True).sample(n=250, random_state=42)

            copa['input_args'] = copa.apply(prepare_copa_inputs, axis=1)
            copa['targets'] = copa.label.apply(lambda x: [x+1])
            copa['subset'] = 'copa'
            copa = copa[['input_args', 'targets', 'subset']]

            # Load WIC
            def prepare_wic_inputs(row):
                return f"""Is the word '{row['word'].strip()}' used with the same sense in both sentences below?\nSentence 1: {row['sentence1'].strip()}\nSentence 2: {row['sentence2'].strip()}"""
            
            wic = load_dataset("aps/super_glue", "wic", split="train", trust_remote_code=True).to_pandas().sample(n=250, random_state=42)

            wic['input_args'] = wic.apply(prepare_wic_inputs, axis=1)
            wic['targets'] = wic.label.apply(lambda x: ['yes' if x==1 else 'no'])
            wic['subset'] = 'wic'
            wic = wic[['input_args', 'targets', 'subset']]

            # Load WSC
            def prepare_wsc_inputs(row):
                return f"""Does the pronoun '{row['span2_text'].strip()}' refer to '{row['span1_text'].strip()}' in the given text?\nText: {row['text'].strip()}"""

            train = load_dataset("aps/super_glue", "wsc", split="train", trust_remote_code=True).to_pandas()
            val = load_dataset("aps/super_glue", "wsc", split="validation", trust_remote_code=True).to_pandas()
            wsc = pd.concat([train, val]).reset_index(drop=True).sample(n=250, random_state=42)

            wsc['input_args'] = wsc.apply(prepare_wsc_inputs, axis=1)
            wsc['targets'] = wsc.label.apply(lambda x: ['yes' if x==1 else 'no'])
            wsc['subset'] = 'wsc'
            wsc = wsc[['input_args', 'targets', 'subset']]

            data_df = pd.concat([boolq, copa, wic, wsc]).reset_index(drop=True)
        

        elif self.dataset_name=="math":

            math = load_dataset("nlile/hendrycks-MATH-benchmark", split="test")
            data_df = math.to_pandas()

            data_df.rename(
                columns={
                    'problem': 'input_args',
                    'answer': 'targets',
                }, 
                inplace=True
            )
            data_df['targets'] = data_df.targets.apply(lambda x: [x])


        elif self.dataset_name=="simpleqa":

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


        else: 
            raise ValueError(f"Invalid dataset_name provided: {self.dataset_name}")


        if self.tupleify:
            data_df = self.tupleify_inputs(data_df)

        self.set_data_df(data_df)

        self.get_num_samples()
        
        return self.data_df


    def tupleify_inputs(self, data_df):

        data_df["input_args"] = data_df["input_args"].apply(lambda x: (x.strip(),))
        return data_df


    def score(self, predictions):
        """
        Compute F1 & EM scores given pd.Series object of predictions (a column from results_df in the main run script).
        
        Returns dict of metrics and list of 0/1 values indicating where errors occurred.
        """

        targets = self.data_df.targets
        preds = predictions.reset_index(drop=True)

        metrics, errors = score_qa(targets, preds, numerical=self.numerical_scoring, gsm8k=self.gsm8k)

        return metrics, errors
