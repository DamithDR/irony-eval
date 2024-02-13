import argparse
import sys

import pandas as pd
import torch
# from langchain import HuggingFacePipeline
# from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from langchain import PromptTemplate, LLMChain
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

from datasets import load_dataset

from data.prompts.single_prompts import prompt_templates as templates


def get_result(llm, alias, attribute_value, message, reply):
    if attribute_value == 'DATA_EXPIRED':
        return 'not'
    else:
        prompt = templates[alias]
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        response = llm_chain.run({alias: attribute_value, 'message': message, 'reply': reply})
        if str(response).lower() == 'yes':
            return 'iro'
        else:
            return 'not'


def run(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    pipeline = pipeline(
        "text-generation",  # task
        model=args.model_name,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=10,
        do_sample=True,
        top_k=10,
        num_return_sequences=10,
        eos_token_id=tokenizer.eos_token_id
    )

    llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature': 1, 'do_sample': True})

    dataset = load_dataset('Multilingual-Perspectivist-NLU/EPIC', split='train')
    dataset = dataset.to_pandas()

    # todo remove after testing
    dataset = dataset[:10]

    results_map = {
        'age': [],
        'sex': [],
        'ethnicity': [],
        'country_of_birth': [],
        'country_of_residence': [],
        'nationality': [],
        'is_student': [],
        'employment_mode': [],
    }

    for age, sex, ethnicity, country_of_birth, country_of_residence, nationality, is_student, employment_mode, message, reply in zip(
            dataset['Age'], dataset['Sex'], dataset['Ethnicity simplified'], dataset['Country of birth'],
            dataset['Country of residence'], dataset['Nationality'], dataset['Student status'],
            dataset['Employment status'], dataset['parent_text'], dataset['text']):
        results_map['age'].append(get_result(llm=llm, alias='age', attribute_value=age, message=message, reply=reply))

    dataset['age_results'] = results_map['age']
    dataset['sex_results'] = results_map['sex']
    dataset['ethnicity_results'] = results_map['ethnicity']
    dataset['country_of_birth_results'] = results_map['country_of_birth']
    dataset['country_of_residence_results'] = results_map['country_of_residence']
    dataset['nationality_results'] = results_map['nationality']
    dataset['is_student_results'] = results_map['is_student']
    dataset['employment_mode_results'] = results_map['employment_mode']

    dataset.to_csv('final_results.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''evaluate llms''')
    parser.add_argument('--model_name', type=str, required=True, help='model_name')

    args = parser.parse_args()
    run(args)
