import argparse

import torch
from datasets import load_dataset
from langchain import LLMChain
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer

from data.prompts.single_prompts import prompt_templates as templates


def get_result(llm, alias, attribute_value, message, reply):
    if attribute_value == 'DATA_EXPIRED':
        return 'not'
    else:
        prompt = templates[alias]
        llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
        response = llm_chain.invoke({alias: attribute_value, 'message': message, 'reply': reply})
        print(f'response : {response}')
        if str(response).lower() == 'yes':
            return 'iro'
        else:
            return 'not'


def run(args):
    from transformers import pipeline
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    pipeline = pipeline(
        "text-generation",  # task
        model=args.model_name,
        tokenizer=tokenizer,
        torch_dtype="auto",
        trust_remote_code=True,
        device_map="auto",
        do_sample=True,
        max_new_tokens=2,
        top_k=1,
        top_p=0.95,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )

    llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature': 1})
    #todo remove after testing
    output = llm.generate("""Irony is a literary or rhetorical device in which there is a discrepancy between what is said and what is meant, 
        or between expectations and reality. It often involves a contrast between appearance and reality, 
        or a situation in which the outcome is the opposite of what one might expect.
        Imagine that you are a 29 years old person. Following is a message and its' reply from a conversation. 
        
        Message : Because the last generation was Gen. Z and there’s no where to go past that, so we gotta start from the beginning.
        Reply   : but we should have just named the first generation "Alpha" instead of doing it now
        
        Do you find the above reply ironic? Answer Yes/No""")
    print(output)

    dataset = load_dataset('Multilingual-Perspectivist-NLU/EPIC', split='train')
    dataset = dataset.to_pandas()
    print('dataset loading finished')

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
        llm()
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
    # todo remove once tested
    import os

    os.environ['HF_HOME'] = '../cache/'
    parser = argparse.ArgumentParser(description='''evaluate llms''')
    parser.add_argument('--model_name', type=str, required=True, help='model_name')

    args = parser.parse_args()
    run(args)
