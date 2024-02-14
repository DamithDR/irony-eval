import argparse

import torch
from datasets import load_dataset
from langchain import LLMChain
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer

from data.prompts.single_prompts import prompt_templates as templates, string_templates
from util.ListDataset import ListDataset


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


def generate_prompts(dataset):
    prompts_list = []
    for age, sex, ethnicity, country_of_birth, country_of_residence, nationality, is_student, employment_mode, message, reply in zip(
            dataset['Age'], dataset['Sex'], dataset['Ethnicity simplified'], dataset['Country of birth'],
            dataset['Country of residence'], dataset['Nationality'], dataset['Student status'],
            dataset['Employment status'], dataset['parent_text'], dataset['text']):
        if age != 'DATA_EXPIRED':
            prompt = string_templates['age']
            prompt = prompt.replace('{age}', age).replace('{message}', message).replace('{reply}', reply)
            prompts_list.append(prompt)
        if sex != 'DATA_EXPIRED':
            prompt = string_templates['sex']
            prompt = prompt.replace('{sex}', sex).replace('{message}', message).replace('{reply}', reply)
            prompts_list.append(prompt)
        if ethnicity != 'DATA_EXPIRED':
            prompt = string_templates['ethnicity']
            prompt = prompt.replace('{ethnicity}', ethnicity).replace('{message}', message).replace('{reply}', reply)
            prompts_list.append(prompt)
        if country_of_birth != 'DATA_EXPIRED':
            prompt = string_templates['country_of_birth']
            prompt = prompt.replace('{country_of_birth}', country_of_birth).replace('{message}', message).replace(
                '{reply}', reply)
            prompts_list.append(prompt)
        if country_of_residence != 'DATA_EXPIRED':
            prompt = string_templates['country_of_residence']
            prompt = prompt.replace('{country_of_residence}', country_of_residence).replace('{message}', message).replace('{reply}', reply)
            prompts_list.append(prompt)
        if nationality != 'DATA_EXPIRED':
            prompt = string_templates['nationality']
            prompt = prompt.replace('{nationality}', nationality).replace('{message}', message).replace('{reply}', reply)
            prompts_list.append(prompt)
        if is_student != 'DATA_EXPIRED':
            prompt = string_templates['is_student']
            prompt = prompt.replace('{is_student}', is_student).replace('{message}', message).replace('{reply}', reply)
            prompts_list.append(prompt)
        if employment_mode != 'DATA_EXPIRED':
            prompt = string_templates['employment_mode']
            prompt = prompt.replace('{employment_mode}', employment_mode).replace('{message}', message).replace('{reply}', reply)
            prompts_list.append(prompt)

    return prompts_list


def resolve_results(results):
    print("Result:", results)


def run(args):
    from transformers import pipeline
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    pipe = pipeline(
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

    # llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature': 1})
    dataset = load_dataset('Multilingual-Perspectivist-NLU/EPIC', split='train')
    dataset = dataset.to_pandas()
    print('dataset loading finished')

    # todo remove after testing
    dataset = dataset[:1]

    prompt_list = generate_prompts(dataset)
    # prompts_set = ListDataset(prompt_list)

    results = pipe(prompt_list)

    resolve_results(results)

    # results_map = {
    #     'age': [],
    #     'sex': [],
    #     'ethnicity': [],
    #     'country_of_birth': [],
    #     'country_of_residence': [],
    #     'nationality': [],
    #     'is_student': [],
    #     'employment_mode': [],
    # }
    #
    # for age, sex, ethnicity, country_of_birth, country_of_residence, nationality, is_student, employment_mode, message, reply in zip(
    #         dataset['Age'], dataset['Sex'], dataset['Ethnicity simplified'], dataset['Country of birth'],
    #         dataset['Country of residence'], dataset['Nationality'], dataset['Student status'],
    #         dataset['Employment status'], dataset['parent_text'], dataset['text']):
    #     llm()
    #     results_map['age'].append(get_result(llm=llm, alias='age', attribute_value=age, message=message, reply=reply))
    #
    # dataset['age_results'] = results_map['age']
    # dataset['sex_results'] = results_map['sex']
    # dataset['ethnicity_results'] = results_map['ethnicity']
    # dataset['country_of_birth_results'] = results_map['country_of_birth']
    # dataset['country_of_residence_results'] = results_map['country_of_residence']
    # dataset['nationality_results'] = results_map['nationality']
    # dataset['is_student_results'] = results_map['is_student']
    # dataset['employment_mode_results'] = results_map['employment_mode']
    #
    # dataset.to_csv('final_results.csv', index=False)


if __name__ == '__main__':
    # todo remove once tested
    import os

    os.environ['HF_HOME'] = '../cache/'
    parser = argparse.ArgumentParser(description='''evaluate llms''')
    parser.add_argument('--model_name', type=str, required=True, help='model_name')

    args = parser.parse_args()
    run(args)
