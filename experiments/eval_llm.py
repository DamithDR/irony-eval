import argparse
from collections import deque

from datasets import load_dataset
from langchain import LLMChain
from tqdm import tqdm
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
    print('generating prompts...')
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
            prompt = prompt.replace('{country_of_residence}', country_of_residence).replace('{message}',
                                                                                            message).replace('{reply}',
                                                                                                             reply)
            prompts_list.append(prompt)
        if nationality != 'DATA_EXPIRED':
            prompt = string_templates['nationality']
            prompt = prompt.replace('{nationality}', nationality).replace('{message}', message).replace('{reply}',
                                                                                                        reply)
            prompts_list.append(prompt)
        if is_student != 'DATA_EXPIRED':
            if str(is_student).lower() == 'yes':
                is_student = 'a'
            else:
                is_student = 'not a'
            prompt = string_templates['is_student']
            prompt = prompt.replace('{is_student}', is_student).replace('{message}', message).replace('{reply}', reply)
            prompts_list.append(prompt)
        if employment_mode != 'DATA_EXPIRED':
            if str(employment_mode).lower().startswith('unemployed') or str(employment_mode).lower().startswith(
                    'not in paid work'):
                employment_mode = 'not'
            prompt = string_templates['employment_mode']
            prompt = prompt.replace('{employment_mode}', employment_mode).replace('{message}', message).replace(
                '{reply}', reply)
            prompts_list.append(prompt)

    return prompts_list


def reindex_results(results, dataset):
    print('results re-indexing...')
    with tqdm(total=len(dataset)) as pbar:
        pbar.set_description('re-indexing results')
        results_queue = deque(results)
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
        for age, sex, ethnicity, country_of_birth, country_of_residence, nationality, is_student, employment_mode in zip(
                dataset['Age'], dataset['Sex'], dataset['Ethnicity simplified'], dataset['Country of birth'],
                dataset['Country of residence'], dataset['Nationality'], dataset['Student status'],
                dataset['Employment status']):
            if age != 'DATA_EXPIRED':
                results_map['age'].append(results_queue.popleft())
            else:
                results_map['age'].append('not')  # this is to handle DATA_EXPIRED case
            if sex != 'DATA_EXPIRED':
                results_map['sex'].append(results_queue.popleft())
            else:
                results_map['sex'].append('not')  # this is to handle DATA_EXPIRED case
            if ethnicity != 'DATA_EXPIRED':
                results_map['ethnicity'].append(results_queue.popleft())
            else:
                results_map['ethnicity'].append('not')  # this is to handle DATA_EXPIRED case
            if country_of_birth != 'DATA_EXPIRED':
                results_map['country_of_birth'].append(results_queue.popleft())
            else:
                results_map['country_of_birth'].append('not')  # this is to handle DATA_EXPIRED case
            if country_of_residence != 'DATA_EXPIRED':
                results_map['country_of_residence'].append(results_queue.popleft())
            else:
                results_map['country_of_residence'].append('not')  # this is to handle DATA_EXPIRED case
            if nationality != 'DATA_EXPIRED':
                results_map['nationality'].append(results_queue.popleft())
            else:
                results_map['nationality'].append('not')  # this is to handle DATA_EXPIRED case
            if is_student != 'DATA_EXPIRED':
                results_map['is_student'].append(results_queue.popleft())
            else:
                results_map['is_student'].append('not')  # this is to handle DATA_EXPIRED case
            if employment_mode != 'DATA_EXPIRED':
                results_map['employment_mode'].append(results_queue.popleft())
            else:
                results_map['employment_mode'].append('not')  # this is to handle DATA_EXPIRED case
            pbar.update(1)

    dataset['age_results'] = results_map['age']
    dataset['sex_results'] = results_map['sex']
    dataset['ethnicity_results'] = results_map['ethnicity']
    dataset['country_of_birth_results'] = results_map['country_of_birth']
    dataset['country_of_residence_results'] = results_map['country_of_residence']
    dataset['nationality_results'] = results_map['nationality']
    dataset['is_student_results'] = results_map['is_student']
    dataset['employment_mode_results'] = results_map['employment_mode']

    return dataset


def resolve_results(results, dataset):
    print('results resolving...')
    result_list = []
    for result in results:
        text = result[0]['generated_text']
        if str(text).lower().startswith('yes'):
            result_list.append('iro')
        else:
            result_list.append('not')
    return reindex_results(results=result_list, dataset=dataset)


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
        batch_size=args.batch_size,
        top_k=1,
        top_p=0.95,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )


    dataset = load_dataset('Multilingual-Perspectivist-NLU/EPIC', split='train')
    dataset = dataset.to_pandas()
    print('dataset loading finished')

    # testing
    dataset = dataset[300:305]

    prompt_list = generate_prompts(dataset)
    prompt_set = ListDataset(prompt_list)

    print('predicting outputs...')
    results = pipe(prompt_set)

    # testing
    print_results = [result[0]['generated_text'] for result in results]

    with open('raw_results.txt', 'w') as f:
        f.writelines("\n".join(print_results))

    dataset = resolve_results(results, dataset)
    dataset.to_csv('final_results.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''evaluate llms''')
    parser.add_argument('--model_name', type=str, required=True, help='model_name')
    parser.add_argument('--batch_size', type=int, required=False, default=64, help='batch_size')

    args = parser.parse_args()
    run(args)
