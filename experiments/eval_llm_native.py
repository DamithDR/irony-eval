# Load model directly
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from experiments.eval_llm import generate_prompts

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

dataset = load_dataset('Multilingual-Perspectivist-NLU/EPIC', split='train')
dataset = dataset.to_pandas()
print('dataset loading finished')

# testing
# dataset = dataset[300:302]
# prompt_list = generate_prompts(dataset)

while(True):
    prompt = input("input : ")
    prompt_list = [prompt]

    generation_config = GenerationConfig(
        max_new_tokens=10, do_sample=True, top_k=50, eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.eos_token_id, temperature=0.2,
        num_return_sequences=1,torch_dtype=torch.bfloat16,
    )

    encoding = tokenizer(prompt_list, padding=True, truncation=False, return_tensors="pt").to(model.device)
    outputs = model.generate(input_ids=encoding.input_ids, attention_mask=encoding.attention_mask,
                             generation_config=generation_config)
    detach = outputs.detach().cpu().numpy()
    outputs = detach.tolist()
    out_list = []
    out_list.extend([tokenizer.decode(out, skip_special_tokens=True) for out in outputs])

    print([tokenizer.decode(out, skip_special_tokens=True) for out in outputs])

    with open('raw_results_from_model.txt', 'w') as f:
        f.writelines("\n".join(out_list))
