import tinker
from dotenv import load_dotenv
from tinker import types
from tinker_cookbook import renderers, tokenizer_utils
import numpy as np
load_dotenv()

messages =[
    {'role': 'system', 'content': 'Answer concisely; at most one sentence per response'},
    {'role': 'user', 'content': 'What is the longest-lived rodent species?'},
    {'role': 'assistant', 'content': 'The naked mole rat, which can live over 30 years.'},
    {'role': 'user', 'content': 'How do they live so long?'},
    {'role': 'assistant', 'content': 'They evolved multiple protective mechanisms including special hyaluronic acid that prevents cancer, extremely stable proteins, and efficient DNA repair systems that work together to prevent aging.'}
]

tokenizer = tokenizer_utils.get_tokenizer('Qwen/Qwen3-30B-A3B')
renderer = renderers.get_renderer('qwen3', tokenizer)
prompt = renderer.build_generation_prompt(messages[:-1])
print(prompt)
print('-'*10)
print(tokenizer.decode(prompt.to_ints()))