# Author: David

import pandas as pd
import pickle
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import torch
import numpy as np
import os

def generate_one_fortune(model_path=os.path.join('..', 'model', 'model_state_dict.pkl'), device="cuda"):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2',
                                          bos_token='<|startoftext|>',
                                          eos_token='<|endoftext|>',
                                          pad_token='<|pad|>')
    
    configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)

    model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration, ignore_mismatched_sizes=True)

    model.resize_token_embeddings(len(tokenizer))
    
    with open(model_path, 'rb') as f:
        loaded_state_dict = pickle.load(f)

    model.load_state_dict(loaded_state_dict)
    
    model.to(device)

    model.eval()

    prompt = "<|startoftext|>"
    output_fortunes = []

    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    generated = generated.to(device)

    sample_outputs = model.generate(
                                    generated,
                                    do_sample=True,
                                    top_k=50,
                                    max_length = 300,
                                    top_p=0.95,
                                    num_return_sequences=1
                                    )
    
    return tokenizer.decode(sample_outputs[0], skip_special_tokens=True)


generate_one_fortune()