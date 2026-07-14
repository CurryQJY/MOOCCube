import argparse
import json
import torch
import os
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

def get_local_embedding_by_API(text_list):
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://172.23.166.108:8021/v1",
    )

    max_token = 8190
    tokenizer = AutoTokenizer.from_pretrained("{your_path}/qwen3-embedding-8B", use_fast=True)
    truncated_text_list = []
    for text in text_list:
        tokens = tokenizer.encode(text)
        if len(tokens) > max_token:
            tokens = tokens[:max_token]
        truncated_text = tokenizer.decode(tokens)
        truncated_text_list.append(truncated_text)


    embedding_list = []
    for text in tqdm(truncated_text_list, desc="Getting embeddings"):
        completion = client.embeddings.create(model="Qwen3-Embedding-8B", input=[text])
        embedding = completion.data[0].embedding
        embedding_list.append(embedding)
    embedding = torch.tensor(embedding_list)
    print(f"Completion result: {embedding.shape}")

    return embedding

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--dataset", type=str, default="mooccubex")
    args = parser.parse_args()
    
    text_file = f"../data/{args.dataset}/resource_embedding_prompt.json"
    text_list = ["None"]
    with open(text_file, "r") as f:
        for key, line in enumerate(f):
            text_json = json.loads(line.strip())
            value = text_json[str(key+1)]
            text_list.append(value)
    embedding = get_local_embedding_by_API(text_list)
    out_file = f"../data/{args.dataset}/resource_semantic_embedding_qwen3.pt"
    torch.save(embedding, out_file)
    print(f"Saved embedding to {out_file}")