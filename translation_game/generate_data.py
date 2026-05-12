from datasets import load_dataset
from tqdm import tqdm
import os

ds = load_dataset("agentlans/high-quality-english-sentences")
sentences = ds['train']['text'][:1000]

file_path = "./data/input/1000_sentences.txt"

os.makedirs(os.path.dirname(file_path), exist_ok=True)

with open(file_path, "w") as file:
    for sentence in tqdm(sentences):
        file.write(f"{sentence}\n")

print(f"File saved: {file_path}")