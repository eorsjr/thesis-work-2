import os
import pandas as pd
from itertools import permutations

def load_client(backend, key):

    if backend=="openai":
        from openai import OpenAI
        
        return OpenAI(api_key=key)
    
    elif backend=="llama":
        import torch
        import transformers
        from huggingface_hub import login
        
        login(token=key)
        
        pipeline = transformers.pipeline("text-generation",
                                         model="meta-llama/Llama-3.1-8B-Instruct",
                                         model_kwargs={"dtype": torch.bfloat16},
                                         device_map="auto")
        return pipeline
    
    else:
        raise ValueError(f"Unsupported backend: {backend}")

# --- Triplet Ranking ---

TRIPLET_PERMUTATIONS = list(permutations([0, 1, 2]))

def build_triplet(row, permutation):
    labeled_alternatives = [("best", row["alt_best"]),
                            ("middle", row["alt_middle"]),
                            ("worst", row["alt_worst"])]

    permuted = [labeled_alternatives[i] for i in permutation]

    return {"ground_truth": row["ground_truth"],
            "alternatives": [alternative for _, alternative in permuted],
            "true_labels": [label for label, _ in permuted]}
    
def make_triplet_ranking_prompt(gt, a, b, c):
    return f"""Ground truth: "{gt}"

Below are three alternative translations of the same sentence. Please rank them from most similar to least similar in meaning compared to the ground truth.

A: "{a}"
B: "{b}"
C: "{c}"

Your response must follow these rules:

1. Rank all three options exactly once.
2. Each option (A, B, C) must appear one time only.
3. Use the strict format: 'A > B > C'
   - Three letters (A, B, C)
   - Two '>' symbols
4. Do not include any other text, explanation, or punctuation.
5. The response must contain only the ranking.

For example (the order is just an example): 'A > C > B'
"""

def make_triplet_ranking_prompt_with_confidence(gt, a, b, c):
    return f"""Ground truth: "{gt}"

Below are three alternative translations of the same sentence. Please rank them from most similar to least similar in meaning compared to the ground truth.

A: "{a}"
B: "{b}"
C: "{c}"

Your response must follow these rules:

1. Rank all three options exactly once.
2. Each option (A, B, C) must appear one time only.
3. Use the strict format: 'A > B > C | X'
   - Three letters (A, B, C)
   - Two '>' symbols
   - A vertical bar '|'
   - A confidence score X (integer from 1 to 10)
4. Do not include any other text, explanation, or punctuation.
5. The response must contain only the ranking and confidence.

For example:
A > C > B | 7
"""

def query_llm_triplet(prompt, backend, client, temperature, max_tokens):
    messages=[{"role": "system","content": "You are a linguistic evaluator. Your task is to rank alternative translations by their similarity to a ground truth sentence."},
              {"role": "user", "content": prompt}]
    
    if backend=="openai":
        response = client.chat.completions.create(model="gpt-4o-mini",
                                                  messages=messages,
                                                  temperature=temperature,
                                                  max_tokens=max_tokens)
        
        return response.choices[0].message.content.strip()
    
    elif backend=="llama":
        response = client(messages,
                          max_new_tokens=max_tokens,
                          temperature=temperature,
                          pad_token_id=128001)
        
        return response[0]["generated_text"][-1]["content"].strip()
    
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    
def generate_triplet_results(df, num_attempts, num_sentences, backend, client, output_dir, temperature=0.3, max_tokens=100, with_confidence=False):
    results = []
    my_index=0

    for i in range(num_attempts):
        for row_idx in range(num_sentences):
            
            row = df.iloc[row_idx]
            
            for perm_id, permutation in enumerate(TRIPLET_PERMUTATIONS):
                
                triplet = build_triplet(row, permutation)
                if (with_confidence):
                    prompt = make_triplet_ranking_prompt_with_confidence(triplet["ground_truth"],
                                                                        triplet["alternatives"][0],
                                                                        triplet["alternatives"][1],
                                                                        triplet["alternatives"][2])
                else: 
                    prompt = make_triplet_ranking_prompt(triplet["ground_truth"],
                                                        triplet["alternatives"][0],
                                                        triplet["alternatives"][1],
                                                        triplet["alternatives"][2])
                
                output = query_llm_triplet(prompt, backend, client, temperature, max_tokens)
                
                print(my_index, output)
                my_index = my_index + 1
                
                results.append({"run_id": i,
                                "row_index": row_idx,
                                "perm_id": perm_id,
                                "true_labels": triplet["true_labels"],
                                "response": output})
            
    os.makedirs(output_dir, exist_ok=True)
    
    if (with_confidence):
        output_file = os.path.join(output_dir, "triplet_ranking_with_confidence_results.csv")
    else:
        output_file = os.path.join(output_dir, "triplet_ranking_results.csv")

    pd.DataFrame(results).to_csv(output_file, index=False)
    
    print(f"\nResults saved to: {output_file}")

# --- Pairwise Ranking ---

PAIR_DEFINITIONS = [("best", "middle"),
                    ("middle", "best"),
                    ("best", "worst"),
                    ("worst", "best"),
                    ("middle", "worst"),
                    ("worst", "middle")]

def build_pair(row, pair_definition):
    labeled_alternatives = {"best": row["alt_best"],
                            "middle": row["alt_middle"],
                            "worst": row["alt_worst"]}

    left_label, right_label = pair_definition

    return {"ground_truth": row["ground_truth"],
            "alternatives": [labeled_alternatives[left_label], labeled_alternatives[right_label]],
            "labels": [left_label, right_label]}

def make_pairwise_ranking_prompt(gt, a, b):
    return f"""Ground truth: "{gt}"

Below are two alternative translations of the same sentence. Choose the one that is more similar in meaning to the ground truth.

A: "{a}"
B: "{b}"

Your response must follow these rules:

1. Respond with exactly one letter: A or B
2. Do not include any other text, explanation, or punctuation.
3. The response must contain only A or B
"""

def make_pairwise_ranking_prompt_with_confidence(gt, a, b):
    return f'''Ground truth: "{gt}"

Below are two alternative translations of the same sentence. Choose the one that is more similar in meaning to the ground truth.

A: "{a}"
B: "{b}"

Your response must follow these rules:

1. Respond using the strict format: 'A | X' or 'B | X'
2. The first part must be exactly one letter: A or B
3. The second part must be a confidence score X (integer from 1 to 10). 1 means least confident, 10 means most confident.
4. Use a vertical bar '|' between the answer and the confidence score
5. Do not include any other text, explanation, or punctuation.
6. The response must contain only the answer and confidence.

For example:
A | 7
'''

def query_llm_pairwise(prompt, backend, client, temperature, max_tokens):
    messages = [
        {"role": "system", "content": "You are a linguistic evaluator. Your task is to choose which alternative translation is more semantically similar to a ground truth sentence."},
        {"role": "user", "content": prompt}
    ]

    if backend == "openai":
        response = client.chat.completions.create(model="gpt-4o-mini",
                                                  messages=messages,
                                                  temperature=temperature,
                                                  max_tokens=max_tokens)
        
        return response.choices[0].message.content.strip()

    elif backend == "llama":
        response = client(messages,
                          max_new_tokens=max_tokens,
                          temperature=temperature,
                          pad_token_id=128001)
        
        return response[0]["generated_text"][-1]["content"].strip()
    
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    
def generate_pairwise_results(df, num_attempts, num_sentences, backend, client, output_dir, temperature=0.3, max_tokens=10, with_confidence=False):
    results = []
    my_index = 0

    for i in range(num_attempts):
        for row_idx in range(num_sentences):

            row = df.iloc[row_idx]

            for pair_id, pair_definition in enumerate(PAIR_DEFINITIONS):
                pair_data = build_pair(row, pair_definition)
                if (with_confidence):
                    prompt = make_pairwise_ranking_prompt_with_confidence(pair_data["ground_truth"],
                                                                        pair_data["alternatives"][0],
                                                                        pair_data["alternatives"][1])
                else: 
                    prompt = make_pairwise_ranking_prompt(pair_data["ground_truth"],
                                                        pair_data["alternatives"][0],
                                                        pair_data["alternatives"][1])


                output = query_llm_pairwise(prompt, backend, client, temperature, max_tokens)

                print(my_index, output)
                my_index += 1

                results.append({"run_id": i,
                                "row_index": row_idx,
                                "pair_id": pair_id,
                                "A_type": pair_data["labels"][0],
                                "B_type": pair_data["labels"][1],
                                "response": output})

    os.makedirs(output_dir, exist_ok=True)
    
    if (with_confidence):
        output_file = os.path.join(output_dir, "pairwise_ranking_with_confidence_results.csv")
    else: 
        output_file = os.path.join(output_dir, "pairwise_ranking_results.csv")
    
    pd.DataFrame(results).to_csv(output_file, index=False)
    
    print(f"\nResults saved to: {output_file}")

def main():
    from dotenv import load_dotenv
    load_dotenv()
    
    backend = "openai" # openai or llama
    with_confidence = False # True or False
    df = pd.read_csv("./data/input/translation_versions_final.csv")
    output_dir = f"data/output/ranking/{backend}"
    num_attempts = 10
    num_sentences = 100
    
    if backend=="llama":
        key = os.getenv("HF_TOKEN")
        
    elif backend=="openai":
        key = os.getenv("API_KEY")
    
    if key:
        print("Key/token found successfully.")
        client = load_client(backend=backend, key=key)
    else:
        print("Error: Key/token not found. Please set the environment variable.")
        return

    generate_triplet_results(df=df, num_attempts=num_attempts, num_sentences=num_sentences, backend=backend, client=client, output_dir=output_dir, with_confidence=with_confidence)
    generate_pairwise_results(df=df, num_attempts=num_attempts, num_sentences=num_sentences, backend=backend, client=client, output_dir=output_dir, with_confidence=with_confidence)

if __name__ == "__main__":
    main()