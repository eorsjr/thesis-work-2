import pandas as pd

def assoc_bert_score(df, device):
    import os
    import numpy as np
    import transformers
    from bert_score import score
    
    # Suppress warnings
    transformers.logging.set_verbosity_error()
    
    references = df["ground_truth"].astype(str).tolist()
    
    # Compute all 10 BERTScore columns
    all_scores = []

    for i in range(1, 11):
        alt_col = f"alt{i}"
        candidates = df[alt_col].astype(str).tolist()
        _, _, F1 = score(candidates, references, lang="en", rescale_with_baseline=True, device=device)
        all_scores.append(F1.cpu().numpy())

    score_matrix = np.vstack(all_scores).T
    output_rows = []

    for idx in range(len(df)):
        scores = score_matrix[idx]

        # Best, worst, middle
        best_idx = np.argmax(scores)
        worst_idx = np.argmin(scores)

        median_value = np.median(scores)
        middle_idx = np.argmin(np.abs(scores - median_value))

        alt_best = df.loc[idx, f"alt{best_idx + 1}"]
        alt_middle = df.loc[idx, f"alt{middle_idx + 1}"]
        alt_worst = df.loc[idx, f"alt{worst_idx + 1}"]

        output_rows.append([df.loc[idx, "ground_truth"],
                            alt_best,
                            alt_middle,
                            alt_worst,
                            scores[best_idx],
                            scores[middle_idx],
                            scores[worst_idx]])

    output_df = pd.DataFrame(output_rows,
                             columns=["ground_truth",
                                      "alt_best",
                                      "alt_middle",
                                      "alt_worst",
                                      "score_best",
                                      "score_middle",
                                      "score_worst"])

    output_path = "./data/input/translation_versions_final.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"Saved CSV with associated bert scores to {output_path}")
    
def main():
    df = pd.read_csv("./data/input/translation_versions_cleaned.csv", header=None)
    df = df.iloc[:, :12]
    df.columns = ["gloss", "ground_truth"] + [f"alt{i}" for i in range(1, 11)]
    assoc_bert_score(df, device="mps")
    
if __name__ == "__main__":
    main()