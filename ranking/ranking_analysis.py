import re
import random
import hashlib
import pandas as pd
from ast import literal_eval

SEED = 42

TRIPLET_ONLY_PATTERN = re.compile(r"^[ABC]\s*>\s*[ABC]\s*>\s*[ABC]\s*$")
TRIPLET_CONF_PATTERN = re.compile(r"^[ABC]\s*>\s*[ABC]\s*>\s*[ABC]\s*\|\s*(10|[1-9])$")

PAIRWISE_ONLY_PATTERN = re.compile(r"^\s*[AB]\s*$")
# PAIRWISE_CONF_PATTERN = re.compile(r"^\s*[AB]\s*\|\s*(10|[1-9])\s*$")
PAIRWISE_CONF_PATTERN = re.compile(r"^\s*[AB]\s*\|\s*(10|[0-9])\s*$") # allow 0

SOURCE_DATA_PATH = "./data/input/translation_versions_final.csv"
CANDIDATE_LOOKUP = None

# --- Helpers ---

def write_lines_to_file(lines, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))

def add_mitigation_result(results, experiment_type, method, metric, value, tie_breaking_strategy=None):
    results.append({"experiment_type": experiment_type,
                    "method": method,
                    "metric": metric,
                    "value": f"{value:.2%}",
                    "tie_breaking_strategy": tie_breaking_strategy})
    
# --- Helpers - Tie Breaking ---

def ensure_candidate_lookup():
    global CANDIDATE_LOOKUP

    if CANDIDATE_LOOKUP is None:
        source_df = pd.read_csv(SOURCE_DATA_PATH).reset_index(drop=True)
        CANDIDATE_LOOKUP = {}

        for row_idx, row in source_df.iterrows():
            CANDIDATE_LOOKUP[row_idx] = {"best": row["alt_best"],
                                         "middle": row["alt_middle"],
                                         "worst": row["alt_worst"]}

    return CANDIDATE_LOOKUP

def label_tie_key(row_idx, label):
    candidate_lookup = ensure_candidate_lookup()
    text = candidate_lookup[row_idx][label]
    s = f"{SEED}|{row_idx}|{text}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def break_label_tie(row_idx, tied_labels):
    return min(tied_labels, key=lambda label: label_tie_key(row_idx, label))

def order_label_tie(row_idx, tied_labels):
    return sorted(tied_labels, key=lambda label: label_tie_key(row_idx, label))

def ranking_tie_key(row_idx, ranking):
    candidate_lookup = ensure_candidate_lookup()
    labels = [label.strip() for label in ranking.split(">")]
    texts = [candidate_lookup[row_idx][label] for label in labels]
    s = f"{SEED}|{row_idx}|{' || '.join(texts)}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def break_ranking_tie(row_idx, tied_rankings):
    return min(tied_rankings, key=lambda ranking: ranking_tie_key(row_idx, ranking))

# --- TRIPLET RANKING ---

# --- Triplet Ranking Stability Reports ---

def report_ranking_variation_across_runs(df, output_path):
    lines = []

    header = f"Across {df['run_id'].nunique()} runs:\n"
    lines.append(header)

    for row_idx, row_group in df.groupby("row_index"):
        sentence_header = f"For sentence {row_idx}:\n"
        lines.append(sentence_header)

        for perm_id, perm_group in row_group.groupby("perm_id"):
            rankings = perm_group["converted_ranking"].tolist()
            unique_rankings = sorted(set(rankings))

            permutation_header = f"Permutation {perm_id}:"
            variation_line = f"{len(unique_rankings)} different rankings:"

            lines.append(permutation_header)
            lines.append(variation_line)

            for index, ranking in enumerate(unique_rankings, start=1):
                ranking_line = f"{index}. {ranking}"
                lines.append(ranking_line)

            lines.append("")

    write_lines_to_file(lines, output_path)

def report_ranking_variation_across_permutations(df, output_path):
    lines = []

    header = f"Across {df['perm_id'].nunique()} permutations:\n"
    lines.append(header)

    for run_id, run_group in df.groupby("run_id"):
        run_header = f"For run {run_id}:\n"
        lines.append(run_header)

        for row_idx, row_group in run_group.groupby("row_index"):
            rankings = row_group["converted_ranking"].tolist()
            unique_rankings = sorted(set(rankings))

            sentence_header = f"Sentence {row_idx}:"
            variation_line = f"{len(unique_rankings)} different rankings:"

            lines.append(sentence_header)
            lines.append(variation_line)

            for index, ranking in enumerate(unique_rankings, start=1):
                ranking_line = f"{index}. {ranking}"
                lines.append(ranking_line)
                
            lines.append("")

    write_lines_to_file(lines, output_path)

def count_distinct_rankings(df, cols):
    distinct_counts = []

    for _, group in df.groupby(cols):
        rankings = group["converted_ranking"].tolist()
        unique_rankings = set(rankings)
        distinct_counts.append(len(unique_rankings))

    return distinct_counts

def build_stability_summary(distinct_counts, max_rankings):
    mean_value = sum(distinct_counts) / len(distinct_counts)
    variance_value = pd.Series(distinct_counts).var()

    distribution = {}
    for i in range(1, max_rankings + 1):
        distribution[f"{i}_rankings"] = distinct_counts.count(i)

    return {"mean_distinct_rankings": mean_value,
            "variance": variance_value,
            "distribution": distribution,
            "total_groups": len(distinct_counts)}
    
def compute_run_stability(df, output_path):
    distinct_counts = count_distinct_rankings(df, ["row_index", "perm_id"])
    stats = build_stability_summary(distinct_counts, max_rankings=6)

    lines = []
    lines.append(f"Total sentence-permutation pairs: {stats['total_groups']}\n")
    lines.append(f"Distribution of distinct rankings across {df['run_id'].nunique()} runs:\n")

    for key, value in stats["distribution"].items():
        lines.append(f"{key}: {value}")

    lines.append(f"\nMean distinct rankings: {stats['mean_distinct_rankings']:.3f}")
    lines.append(f"\nVariance: {stats['variance']:.6f}\n")

    write_lines_to_file(lines, output_path)
    
def compute_permutation_stability(df, output_path):
    distinct_counts = count_distinct_rankings(df, ["run_id", "row_index"])
    stats = build_stability_summary(distinct_counts, max_rankings=6)

    lines = []
    lines.append(f"Total run-sentence pairs: {stats['total_groups']}\n")
    lines.append(f"Distribution of distinct rankings across {df['perm_id'].nunique()} permutations:\n")

    for key, value in stats["distribution"].items():
        lines.append(f"{key}: {value}")

    lines.append(f"\nMean distinct rankings: {stats['mean_distinct_rankings']:.3f}")
    lines.append(f"\nVariance: {stats['variance']:.6f}\n")

    write_lines_to_file(lines, output_path)
    
def subsampling_consistency_analysis(df, output_path, sample_size=80, n_samples=10):
    random.seed(SEED)

    lines = []
    header = f"\nRunning subsampling consistency analysis: {n_samples} samples, each with {sample_size} sentences.\n"
    lines.append(header)

    row_indices = df["row_index"].unique()
    sample_consistencies = []

    for sample_index in range(n_samples):
        sampled_rows = random.sample(list(row_indices), sample_size)
        subset = df[df["row_index"].isin(sampled_rows)]

        subset_consistency_scores = []

        for _, row_group in subset.groupby("row_index"):
            rankings = row_group["converted_ranking"].tolist()
            most_common_ranking = max(set(rankings), key=rankings.count)
            consistency = rankings.count(most_common_ranking) / len(rankings)
            subset_consistency_scores.append(consistency)

        if subset_consistency_scores:
            average_consistency = sum(subset_consistency_scores) / len(subset_consistency_scores)
        else:
            average_consistency = 0

        sample_consistencies.append(average_consistency)

        line = f"Sample {sample_index + 1}: average consistency = {average_consistency:.3f}"
        lines.append(line)

    mean_consistency = pd.Series(sample_consistencies).mean()
    variance = pd.Series(sample_consistencies).var()

    summary_1 = f"\nSubsampling consistency summary across {n_samples} samples:"
    summary_2 = f"Mean consistency: {mean_consistency:.3f}"
    summary_3 = f"Variance across samples: {variance:.6f}\n"

    lines.extend([summary_1, summary_2, summary_3])
    write_lines_to_file(lines, output_path)

def check_correct_ranking_exists(df, expected_ranking="best > middle > worst"):
    sentences_with_correct = 0
    sentences_without_correct = []

    for row_idx, group in df.groupby("row_index"):
        rankings = group["converted_ranking"].dropna().tolist()

        if expected_ranking in rankings:
            sentences_with_correct += 1
        else:
            sentences_without_correct.append(row_idx)

    print("\n--- Diagnostics: Correct Ranking ---")
    print(f"Sentences where correct ranking exists: {sentences_with_correct}")
    print(f"Sentences without correct ranking: {len(sentences_without_correct)}")

    if sentences_without_correct:
        print("Sentence IDs without correct ranking:")
        print(sentences_without_correct, "\n")

# --- Triplet Results Prep ---

def detect_triplet_response_type(text):
    stripped = str(text).strip().upper()
    ranking_part = stripped.split("|")[0].strip()

    letters = []
    for char in ranking_part:
        if char in "ABC":
            letters.append(char)

    has_valid_letters = sorted(letters) == ["A", "B", "C"]

    if TRIPLET_ONLY_PATTERN.match(stripped) and has_valid_letters:
        return "ranking_only"

    if TRIPLET_CONF_PATTERN.match(stripped) and has_valid_letters:
        return "ranking_with_confidence"

    return None

def parse_original_ranking(response):
    ranking_part = str(response).upper().split("|")[0].strip()
    letters = []

    for char in ranking_part:
        if char in "ABC":
            letters.append(char)

    if len(letters) != 3:
        raise ValueError(f"Failed to parse original ranking: {response}")

    return " > ".join(letters)

def parse_converted_ranking(response, true_labels):
    original_ranking = parse_original_ranking(response)
    letters = original_ranking.split(" > ")

    mapping = {"A": true_labels[0],
               "B": true_labels[1],
               "C": true_labels[2]}

    converted_labels = []
    for letter in letters:
        converted_labels.append(mapping[letter])

    return " > ".join(converted_labels)

def parse_confidence(response):
    response_text = str(response).strip()
    confidence = int(response_text.split("|")[1].strip())
    return confidence

def prepare_triplet_results(df):
    prepared_rows = []

    # --- Detect response type ---
    response_types = []

    for response in df["response"]:
        response_type = detect_triplet_response_type(response)
        response_types.append(response_type)

    invalid_count = response_types.count(None)

    if invalid_count > 0:
        raise ValueError(f"Found {invalid_count} invalid triplet responses.")

    if len(set(response_types)) != 1:
        raise ValueError("Mixed response formats detected.")

    response_type = response_types[0]
    
    has_confidence = response_type == "ranking_with_confidence"

    # --- Convert each row into the prepared triplet format ---
    for _, row in df.iterrows():
        true_labels = literal_eval(row["true_labels"])

        prepared_row = {"run_id": row["run_id"],
                        "row_index": row["row_index"],
                        "perm_id": row["perm_id"],
                        "original_ranking": parse_original_ranking(row["response"]),
                        "converted_ranking": parse_converted_ranking(row["response"], true_labels)}

        if has_confidence:
            prepared_row["confidence"] = parse_confidence(row["response"])

        prepared_rows.append(prepared_row)

    return pd.DataFrame(prepared_rows)

# --- Triplet Majority Voting ---

def apply_majority_voting_by_group(df, cols, tie_breaking_strategy):
    aggregated = []

    for group_values, group in df.groupby(cols):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        # --- Count votes and confidence totals for each ranking ---
        ranking_counts = {}
        confidence_totals = {}

        for _, row in group.iterrows():
            ranking = row["converted_ranking"]
            ranking_counts[ranking] = ranking_counts.get(ranking, 0) + 1

            if "confidence" in group.columns:
                confidence_totals[ranking] = confidence_totals.get(ranking, 0) + row["confidence"]

        # --- Find rankings with the highest vote count ---
        max_count = max(ranking_counts.values())

        tied_rankings = []
        for ranking, count in ranking_counts.items():
            if count == max_count:
                tied_rankings.append(ranking)


        # --- Resolve the winning ranking ---
        if len(tied_rankings) == 1:
            winner = tied_rankings[0]

        elif tie_breaking_strategy == "random":
            winner = break_ranking_tie(group_values[0], tied_rankings)

        elif tie_breaking_strategy == "confidence":
            if "confidence" not in group.columns:
                winner = break_ranking_tie(group_values[0], tied_rankings)
            else:
                max_confidence = max(confidence_totals[ranking] for ranking in tied_rankings)

                best_rankings = []
                for ranking in tied_rankings:
                    if confidence_totals[ranking] == max_confidence:
                        best_rankings.append(ranking)

                if len(best_rankings) == 1:
                    winner = best_rankings[0]
                else:
                    winner = break_ranking_tie(group_values[0], best_rankings)

        else:
            raise ValueError(f"Unsupported tie_breaking_strategy: {tie_breaking_strategy}")

        # --- Store the aggregated result row ---
        result_row = {}

        for column, value in zip(cols, group_values):
            result_row[column] = value

        result_row["converted_ranking"] = winner
        aggregated.append(result_row)

    return pd.DataFrame(aggregated)

def apply_majority_voting_per_sentence(df, tie_breaking_strategy):
    return apply_majority_voting_by_group(df, cols=["row_index"], tie_breaking_strategy=tie_breaking_strategy)

def apply_majority_voting_over_permutations(df, tie_breaking_strategy):
    return apply_majority_voting_by_group(df, cols=["row_index", "run_id"], tie_breaking_strategy=tie_breaking_strategy)

# --- Triplet Copeland Voting ---

def apply_copeland_voting_by_group(df, cols, tie_breaking_strategy):
    aggregated = []

    for group_values, group in df.groupby(cols):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        row_idx = group_values[0]

        # --- Build a stable label order from first appearance in the rankings ---
        label_order = []
        seen = set()

        for _, row in group.iterrows():
            for label in row["converted_ranking"].split(">"):
                label = label.strip()
                if label not in seen:
                    seen.add(label)
                    label_order.append(label)
                    
        # --- Enumerate all pairwise label comparisons ---
        label_pairs = []
        pair_index = 0
        for i in range(len(label_order)):
            for j in range(i + 1, len(label_order)):
                label_pairs.append((label_order[i], label_order[j], pair_index))
                pair_index += 1
        
        # --- Track Copeland wins and confidence support for each label ---
        pairwise_wins = {label: 0 for label in label_order}
        confidence_scores = {label: 0 for label in label_order}

        # --- Resolve each pairwise comparison across the whole group ---
        for label_a, label_b, pair_index in label_pairs:
            label_a_votes = 0
            label_b_votes = 0
            label_a_confidence = 0
            label_b_confidence = 0
            
            # --- Count how often each label is preferred in this pair ---
            for _, row in group.iterrows():
                ranking_labels = [label.strip() for label in row["converted_ranking"].split(">")]
                if ranking_labels.index(label_a) < ranking_labels.index(label_b):
                    preferred_label = label_a
                else:
                    preferred_label = label_b

                if "confidence" in group.columns:
                    confidence = row["confidence"]
                else:
                    confidence = 1

                if preferred_label == label_a:
                    label_a_votes += 1
                    label_a_confidence += confidence
                else:
                    label_b_votes += 1
                    label_b_confidence += confidence
                    
            # --- Choose the winner of this pairwise matchup ---
            if label_a_votes > label_b_votes:
                pairwise_winner = label_a

            elif label_b_votes > label_a_votes:
                pairwise_winner = label_b

            elif tie_breaking_strategy == "random":
                pairwise_winner = break_label_tie(row_idx, [label_a, label_b])

            elif tie_breaking_strategy == "confidence":
                if label_a_confidence > label_b_confidence:
                    pairwise_winner = label_a
                elif label_b_confidence > label_a_confidence:
                    pairwise_winner = label_b
                else:
                    pairwise_winner = break_label_tie(row_idx, [label_a, label_b])

            else:
                raise ValueError(f"Unsupported tie_breaking_strategy: {tie_breaking_strategy}")
            
             # --- Add one Copeland win to the pairwise winner ---
            pairwise_wins[pairwise_winner] += 1

            if pairwise_winner == label_a:
                confidence_scores[pairwise_winner] += label_a_confidence
            else:
                confidence_scores[pairwise_winner] += label_b_confidence
                
        # --- Group labels by their Copeland scores ---
        score_groups = {}

        for label, score in pairwise_wins.items():
            score_groups.setdefault(score, []).append(label)
            
        # --- Build the final ranking from highest score to lowest ---
        ordered_labels = []

        for score in sorted(score_groups.keys(), reverse=True):
            tied_labels = score_groups[score]

            if len(tied_labels) == 1:
                ordered_labels.extend(tied_labels)

            else:
                if tie_breaking_strategy == "random":
                    ordered_labels.extend(order_label_tie(row_idx, tied_labels))

                elif tie_breaking_strategy == "confidence":
                    confidence_groups = {}

                    for label in tied_labels:
                        confidence_groups.setdefault(confidence_scores[label], []).append(label)

                    for confidence_score in sorted(confidence_groups.keys(), reverse=True):
                        labels_at_score = confidence_groups[confidence_score]

                        if len(labels_at_score) == 1:
                            ordered_labels.extend(labels_at_score)
                        else:
                            ordered_labels.extend(order_label_tie(row_idx, labels_at_score))

                else:
                    raise ValueError(f"Unsupported tie_breaking_strategy: {tie_breaking_strategy}")
                
        # --- Store the aggregated ranking for this group ---
        final_ranking = " > ".join(ordered_labels)

        row = {}

        for column, value in zip(cols, group_values):
            row[column] = value

        row["converted_ranking"] = final_ranking
        aggregated.append(row)

    return pd.DataFrame(aggregated)

def apply_copeland_voting_per_sentence(df, tie_breaking_strategy):
    return apply_copeland_voting_by_group(df, cols=["row_index"], tie_breaking_strategy=tie_breaking_strategy)

def apply_copeland_voting_over_permutations(df, tie_breaking_strategy):
    return apply_copeland_voting_by_group(df, cols=["row_index", "run_id"], tie_breaking_strategy=tie_breaking_strategy)

# --- Triplet Accuracy ---

def calc_accuracy(df):
    is_correct = df["converted_ranking"] == "best > middle > worst"
    sentence_stats = is_correct.groupby(df["row_index"]).mean()
    overall_accuracy = sentence_stats.mean()

    return overall_accuracy

def calc_top1_accuracy(df):
    top_choice = df["converted_ranking"].str.split(">").str[0].str.strip()
    is_top1_correct = top_choice == "best"
    sentence_stats = is_top1_correct.groupby(df["row_index"]).mean()
    overall_accuracy = sentence_stats.mean()

    return overall_accuracy

# --- Report Triplet Mitigation Results ---

def report_triplet_mitigation_results(base_path, output_path):
    print("Generating report for triplet mitigation results...")
    results = []

    # --- No Confidence ---
    triplet_df = pd.read_csv(f"{base_path}/triplet_ranking_results.csv")
    triplet_df = prepare_triplet_results(triplet_df)
    strategy = "random"
    
    # --- Baseline ---
    method = "Baseline"
    add_mitigation_result(results, "triplet", method, "accuracy", calc_accuracy(triplet_df))
    add_mitigation_result(results, "triplet", method, "top1_accuracy", calc_top1_accuracy(triplet_df))
    
    # --- Majority Voting ---
    method = "Majority Voting per Sentence"
    majority_df = apply_majority_voting_per_sentence(triplet_df, strategy)
    add_mitigation_result(results, "triplet", method, "accuracy", calc_accuracy(majority_df), strategy)
    add_mitigation_result(results, "triplet", method, "top1_accuracy", calc_top1_accuracy(majority_df), strategy)
    
    # --- Copeland Voting ---
    method = "Copeland Voting per Sentence"
    copeland_df = apply_copeland_voting_per_sentence(triplet_df, strategy)
    add_mitigation_result(results, "triplet", method, "accuracy", calc_accuracy(copeland_df), strategy)
    add_mitigation_result(results, "triplet", method, "top1_accuracy", calc_top1_accuracy(copeland_df), strategy)

    # --- Ensemble ---
    method = "Majority Voting over Permutations -> per Sentence"
    stage1 = apply_majority_voting_over_permutations(triplet_df, strategy)
    stage2 = apply_majority_voting_per_sentence(stage1, "random")
    add_mitigation_result(results, "triplet", method, "accuracy", calc_accuracy(stage2), strategy)
    add_mitigation_result(results, "triplet", method, "top1_accuracy", calc_top1_accuracy(stage2), strategy)
    
    method = "Copeland Voting over Permutations -> per Sentence"
    stage1 = apply_copeland_voting_over_permutations(triplet_df, strategy)
    stage2 = apply_copeland_voting_per_sentence(stage1, "random")
    add_mitigation_result(results, "triplet", method, "accuracy", calc_accuracy(stage2), strategy)
    add_mitigation_result(results, "triplet", method, "top1_accuracy", calc_top1_accuracy(stage2), strategy)
    
    method = "Majority Voting over Permutations -> Copeland per Sentence"
    stage1 = apply_majority_voting_over_permutations(triplet_df, strategy)
    stage2 = apply_copeland_voting_per_sentence(stage1, strategy)
    add_mitigation_result(results, "triplet", method, "accuracy", calc_accuracy(stage2), strategy)
    add_mitigation_result(results, "triplet", method, "top1_accuracy", calc_top1_accuracy(stage2), strategy)

    method = "Copeland Voting over Permutations -> Majority per Sentence"
    stage1 = apply_copeland_voting_over_permutations(triplet_df, strategy)
    stage2 = apply_majority_voting_per_sentence(stage1, strategy)
    add_mitigation_result(results, "triplet", method, "accuracy", calc_accuracy(stage2), strategy)
    add_mitigation_result(results, "triplet", method, "top1_accuracy", calc_top1_accuracy(stage2), strategy)

    # --- With Confidence ---
    triplet_df = pd.read_csv(f"{base_path}/triplet_ranking_with_confidence_results.csv")
    triplet_df = prepare_triplet_results(triplet_df)
    
    # --- Baseline ---
    method = "Baseline"
    add_mitigation_result(results, "triplet_with_confidence", method, "accuracy", calc_accuracy(triplet_df))
    add_mitigation_result(results, "triplet_with_confidence", method, "top1_accuracy", calc_top1_accuracy(triplet_df))

    for strategy in ["random", "confidence"]:
        # --- Copeland Voting ---
        method = "Copeland Voting per Sentence"
        copeland_df = apply_copeland_voting_per_sentence(triplet_df, strategy)
        add_mitigation_result(results, "triplet_with_confidence", method, "accuracy", calc_accuracy(copeland_df), strategy)
        add_mitigation_result(results, "triplet_with_confidence", method, "top1_accuracy", calc_top1_accuracy(copeland_df), strategy)
        
        # --- Majority Voting ---
        method = "Majority Voting per Sentence"
        majority_df = apply_majority_voting_per_sentence(triplet_df, strategy)
        add_mitigation_result(results, "triplet_with_confidence", method, "accuracy", calc_accuracy(majority_df), strategy)
        add_mitigation_result(results, "triplet_with_confidence", method, "top1_accuracy", calc_top1_accuracy(majority_df), strategy)

        # --- Ensemble ---
        method = "Majority Voting over Permutations -> per Sentence"
        stage1 = apply_majority_voting_over_permutations(triplet_df, strategy)
        stage2 = apply_majority_voting_per_sentence(stage1, strategy)
        add_mitigation_result(results, "triplet_with_confidence", method, "accuracy", calc_accuracy(stage2), strategy)
        add_mitigation_result(results, "triplet_with_confidence", method, "top1_accuracy", calc_top1_accuracy(stage2), strategy)
        
        method = "Copeland Voting over Permutations -> per Sentence"
        stage1 = apply_copeland_voting_over_permutations(triplet_df, strategy)
        stage2 = apply_copeland_voting_per_sentence(stage1, strategy)
        add_mitigation_result(results, "triplet_with_confidence", method, "accuracy", calc_accuracy(stage2), strategy)
        add_mitigation_result(results, "triplet_with_confidence", method, "top1_accuracy", calc_top1_accuracy(stage2), strategy)
        
        method = "Majority Voting over Permutations -> Copeland per Sentence"
        stage1 = apply_majority_voting_over_permutations(triplet_df, strategy)
        stage2 = apply_copeland_voting_per_sentence(stage1, strategy)
        add_mitigation_result(results, "triplet_with_confidence", method, "accuracy", calc_accuracy(stage2), strategy)
        add_mitigation_result(results, "triplet_with_confidence", method, "top1_accuracy", calc_top1_accuracy(stage2), strategy)

        method = "Copeland Voting over Permutations -> Majority per Sentence"
        stage1 = apply_copeland_voting_over_permutations(triplet_df, strategy)
        stage2 = apply_majority_voting_per_sentence(stage1, strategy)
        add_mitigation_result(results, "triplet_with_confidence", method, "accuracy", calc_accuracy(stage2), strategy)
        add_mitigation_result(results, "triplet_with_confidence", method, "top1_accuracy", calc_top1_accuracy(stage2), strategy)

    pd.DataFrame(results).to_csv(output_path, index=False)

# --- PAIRWISE ---

# --- Pairwise Results Prep ---

def detect_pairwise_response_type(text):
    stripped = str(text).strip().upper()

    if PAIRWISE_ONLY_PATTERN.match(stripped):
        return "winner_only"

    if PAIRWISE_CONF_PATTERN.match(stripped):
        return "winner_with_confidence"

    return None

def parse_pairwise_response(response):
    response_text = str(response).strip().upper()
    winner_part = response_text.split("|")[0].strip()

    match = re.search(r"\b([AB])\b", winner_part)
    if not match:
        raise ValueError(f"Failed to parse valid pairwise response: {response}")

    return match.group(1)

def parse_pairwise_confidence(response):
    response_text = str(response).strip()
    confidence = int(response_text.split("|")[1].strip())
    return confidence

def prepare_pairwise_results(df):
    prepared_rows = []

    response_types = []

    for response in df["response"]:
        response_type = detect_pairwise_response_type(response)
        response_types.append(response_type)

    invalid_count = response_types.count(None)

    if invalid_count > 0:
        raise ValueError(f"Found {invalid_count} invalid pairwise responses.")

    if len(set(response_types)) != 1:
        raise ValueError("Mixed response formats detected.")

    response_type = response_types[0]
    has_confidence = response_type == "winner_with_confidence"

    for _, row in df.iterrows():
        parsed_response = parse_pairwise_response(row["response"])

        if parsed_response == "A":
            winner_label = row["A_type"]
        else:
            winner_label = row["B_type"]

        pair_key = frozenset([row["A_type"], row["B_type"]])

        prepared_row = {
            "run_id": row["run_id"],
            "row_index": row["row_index"],
            "pair_id": row["pair_id"],
            "A_type": row["A_type"],
            "B_type": row["B_type"],
            "pair_key": pair_key,
            "winner_label": winner_label,
        }

        if has_confidence:
            prepared_row["confidence"] = parse_pairwise_confidence(row["response"])

        prepared_rows.append(prepared_row)

    return pd.DataFrame(prepared_rows)

# --- Pairwise Majority Voting ---

def apply_pairwise_majority_voting_by_group(df, cols, tie_breaking_strategy):
    aggregated = []

    for group_values, group in df.groupby(cols):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        # --- Count votes and confidence totals for each winner ---
        winner_counts = {}
        confidence_totals = {}

        for _, row in group.iterrows():
            winner_label = row["winner_label"]
            winner_counts[winner_label] = winner_counts.get(winner_label, 0) + 1

            if "confidence" in group.columns:
                confidence_totals[winner_label] = confidence_totals.get(winner_label, 0) + row["confidence"]

        # --- Find winners with the highest vote count ---
        max_count = max(winner_counts.values())

        tied_winners = []
        for winner_label, count in winner_counts.items():
            if count == max_count:
                tied_winners.append(winner_label)

        # --- Resolve the winning label ---
        if len(tied_winners) == 1:
            winner = tied_winners[0]

        elif tie_breaking_strategy == "random":
            winner = break_label_tie(group_values[0], tied_winners)

        elif tie_breaking_strategy == "confidence":
            if "confidence" not in group.columns:
                winner = break_label_tie(group_values[0], tied_winners)
            else:
                max_confidence = max(confidence_totals[winner_label] for winner_label in tied_winners)

                best_winners = []
                for winner_label in tied_winners:
                    if confidence_totals[winner_label] == max_confidence:
                        best_winners.append(winner_label)

                if len(best_winners) == 1:
                    winner = best_winners[0]
                else:
                    winner = break_label_tie(group_values[0], best_winners)

        else:
            raise ValueError(f"Unsupported tie_breaking_strategy: {tie_breaking_strategy}")
        
        # --- Store the aggregated result row ---
        result_row = {}
        
        for column, value in zip(cols, group_values):
            result_row[column] = value
            
        if "pair_key" not in result_row:
            result_row["pair_key"] = group.iloc[0]["pair_key"]


        result_row["winner_label"] = winner
        result_row["A_type"] = group.iloc[0]["A_type"]
        result_row["B_type"] = group.iloc[0]["B_type"]
        aggregated.append(result_row)

    return pd.DataFrame(aggregated)

def apply_pairwise_majority_voting_over_directions(df, tie_breaking_strategy):
    return apply_pairwise_majority_voting_by_group(df, cols=["row_index", "pair_key", "run_id"], tie_breaking_strategy=tie_breaking_strategy)

def apply_pairwise_majority_voting_per_pair(df, tie_breaking_strategy):
    return apply_pairwise_majority_voting_by_group(df, cols=["row_index", "pair_key"], tie_breaking_strategy=tie_breaking_strategy)

# --- Triplet Reconstruction ---

def reconstruct_triplets_from_pairwise(pairwise_majority_df, method=None):
    # --- Validate that each sentence has all three pairwise comparisons ---
    assert (
        pairwise_majority_df.groupby("row_index").size().eq(3).all()
    ), "Each row_index must have exactly 3 pairwise rows."
    
    assert (
        pairwise_majority_df.groupby("row_index")["pair_key"].nunique().eq(3).all()
    ), "Each row_index must have exactly 3 unique pairwise comparisons."

    reconstructed = []
    sentences_with_cycles = []

    for row_idx, group in pairwise_majority_df.groupby("row_index"):
        label_order = []
        seen = set()

        for _, row in group.iterrows():
            winner = row["winner_label"]
            if winner not in seen:
                seen.add(winner)
                label_order.append(winner)

        for _, row in group.iterrows():
            for label in [row["A_type"], row["B_type"]]:
                if label not in seen:
                    seen.add(label)
                    label_order.append(label)
                    
        # --- Count how many pairwise matchups each label wins ---
        win_counts = {label: 0 for label in label_order}

        for _, row in group.iterrows():
            winner = row["winner_label"]
            if winner in win_counts:
                win_counts[winner] += 1

        # --- Detect cyclic preferences where no strict ranking can be reconstructed ---
        if sorted(win_counts.values()) == [1, 1, 1]:
            sentences_with_cycles.append(row_idx)
            reconstructed.append({"row_index": row_idx,
                                  "converted_ranking": "cycle"})
            continue
        
        # --- Group labels by their number of pairwise wins ---
        score_groups = {}

        for label, score in win_counts.items():
            score_groups.setdefault(score, []).append(label)
            
        # --- Build the final triplet ranking from most wins to fewest wins ---
        ordered_labels = []

        for score in sorted(score_groups.keys(), reverse=True):
            tied_labels = score_groups[score]

            if len(tied_labels) == 1:
                ordered_labels.extend(tied_labels)
            else:
                ordered_labels.extend(order_label_tie(row_idx, tied_labels))
                
        # --- Store the reconstructed ranking for this sentence ---
        converted_ranking = " > ".join(ordered_labels)

        reconstructed.append({"row_index": row_idx,
                              "converted_ranking": converted_ranking})

    cyclic_count = len(sentences_with_cycles)

    if method:
        print(f"\n--- Diagnostics: Sentences with Cycles ({method}) ---")
    else:
        print("\n--- Diagnostics: Sentences with Cycles ---")
    print(f"Sentences with cycles: {cyclic_count}")

    if sentences_with_cycles:
        print("Sentence IDs with cycles:")
        print(sentences_with_cycles, "\n")

    return pd.DataFrame(reconstructed)

# --- Pairwise Accuracy ---

def calc_pairwise_accuracy(df):
    label_rank = {"best": 0, "middle": 1, "worst": 2}
    correct_winners = []

    for _, row in df.iterrows():
        label_a = row["A_type"]
        label_b = row["B_type"]

        if label_rank[label_a] < label_rank[label_b]:
            correct_winner = label_a
        else:
            correct_winner = label_b

        correct_winners.append(correct_winner)

    is_correct = df["winner_label"] == correct_winners
    sentence_stats = is_correct.groupby(df["row_index"]).mean()
    overall_accuracy = sentence_stats.mean()

    return overall_accuracy

# --- Report Pairwise Mitigation Results ---

def report_pairwise_mitigation_results(base_path, output_path):
    print("Generating report for pairwise mitigation results...")
    
    results = []
    
    # --- No Confidence ---
    pairwise_df = pd.read_csv(f"{base_path}/pairwise_ranking_results.csv")
    pairwise_df = prepare_pairwise_results(pairwise_df)
    
    # --- Baseline ---
    method = "Baseline Pairwise"
    add_mitigation_result(results, "pairwise", method, "pairwise_accuracy", calc_pairwise_accuracy(pairwise_df))

    # --- Majority Voting ---
    method = "Pairwise Majority Voting per Pair"
    pairwise_majority_df = apply_pairwise_majority_voting_per_pair(pairwise_df, "random")
    add_mitigation_result(results, "pairwise", method, "pairwise_accuracy", calc_pairwise_accuracy(pairwise_majority_df), "random")
    
    # Triplet Reconstruction
    method = "Pairwise Majority per Pair -> Triplet"
    pairwise_triplet_df = reconstruct_triplets_from_pairwise(pairwise_majority_df, method=method)
    add_mitigation_result(results, "pairwise_to_triplet", method, "accuracy", calc_accuracy(pairwise_triplet_df), "random")
    add_mitigation_result(results, "pairwise_to_triplet", method, "top1_accuracy", calc_top1_accuracy(pairwise_triplet_df), "random")

    # --- Ensemble ---
    method = "Pairwise Majority over Directions -> per Pair"
    stage1 = apply_pairwise_majority_voting_over_directions(pairwise_df, "random")
    stage2 = apply_pairwise_majority_voting_per_pair(stage1, "random")
    add_mitigation_result(results, "pairwise", method, "pairwise_accuracy", calc_pairwise_accuracy(stage2), "random")

    # Triplet Reconstruction
    method = "Pairwise Majority over Directions -> per Pair -> Triplet"
    pairwise_triplet_df = reconstruct_triplets_from_pairwise(stage2, method=method)
    add_mitigation_result(results, "pairwise_to_triplet", method, "accuracy", calc_accuracy(pairwise_triplet_df), "random")
    add_mitigation_result(results, "pairwise_to_triplet", method, "top1_accuracy", calc_top1_accuracy(pairwise_triplet_df), "random")

    # --- With Confidence ---
    pairwise_df = pd.read_csv(f"{base_path}/pairwise_ranking_with_confidence_results.csv")
    pairwise_df = prepare_pairwise_results(pairwise_df)
    
    # --- Baseline ---
    method = "Baseline Pairwise"
    add_mitigation_result(results, "pairwise_with_confidence", method, "pairwise_accuracy", calc_pairwise_accuracy(pairwise_df))

    for strategy in ["random", "confidence"]:
        # --- Majority Voting ---
        method = "Pairwise Majority Voting per Pair"
        pairwise_majority_df = apply_pairwise_majority_voting_per_pair(pairwise_df, strategy)
        add_mitigation_result(results, "pairwise_with_confidence", method, "pairwise_accuracy", calc_pairwise_accuracy(pairwise_majority_df), strategy)

        # Triplet Reconstruction
        method = "Pairwise Majority per Pair -> Triplet"
        pairwise_triplet_df = reconstruct_triplets_from_pairwise(pairwise_majority_df, method=f"{method}, {strategy}")
        add_mitigation_result(results, "pairwise_with_confidence_to_triplet", method, "accuracy", calc_accuracy(pairwise_triplet_df), strategy)
        add_mitigation_result(results, "pairwise_with_confidence_to_triplet", method, "top1_accuracy", calc_top1_accuracy(pairwise_triplet_df), strategy)
        
        # --- Ensemble ---
        method = "Pairwise Majority over Directions -> per Pair"
        stage1 = apply_pairwise_majority_voting_over_directions(pairwise_df, strategy)
        stage2 = apply_pairwise_majority_voting_per_pair(stage1, strategy)
        add_mitigation_result(results, "pairwise_with_confidence", method, "pairwise_accuracy", calc_pairwise_accuracy(stage2), strategy)
        
        # Triplet Reconstruction
        method = "Pairwise Majority over Directions -> per Pair -> Triplet"
        pairwise_triplet_df = reconstruct_triplets_from_pairwise(stage2, method=f"{method}, {strategy}")
        add_mitigation_result(results, "pairwise_with_confidence_to_triplet", method, "accuracy", calc_accuracy(pairwise_triplet_df), strategy)
        add_mitigation_result(results, "pairwise_with_confidence_to_triplet", method, "top1_accuracy", calc_top1_accuracy(pairwise_triplet_df), strategy)

    pd.DataFrame(results).to_csv(output_path, index=False)

def main():
    for backend in ["openai", "llama"]:
        base_path = f"./data/output/ranking/{backend}"
        print(f"\nBackend: {backend}")
        
        # --- Stability Reports ---
        triplet_df = pd.read_csv(f"{base_path}/triplet_ranking_results.csv")
        triplet_df = prepare_triplet_results(triplet_df)
        
        print("Generating stability reports...")
        
        check_correct_ranking_exists(triplet_df)
        report_ranking_variation_across_runs(triplet_df, output_path=f"{base_path}/triplet_ranking_variation_across_runs.txt")
        report_ranking_variation_across_permutations(triplet_df, output_path=f"{base_path}/triplet_ranking_variation_across_permutations.txt")
        compute_run_stability(triplet_df, output_path=f"{base_path}/triplet_ranking_run_stability.txt")
        compute_permutation_stability(triplet_df, output_path=f"{base_path}/triplet_ranking_permutation_stability.txt")
        subsampling_consistency_analysis(triplet_df, output_path=f"{base_path}/triplet_ranking_subsampling_consistency_analysis.txt")
        
        triplet_df = pd.read_csv(f"{base_path}/triplet_ranking_with_confidence_results.csv")
        triplet_df = prepare_triplet_results(triplet_df)
        
        check_correct_ranking_exists(triplet_df)
        report_ranking_variation_across_runs(triplet_df, output_path=f"{base_path}/triplet_ranking_with_confidence_variation_across_runs.txt")
        report_ranking_variation_across_permutations(triplet_df, output_path=f"{base_path}/triplet_ranking_with_confidence_variation_across_permutations.txt")
        compute_run_stability(triplet_df, output_path=f"{base_path}/triplet_ranking_with_confidence_run_stability.txt")
        compute_permutation_stability(triplet_df, output_path=f"{base_path}/triplet_ranking_with_confidence_permutation_stability.txt")
        subsampling_consistency_analysis(triplet_df, output_path=f"{base_path}/triplet_ranking_with_confidence_subsampling_consistency_analysis.txt")
        
        # --- Mitigation Reports ---
        report_triplet_mitigation_results(base_path, output_path=f"{base_path}/triplet_mitigation_results.csv")
        report_pairwise_mitigation_results(base_path, output_path=f"{base_path}/pairwise_mitigation_results.csv")

if __name__ == "__main__":
    main()