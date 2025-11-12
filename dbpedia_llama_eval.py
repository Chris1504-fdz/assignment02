import os
import argparse
import difflib
import glob

import torch
import torch.distributed as dist
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


# 14 labels of DBpedia_14
DBPEDIA_LABELS = [
    "Company",
    "EducationalInstitution",
    "Artist",
    "Athlete",
    "OfficeHolder",
    "MeanOfTransportation",
    "Building",
    "NaturalPlace",
    "Village",
    "Animal",
    "Plant",
    "Album",
    "Film",
    "WrittenWork",
]


# ---------- prompt & normalization helpers ----------

def build_prompt(title, content, label_names):
    """
    Build a simple instruction-style prompt for classification.
    """
    content = content[:600]  # truncate long content
    cats = ", ".join(label_names)
    prompt = (
        "You are a classifier for the DBpedia14 dataset.\n"
        "Your job is to assign exactly one category to the article.\n"
        "Answer with only the category name, exactly as written in the list.\n\n"
        f"Categories: {cats}\n\n"
        f"Title: {title}\n"
        f"Content: {content}\n"
        "Category:"
    )
    return prompt


def normalize_to_label(answer_text, label_names):
    """
    Take the raw continuation text and map it to one of the 14 labels.
    """
    answer_text = answer_text.strip()
    if not answer_text:
        return "UNKNOWN"

    # Take only the first line
    first_line = answer_text.splitlines()[0].strip()

    # Take only the first token (handles "Company." or "Company category")
    token = first_line.split()[0] if first_line else ""
    token = token.strip(".,:;! ").strip()

    # Exact match
    if token in label_names:
        return token

    # Case-insensitive match
    for name in label_names:
        if token.lower() == name.lower():
            return name

    # Fuzzy match (best-effort)
    match = difflib.get_close_matches(token, label_names, n=1)
    if match:
        return match[0]

    return "UNKNOWN"


# ---------- model loader ----------

def load_model_and_tokenizer(model_id, device):
    """
    Load Llama model and tokenizer on a specific device.
    Sets left padding to avoid decoder-only warnings.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # make sure we have a pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # IMPORTANT: left padding for decoder-only models
    tokenizer.padding_side = "left"

    if torch.cuda.is_available():
        # prefer bfloat16 if supported, otherwise float16
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
    )
    model.to(device)
    model.eval()

    return model, tokenizer


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Hugging Face model id",
    )
    parser.add_argument(
        "--total_samples",
        type=int,
        default=100000,
        help="Total number of test samples to evaluate (across all ranks)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size per GPU",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save partial CSVs and confusion matrix",
    )
    args = parser.parse_args()

    # ----- Distributed setup -----
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    using_distributed = world_size_env > 1

    if using_distributed:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if rank == 0:
        print(f"Using distributed: {using_distributed}, world_size={world_size}")
        print(f"Evaluating model: {args.model_id}")
        print(f"Total requested samples: {args.total_samples}")
        print(f"Batch size: {args.batch_size}")
        print(f"Output dir: {args.output_dir}")

    # ----- Load dataset -----
    dataset = load_dataset("fancyzhx/dbpedia_14")
    total_test = len(dataset["test"])
    total_samples = min(args.total_samples, total_test)
    dataset_shuffled = dataset["test"].shuffle(seed=42)

    if rank == 0:
        print(f"DBpedia test set size: {total_test}")
        print(f"Evaluating on {total_samples} samples in total.")

    per_rank = total_samples // world_size
    start_idx = local_rank * per_rank
    end_idx = start_idx + per_rank
    subset = dataset_shuffled.select(range(start_idx, end_idx))

    print(f"Rank {local_rank}: processing indices [{start_idx}, {end_idx}) -> {len(subset)} samples")

    # if rank == 0:
    #     print(f"Per-rank sample counts (base={base}, remainder={remainder}):")
    if using_distributed:
        # Let each rank print its info
        print(f"Rank {rank}: processing indices [{start_idx}, {end_idx}) -> {len(subset)} samples")

    # ----- Load model & tokenizer -----
    model, tokenizer = load_model_and_tokenizer(args.model_id, device)

    # ----- Evaluation loop (batched) -----
    y_true, y_pred = [], []

    num_samples = len(subset)
    batch_size = args.batch_size

    torch.set_grad_enabled(False)

    for start in tqdm(
        range(0, num_samples, batch_size),
        desc=f"GPU {local_rank} (rank {rank}) evaluating",
        position=rank,
    ):
        end = min(start + batch_size, num_samples)
        rows = [subset[i] for i in range(start, end)]

        # build prompts batch
        prompts = [
            build_prompt(row["title"], row["content"], DBPEDIA_LABELS)
            for row in rows
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # All sequences in the batch have the same prompt length
        prompt_len = inputs["input_ids"].shape[1]

        for b_idx, row in enumerate(rows):
            cont_ids = outputs[b_idx][prompt_len:]
            answer_text = tokenizer.decode(cont_ids, skip_special_tokens=True)
            pred_label = normalize_to_label(answer_text, DBPEDIA_LABELS)

            true_label = DBPEDIA_LABELS[row["label"]]
            y_true.append(true_label)
            y_pred.append(pred_label)

    # ----- Save partial results -----
    os.makedirs(args.output_dir, exist_ok=True)
    partial_path = os.path.join(args.output_dir, f"partial_rank{rank}.csv")

    df_local = pd.DataFrame({"true": y_true, "pred": y_pred})
    df_local.to_csv(partial_path, index=False)
    print(f"Rank {rank}: saved partial results to {partial_path}")

    # ----- Sync ranks -----
    if using_distributed:
        dist.barrier()

    # ----- Rank 0 merges & computes metrics -----
    if rank == 0:
        print("Rank 0: merging partial results and computing metrics...")

        partial_files = sorted(
            glob.glob(os.path.join(args.output_dir, "partial_rank*.csv"))
        )
        if not partial_files:
            print("No partial files found. Did the other ranks finish correctly?")
            return

        dfs = [pd.read_csv(f) for f in partial_files]
        df_all = pd.concat(dfs, ignore_index=True)

        acc = accuracy_score(df_all["true"], df_all["pred"])
        macro_f1 = f1_score(df_all["true"], df_all["pred"], average="macro")

        print(f"\nTotal evaluated samples: {len(df_all)}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Macro F1: {macro_f1:.4f}\n")

        print("Classification report:")
        print(classification_report(df_all["true"], df_all["pred"],
                                   labels=DBPEDIA_LABELS, zero_division=0))

        cm = confusion_matrix(df_all["true"], df_all["pred"], labels=DBPEDIA_LABELS)
        cm_df = pd.DataFrame(cm, index=DBPEDIA_LABELS, columns=DBPEDIA_LABELS)

        cm_path = os.path.join(args.output_dir, "confusion_matrix.xlsx")
        cm_df.to_excel(cm_path)
        print(f"\nConfusion matrix saved to {cm_path}")


if __name__ == "__main__":
    main()
