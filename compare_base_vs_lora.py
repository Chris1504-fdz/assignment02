import os
import argparse
import glob
import difflib

import torch
import torch.distributed as dist
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# ---------------------------------
# DBpedia label set
# ---------------------------------
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


def build_prompt(title, content, label_names):
    content = (content or "")[:600]
    cats = ", ".join(label_names)
    return (
        "You are a classifier for the DBpedia14 dataset.\n"
        "Your job is to assign exactly one category to the article.\n"
        "Answer with only the category name, exactly as written in the list.\n\n"
        f"Categories: {cats}\n\n"
        f"Title: {title}\n"
        f"Content: {content}\n"
        "Category:"
    )


def normalize_to_label(answer_text, label_names):
    answer_text = answer_text.strip()
    if not answer_text:
        return "UNKNOWN"

    first_line = answer_text.splitlines()[0].strip()
    token = first_line.split()[0].strip(".,:;! ")

    if token in label_names:
        return token

    for name in label_names:
        if token.lower() == name.lower():
            return name

    match = difflib.get_close_matches(token, label_names, n=1)
    return match[0] if match else "UNKNOWN"


def load_base_model_and_tokenizer(model_id, device):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
    model.eval()
    return model, tokenizer


def run_eval_on_subset(
    model,
    tokenizer,
    subset,
    device,
    rank: int,
    local_rank: int,
    batch_size: int,
    prefix: str,
    output_dir: str,
):
    y_true, y_pred = [], []

    num_samples = len(subset)
    for start in tqdm(
        range(0, num_samples, batch_size),
        desc=f"{prefix} | GPU {local_rank} (rank {rank}) evaluating",
        position=rank,
    ):
        end = min(start + batch_size, num_samples)
        rows = [subset[i] for i in range(start, end)]

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

        prompt_len = inputs["input_ids"].shape[1]

        for b_idx, row in enumerate(rows):
            cont_ids = outputs[b_idx][prompt_len:]
            answer_text = tokenizer.decode(cont_ids, skip_special_tokens=True)
            pred_label = normalize_to_label(answer_text, DBPEDIA_LABELS)

            true_label = DBPEDIA_LABELS[row["label"]]
            y_true.append(true_label)
            y_pred.append(pred_label)

    os.makedirs(output_dir, exist_ok=True)
    partial_path = os.path.join(output_dir, f"{prefix}_partial_rank{rank}.csv")
    pd.DataFrame({"true": y_true, "pred": y_pred}).to_csv(partial_path, index=False)
    print(f"{prefix} | rank {rank}: saved {len(y_true)} rows to {partial_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct"
    )
    parser.add_argument(
        "--lora_dir",
        type=str,
        required=True,
        help="Path to LoRA adapter, e.g. ./lora_output/final_adapter",
    )
    parser.add_argument("--total_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="compare_out")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude_indices_file",
        type=str,
        default=None,
        help="Optional txt with one original index per line to exclude",
    )  # <<< added
    args = parser.parse_args()

    # ------------- distributed -------------
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    using_dist = world_size_env > 1
    if using_dist:
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

    # ------------- dataset -------------
    dataset = load_dataset("fancyzhx/dbpedia_14")
    test_ds = dataset["test"].shuffle(seed=args.seed)

    # load exclusions
    exclude_set = set()
    if args.exclude_indices_file is not None and os.path.exists(args.exclude_indices_file):
        with open(args.exclude_indices_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    exclude_set.add(int(line))
        if rank == 0:
            print(f"Loaded {len(exclude_set)} excluded indices from {args.exclude_indices_file}")

    # Important: the indices in exclude_set must match the split you are evaluating on.
    # Right now we are evaluating on test_ds, whose original indices come from dataset["test"].
    # If your file contains indices from dataset["train"], this will just exclude 0 items.

    # filter test_ds if exclude_set refers to test indices
    if exclude_set:
        # keep items whose original index is not in exclude_set
        # test_ds[i]["__index_level_0__"] may not be present, so we rely on enumerate
        kept = [i for i in range(len(test_ds)) if i not in exclude_set]
        test_ds = test_ds.select(kept)

    total_test = len(test_ds)
    total_samples = min(args.total_samples, total_test)

    # split evenly
    base = total_samples // world_size
    remainder = total_samples % world_size
    if rank < remainder:
        start_idx = rank * (base + 1)
        end_idx = start_idx + (base + 1)
    else:
        start_idx = remainder * (base + 1) + (rank - remainder) * base
        end_idx = start_idx + base

    subset = test_ds.select(range(start_idx, end_idx))

    if rank == 0:
        print(f"Total test available: {total_test}")
        print(f"Evaluating on: {total_samples}")
        print(f"Split: base={base}, remainder={remainder}")
    print(f"Rank {rank}: indices [{start_idx}, {end_idx}) -> {len(subset)} samples")

    # ------------- 1) base -------------
    if rank == 0:
        print("\n===== 1) Evaluating BASE model =====")
    base_model, tokenizer = load_base_model_and_tokenizer(args.model_id, device)
    run_eval_on_subset(
        base_model,
        tokenizer,
        subset,
        device,
        rank,
        local_rank,
        args.batch_size,
        prefix="base",
        output_dir=args.output_dir,
    )

    if using_dist:
        dist.barrier()

    # ------------- 2) lora -------------
    if rank == 0:
        print("\n===== 2) Evaluating LoRA model =====")
    lora_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    lora_model = PeftModel.from_pretrained(lora_model, args.lora_dir)
    lora_model.to(device)
    lora_model.eval()

    run_eval_on_subset(
        lora_model,
        tokenizer,
        subset,
        device,
        rank,
        local_rank,
        args.batch_size,
        prefix="lora",
        output_dir=args.output_dir,
    )

    if using_dist:
        dist.barrier()

    # ------------- 3) merge & write excels (rank 0) -------------
    if rank == 0:
        print("\n===== 3) Merging & metrics =====")
        base_files = sorted(
            glob.glob(os.path.join(args.output_dir, "base_partial_rank*.csv"))
        )
        base_df = pd.concat([pd.read_csv(f) for f in base_files], ignore_index=True)
        base_acc = accuracy_score(base_df["true"], base_df["pred"])
        base_cm = confusion_matrix(
            base_df["true"], base_df["pred"], labels=DBPEDIA_LABELS
        )

        base_report = classification_report(
            base_df["true"],
            base_df["pred"],
            labels=DBPEDIA_LABELS,
            zero_division=0,
            output_dict=True,
        )
        print(f"BASE accuracy: {base_acc:.4f}")
        print(
            classification_report(
                base_df["true"],
                base_df["pred"],
                labels=DBPEDIA_LABELS,
                zero_division=0,
            )
        )

        base_cm_df = pd.DataFrame(
            base_cm, index=DBPEDIA_LABELS, columns=DBPEDIA_LABELS
        )

        base_xlsx = os.path.join(args.output_dir, "base_confusion_matrix.xlsx")
        with pd.ExcelWriter(base_xlsx, engine="openpyxl") as writer:
            base_cm_df.to_excel(writer, sheet_name="confusion_matrix")
            base_metrics_df = pd.DataFrame.from_dict(
                {
                    "accuracy": {"value": base_acc},
                    "macro avg precision": {"value": base_report["macro avg"]["precision"]},
                    "macro avg recall": {"value": base_report["macro avg"]["recall"]},
                    "macro avg f1": {"value": base_report["macro avg"]["f1-score"]},
                    "weighted avg precision": {"value": base_report["weighted avg"]["precision"]},
                    "weighted avg recall": {"value": base_report["weighted avg"]["recall"]},
                    "weighted avg f1": {"value": base_report["weighted avg"]["f1-score"]},
                },
                orient="index",
            )
            base_metrics_df.to_excel(writer, sheet_name="metrics")

        # LORA
        lora_files = sorted(
            glob.glob(os.path.join(args.output_dir, "lora_partial_rank*.csv"))
        )
        lora_df = pd.concat([pd.read_csv(f) for f in lora_files], ignore_index=True)
        lora_acc = accuracy_score(lora_df["true"], lora_df["pred"])
        lora_cm = confusion_matrix(
            lora_df["true"], lora_df["pred"], labels=DBPEDIA_LABELS
        )

        lora_report = classification_report(
            lora_df["true"],
            lora_df["pred"],
            labels=DBPEDIA_LABELS,
            zero_division=0,
            output_dict=True,
        )

        print(f"\nLORA accuracy: {lora_acc:.4f}")
        print(
            classification_report(
                lora_df["true"],
                lora_df["pred"],
                labels=DBPEDIA_LABELS,
                zero_division=0,
            )
        )

        lora_cm_df = pd.DataFrame(
            lora_cm, index=DBPEDIA_LABELS, columns=DBPEDIA_LABELS
        )

        lora_xlsx = os.path.join(args.output_dir, "lora_confusion_matrix.xlsx")
        with pd.ExcelWriter(lora_xlsx, engine="openpyxl") as writer:
            lora_cm_df.to_excel(writer, sheet_name="confusion_matrix")
            lora_metrics_df = pd.DataFrame.from_dict(
                {
                    "accuracy": {"value": lora_acc},
                    "macro avg precision": {"value": lora_report["macro avg"]["precision"]},
                    "macro avg recall": {"value": lora_report["macro avg"]["recall"]},
                    "macro avg f1": {"value": lora_report["macro avg"]["f1-score"]},
                    "weighted avg precision": {"value": lora_report["weighted avg"]["precision"]},
                    "weighted avg recall": {"value": lora_report["weighted avg"]["recall"]},
                    "weighted avg f1": {"value": lora_report["weighted avg"]["f1-score"]},
                },
                orient="index",
            )
            lora_metrics_df.to_excel(writer, sheet_name="metrics")

        print("\nSaved Excel confusion matrices to:")
        print(f" - {base_xlsx}")
        print(f" - {lora_xlsx}")


if __name__ == "__main__":
    main()
