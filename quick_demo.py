import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
LORA_DIR = os.getenv("LORA_DIR", "./lora_output/final_adapter")
import random
import torch
import difflib
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------------
# Config
# -----------------------------
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
# LORA_DIR = "./lora_output/final_adapter"  # path to your trained LoRA adapter
EXCLUDE_INDICES_FILE = "./output/finetune_original_indices.txt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAMPLE_INDICES = [2783, 4816]   # specific test indices you care about
N_CM_EXAMPLES = 20             # small subset for quick confusion matrix

# -----------------------------
# DBpedia label set
# -----------------------------
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

# -----------------------------
# Helpers
# -----------------------------
def build_prompt(title, content, label_names):
    content = (content or "")[:600]
    cats = ", ".join(label_names)
    return (
        "You are a classifier for the DBpedia14 dataset.\n"
        "Your job is to assign exactly one category to the article.\n"
        "Answer with only one category name, exactly as written in the list.\n\n"
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

def load_models():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # use `dtype` instead of deprecated `torch_dtype`
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
    ).to(DEVICE)
    base_model.eval()

    lora_base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
    )
    lora_model = PeftModel.from_pretrained(lora_base, LORA_DIR).to(DEVICE)
    lora_model.eval()

    return tokenizer, base_model, lora_model

def run_model_on_indices(model, tokenizer, dataset, indices, batch_size=8):
    examples = [dataset[i] for i in indices]
    prompts = [
        build_prompt(ex["title"], ex["content"], DBPEDIA_LABELS)
        for ex in examples
    ]

    preds, raws = [], []

    for start in range(0, len(prompts), batch_size):
        end = start + batch_size
        batch_prompts = prompts[start:end]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        for i in range(len(batch_prompts)):
            cont_ids = outputs[i][prompt_len:]
            answer_text = tokenizer.decode(cont_ids, skip_special_tokens=True)
            label = normalize_to_label(answer_text, DBPEDIA_LABELS)
            preds.append(label)
            raws.append(answer_text)

    return preds, raws

def load_exclude_indices(path):
    exclude = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        exclude.add(int(line))
                    except ValueError:
                        pass
    return exclude

# -----------------------------
# Parts of the demo
# -----------------------------
def print_two_sample_outputs(tokenizer, base_model, lora_model, dataset, exclude_set):
    print("\n================ TWO SAMPLE OUTPUTS ================\n")

    base_preds, base_raws = run_model_on_indices(
        base_model, tokenizer, dataset, SAMPLE_INDICES, batch_size=2
    )
    lora_preds, lora_raws = run_model_on_indices(
        lora_model, tokenizer, dataset, SAMPLE_INDICES, batch_size=2
    )

    for idx, base_pred, base_raw, lora_pred, lora_raw in zip(
        SAMPLE_INDICES, base_preds, base_raws, lora_preds, lora_raws
    ):
        ex = dataset[idx]
        true_label = DBPEDIA_LABELS[ex["label"]]
        title = ex["title"]
        content_snip = (ex["content"] or "").replace("\n", " ")[:300]

        print(f"--- Example with test index {idx} ---")
        print(f"Title: {title}")
        print(f"True label: {true_label}")
        print(f"Base prediction: {base_pred}")
        print(f"LoRA prediction: {lora_pred}")
        print()
        print("Base raw first line:")
        print(base_raw.splitlines()[0] if base_raw.strip() else "[EMPTY]")
        print()
        print("LoRA raw first line:")
        print(lora_raw.splitlines()[0] if lora_raw.strip() else "[EMPTY]")
        print()
        print("Content snippet:")
        print(content_snip, "...")
        print("----------------------------------------------------\n")

def small_confusion_matrix_demo(tokenizer, base_model, lora_model, dataset, exclude_set):
    print("\n================ SMALL CONFUSION MATRIX DEMO ================\n")

    # candidate indices = all test indices that are NOT in the exclude set
    all_indices = list(range(len(dataset)))
    candidate_indices = [i for i in all_indices if i not in exclude_set]

    # shuffle candidates after exclusion, with fixed seed for reproducibility
    random.seed(123)
    random.shuffle(candidate_indices)

    if len(candidate_indices) < N_CM_EXAMPLES:
        indices = candidate_indices
    else:
        indices = candidate_indices[:N_CM_EXAMPLES]

    print(f"Evaluating on {len(indices)} examples from the DBpedia test split "
          f"(skipping fine-tune indices).")

    y_true = [DBPEDIA_LABELS[dataset[i]["label"]] for i in indices]

    base_preds, _ = run_model_on_indices(base_model, tokenizer, dataset, indices)
    lora_preds, _ = run_model_on_indices(lora_model, tokenizer, dataset, indices)

    base_acc = accuracy_score(y_true, base_preds)
    lora_acc = accuracy_score(y_true, lora_preds)

    print(f"Base accuracy on {len(indices)} examples:  {base_acc:.3f}")
    print(f"LoRA accuracy on {len(indices)} examples:  {lora_acc:.3f}\n")

    base_cm = confusion_matrix(y_true, base_preds, labels=DBPEDIA_LABELS)
    lora_cm = confusion_matrix(y_true, lora_preds, labels=DBPEDIA_LABELS)

    base_cm_df = pd.DataFrame(base_cm, index=DBPEDIA_LABELS, columns=DBPEDIA_LABELS)
    lora_cm_df = pd.DataFrame(lora_cm, index=DBPEDIA_LABELS, columns=DBPEDIA_LABELS)

    print("Base model confusion matrix (small subset):")
    print(base_cm_df)
    print()
    print("LoRA model confusion matrix (small subset):")
    print(lora_cm_df)
    print()

# -----------------------------
# main
# -----------------------------
def main():
    tokenizer, base_model, lora_model = load_models()
    ds_test = load_dataset("fancyzhx/dbpedia_14")["test"]

    exclude_set = load_exclude_indices(EXCLUDE_INDICES_FILE)
    print_two_sample_outputs(tokenizer, base_model, lora_model, ds_test, exclude_set)
    small_confusion_matrix_demo(tokenizer, base_model, lora_model, ds_test, exclude_set)

if __name__ == "__main__":
    main()
