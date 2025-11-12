import os
import torch
import argparse
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType


# --------------------------
# Configurable random seed
# --------------------------
def set_random_seed(seed: int = 42):
    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------
# DBpedia label names
# --------------------------
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


# --------------------------
# Prompt builder
# --------------------------
def build_prompt(title, content, label_name=None):
    cats = ", ".join(DBPEDIA_LABELS)
    content = content[:512] if content else ""
    prompt = (
        "You are a classifier for the DBpedia14 dataset.\n"
        f"Categories: {cats}\n\n"
        f"Title: {title}\nContent: {content}\n"
        "Category:"
    )
    if label_name:
        prompt += " " + label_name
    return prompt


# --------------------------
# Dataset preprocessing
# --------------------------
def get_datasets(total_samples=5000, seed=42):
    dataset = load_dataset("fancyzhx/dbpedia_14")

    # shuffle for randomness (seeded for reproducibility)
    dataset = dataset["train"].shuffle(seed=seed)
    total = min(total_samples, len(dataset))
    subset = dataset.select(range(total))

    train_size = int(0.9 * total)
    train_ds = subset.select(range(train_size))
    test_ds = subset.select(range(train_size, total))

    return train_ds, test_ds


def tokenize_function(examples, tokenizer):
    # Convert int label to string name
    labels_text = [DBPEDIA_LABELS[l] for l in examples["label"]]
    prompts = [
        build_prompt(t, c, lbl)
        for t, c, lbl in zip(examples["title"], examples["content"], labels_text)
    ]
    return tokenizer(
        prompts,
        truncation=True,
        padding="max_length",
        max_length=512,
    )


# --------------------------
# Main fine-tuning function
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./lora_output",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
    )
    parser.add_argument(
        "--total_samples",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    # Set random seed
    set_random_seed(args.seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # left padding for decoder-only model + batched generation
    tokenizer.padding_side = "left"

    # Load datasets (randomized)
    train_ds, test_ds = get_datasets(args.total_samples, args.seed)
    print(f"Train size: {len(train_ds)} | Test size: {len(test_ds)}")

    # Quick preview
    example_label = DBPEDIA_LABELS[train_ds[0]["label"]]
    print(
        "Example prompt:\n"
        + build_prompt(
            train_ds[0]["title"], train_ds[0]["content"], example_label
        )[:300]
        + "...\n"
    )

    # Tokenize
    tokenized_train = train_ds.map(
        lambda e: tokenize_function(e, tokenizer),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    tokenized_test = test_ds.map(
        lambda e: tokenize_function(e, tokenizer),
        batched=True,
        remove_columns=test_ds.column_names,
    )

    # Load model (no device_map; Trainer + torchrun will handle devices)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    # Apply LoRA adapter
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # --------------------------
    # TrainingArguments with backwards-compatible eval arg
    # --------------------------
    base_kwargs = dict(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        save_strategy="epoch",
        logging_steps=20,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
        report_to="none",  # no wandb, etc.
    )

    # Try the modern name first, then fall back if your transformers is older
    try:
        training_args = TrainingArguments(
            evaluation_strategy="epoch",
            **base_kwargs,
        )
    except TypeError:
        # Older transformers versions might use a different argument name
        try:
            training_args = TrainingArguments(
                eval_strategy="epoch",
                **base_kwargs,
            )
        except TypeError:
            # Very old version: no per-epoch evaluation, just save per epoch
            training_args = TrainingArguments(**base_kwargs)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train
    trainer.train()

    # Save LoRA adapter
    save_path = os.path.join(args.output_dir, "final_adapter")
    model.save_pretrained(save_path)
    print("\n✅ Fine-tuning completed and saved to:", save_path)


if __name__ == "__main__":
    main()
