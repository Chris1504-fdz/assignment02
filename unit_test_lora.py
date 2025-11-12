# unit_test_lora.py
#
# Tiny LoRA fine-tuning run on a very small subset of DBpedia14.
# Designed to finish in a few minutes and just verify that the
# fine-tuning pipeline works and saves an adapter.

import os
import torch
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
# Same helpers as your main script
# --------------------------
def set_random_seed(seed: int = 42):
    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

def get_datasets(total_samples=200, seed=123):
    # much smaller subset than the main script!
    dataset = load_dataset("fancyzhx/dbpedia_14")
    dataset = dataset["train"].shuffle(seed=seed)
    total = min(total_samples, len(dataset))
    subset = dataset.select(range(total))

    train_size = int(0.9 * total)
    train_ds = subset.select(range(train_size))
    test_ds = subset.select(range(train_size, total))
    return train_ds, test_ds

def tokenize_function(examples, tokenizer):
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
# Mini fine-tuning run
# --------------------------
def main():
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    output_dir = "./lora_output_unit"
    batch_size = 2
    epochs = 1
    learning_rate = 2e-4
    total_samples = 20
    seed = 123

    set_random_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    train_ds, test_ds = get_datasets(total_samples, seed)
    print(f"[UNIT TEST] Train size: {len(train_ds)} | Test size: {len(test_ds)}")

    example_label = DBPEDIA_LABELS[train_ds[0]["label"]]
    print(
        "Example prompt (unit test):\n"
        + build_prompt(train_ds[0]["title"], train_ds[0]["content"], example_label)[:300]
        + "...\n"
    )

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

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

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

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # --------------------------
    # TrainingArguments with backwards-compatible eval arg
    # (copied style from your main script)
    # --------------------------
    base_kwargs = dict(
        output_dir=output_dir,
        overwrite_output_dir=True,
        save_strategy="no",
        # 👇 log more often
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        warmup_ratio=0.0,
        weight_decay=0.0,
        fp16=torch.cuda.is_available(),
        save_total_limit=1,
        report_to="none",   # keeps it printing to stdout
    )

    try:
        # newer transformers
        training_args = TrainingArguments(
            evaluation_strategy="no",
            **base_kwargs,
        )
    except TypeError:
        try:
            # some older versions use eval_strategy
            training_args = TrainingArguments(
                eval_strategy="no",
                **base_kwargs,
            )
        except TypeError:
            # very old version: just don't set an eval schedule
            training_args = TrainingArguments(**base_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    save_path = os.path.join(output_dir, "final_adapter")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(save_path)
    print("\n✅ UNIT TEST: fine-tuning completed and saved to:", save_path)

if __name__ == "__main__":
    main()
