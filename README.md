# Assignment 2 – LoRA Fine-Tuning with PEFT
## Access Requirement

This project uses the model meta-llama/Llama-3.2-1B-Instruct, which is gated on Hugging Face

To run this project, you must:
- Have a Hugging Face account.

- Visit the model page and click “Request Access” (approval may take a few minutes).

- Generate a personal access token at https://huggingface.co/settings/tokens


- The token should have “Read” permissions.

# 1) Set your Hugging Face token (by requesting to the Huggin Face provided link for "meta" model

# 2) Build the Docker image (once)
cd assignment02

docker build -t lora-demo .

# 3) Run the project

docker run --rm -e HUGGING_FACE_HUB_TOKEN=hf_xxx -e LORA_DIR=/app/lora_output/final_adapter --shm-size=2g lora-demo
 
## Overview

This project fine-tunes a pretrained language model using LoRA (Low-Rank Adaptation) through the PEFT
 library (Hugging Face).
The model learns to classify Wikipedia-style articles from the DBpedia 14 dataset into 14 categories.
All experiments were performed on Vision GPUs.

### Data and Splits
- **Dataset:** `fancyzhx/dbpedia_14` (Hugging Face Datasets), 14 classes.
- **Fine-tune set size:** **5,000** samples drawn from the dataset’s **train** split (seeded shuffle).
- **Evaluation subset size (compare run):** **500** samples from the **test** split, evenly sharded across 3 ranks then merged.
- **Leakage prevention:** Test evaluation **explicitly excludes** the exact train indices used for fine-tuning.
  - Exclusion list file: `./output/finetune_original_indices.txt`
  - Both `compare_base_vs_lora.py` and `quick_demo.py` filter out any test indices present in this file before sampling.


### Models
- **Base model:** `meta-llama/Llama-3.2-1B-Instruct` (≈1.2B params)
- **LoRA adapter:** saved to `./lora_output/final_adapter`

### Inference / Evaluation Settings

- **Prompt template:** single-turn classifier prompt listing all 14 categories and asking for exactly one category token.
- **Label post-processing:** exact match, case-insensitive match, to the 14 labels.
- **Metrics:** `accuracy`, `confusion_matrix` (scikit-learn). Separate **macro** and **weighted** precision/recall/F1 are computed and exported.





# Base vs LoRA – Confusion Matrices and Metrics

## Metrics (side-by-side)

|                        |   Base |   LoRA |
|:-----------------------|-------:|-------:|
| accuracy               |  0.642 |  0.972 |
| macro avg precision    |  0.738 |  0.972 |
| macro avg recall       |  0.643 |  0.972 |
| macro avg f1           |  0.644 |  0.972 |
| weighted avg precision |  0.716 |  0.973 |
| weighted avg recall    |  0.642 |  0.972 |
| weighted avg f1        |  0.632 |  0.972 |

## Base Model – Confusion Matrix

| True \ Pred            |   Company |   EducationalInstitution |   Artist |   Athlete |   OfficeHolder |   MeanOfTransportation |   Building |   NaturalPlace |   Village |   Animal |   Plant |   Album |   Film |   WrittenWork |
|:-----------------------|----------:|-------------------------:|---------:|----------:|---------------:|-----------------------:|-----------:|---------------:|----------:|---------:|--------:|--------:|-------:|--------------:|
| Company                |        42 |                        1 |        3 |         0 |              0 |                      0 |          0 |              0 |         0 |        0 |       0 |       0 |      0 |             0 |
| EducationalInstitution |         3 |                       31 |        2 |         0 |              0 |                      0 |          2 |              0 |         0 |        0 |       0 |       0 |      1 |             0 |
| Artist                 |         0 |                        0 |       34 |         0 |              0 |                      0 |          0 |              0 |         0 |        0 |       0 |       0 |      0 |             0 |
| Athlete                |         0 |                        0 |        7 |        24 |              0 |                      0 |          0 |              0 |         0 |        0 |       0 |       0 |      0 |             0 |
| OfficeHolder           |         0 |                        0 |       19 |         1 |              2 |                      0 |          0 |              0 |         0 |        0 |       0 |       0 |      0 |             0 |
| MeanOfTransportation   |        11 |                        0 |        3 |         0 |              0 |                      0 |          0 |              1 |         0 |        0 |       0 |       0 |      1 |             0 |
| Building               |         0 |                        1 |        1 |         0 |              0 |                      0 |         22 |              1 |         0 |        0 |       0 |       0 |      0 |             0 |
| NaturalPlace           |         0 |                        0 |        0 |         0 |              0 |                      0 |          0 |             21 |         0 |        0 |       0 |       0 |      0 |             0 |
| Village                |         0 |                        0 |        0 |         0 |              0 |                      0 |          0 |              0 |        24 |        0 |       0 |       0 |      0 |             0 |
| Animal                 |         0 |                        0 |        0 |         0 |              0 |                      0 |          0 |              3 |         0 |       26 |       2 |       0 |      0 |             0 |
| Plant                  |         0 |                        0 |        1 |         0 |              0 |                      0 |          0 |              1 |         0 |        0 |      34 |       0 |      0 |             0 |
| Album                  |         0 |                        0 |       11 |         0 |              0 |                      0 |          0 |              0 |         0 |        0 |       0 |      16 |      0 |             0 |
| Film                   |         0 |                        0 |        3 |         0 |              0 |                      0 |          0 |              0 |         0 |        0 |       0 |       0 |     45 |             0 |
| WrittenWork            |         4 |                        0 |        6 |         0 |              0 |                      0 |          0 |              0 |         0 |        0 |       0 |       0 |     16 |             0 |

## LoRA Model – Confusion Matrix

| True \ Pred            |   Company |   EducationalInstitution |   Artist |   Athlete |   OfficeHolder |   MeanOfTransportation |   Building |   NaturalPlace |   Village |   Animal |   Plant |   Album |   Film |   WrittenWork |
|:-----------------------|----------:|-------------------------:|---------:|----------:|---------------:|-----------------------:|-----------:|---------------:|----------:|---------:|--------:|--------:|-------:|--------------:|
| Company                |        46 |                        1 |        0 |         0 |              0 |                      1 |          0 |              0 |         0 |        0 |       0 |       0 |      0 |             0 |
| EducationalInstitution |         0 |                       41 |        0 |         0 |              0 |                      0 |          2 |              0 |         0 |        0 |       0 |       0 |      0 |             0 |
| Artist                 |         0 |                        0 |       35 |         4 |              0 |                      0 |          0 |              0 |         0 |        0 |       0 |       0 |      0 |             0 |
| Athlete                |         0 |                        0 |        0 |        31 |              0 |                      0 |          0 |              0 |         0 |        0 |       0 |       0 |      0 |             0 |
| OfficeHolder           |         0 |                        0 |        1 |         0 |             36 |                      0 |          0 |              0 |         0 |        0 |       0 |       0 |      0 |             0 |
| MeanOfTransportation   |         1 |                        0 |        0 |         0 |              0 |                     29 |          0 |              0 |         0 |        0 |       0 |       0 |      0 |             0 |
| Building               |         0 |                        1 |        0 |         0 |              0 |                      0 |         26 |              2 |         0 |        0 |       0 |       0 |      0 |             0 |
| NaturalPlace           |         0 |                        0 |        0 |         0 |              0 |                      0 |          0 |             32 |         0 |        0 |       0 |       0 |      0 |             0 |
| Village                |         0 |                        0 |        0 |         0 |              0 |                      0 |          0 |              0 |        24 |        0 |       0 |       0 |      0 |             0 |
| Animal                 |         0 |                        0 |        0 |         0 |              0 |                      0 |          0 |              0 |         0 |       31 |       0 |       0 |      0 |             1 |
| Plant                  |         0 |                        0 |        0 |         0 |              0 |                      0 |          0 |              0 |         0 |        0 |      36 |       0 |      0 |             0 |
| Album                  |         0 |                        0 |        0 |         0 |              0 |                      0 |          0 |              0 |         0 |        0 |       0 |      27 |      0 |             0 |
| Film                   |         0 |                        0 |        0 |         0 |              0 |                      0 |          0 |              0 |         0 |        0 |       0 |       0 |     48 |             0 |
| WrittenWork            |         0 |                        0 |        0 |         0 |              0 |                      0 |          0 |              0 |         0 |        0 |       0 |       0 |      0 |            44 |


### EXTRA: Reproducible Commands

For documentation purpuses I am assing the lines of code used to train the model, the model was trained using the cluster "vision" at Northwestern

```bash
CUDA_VISIBLE_DEVICES=0,2,3 torchrun --nproc_per_node=3 --master_port=29610 compare_base_vs_lora.py --model_id meta-llama/Llama-3.2-1B-Instruct --lora_dir ./lora_output/final_adapter --total_samples 500 --batch_size 16 --output_dir ./compare_out

CUDA_VISIBLE_DEVICES=0,2,3 torchrun --nproc_per_node=3 dbpedia_lora_finetune.py --total_samples 5000 --batch_size 4 --epochs 3 --output_dir ./lora_output

CUDA_VISIBLE_DEVICES=0,2,3 torchrun --nproc_per_node=3 --master_port=29600 dbpedia_llama_eval.py --total_samples 500 --batch_size 16 --output_dir output
```

The commands above capture the exact workflow used to fine-tune and evaluate the model for this assignment. The LoRA adapter was fine-tuned specifically for DBpedia14 entity classification, where the goal is to map short Wikipedia-style articles into one of 14 predefined categories (such as Company, Artist, Film, etc.). All training and evaluation was performed on the Northwestern vision GPU cluster using 3 GPUs in parallel.

## LoRA fine-tuning.
The model meta-llama/Llama-3.2-1B-Instruct was trained on 5,000 examples from the DBpedia training split for 3 epochs. Only a small set of low-rank LoRA parameters was updated, allowing the model to specialize toward DBpedia’s categories while keeping the base weights frozen. This produced the adapter saved under ./lora_output/final_adapter.

## Base vs LoRA comparison.
After fine-tuning, the base model and the LoRA-adapted model were evaluated side-by-side on 500 held-out test examples. This command generates the confusion matrices and accuracy comparison shown in the report.

## Final evaluation of the LoRA model.
The LoRA adapter was then evaluated on a separate 500-example subset of DBpedia test data to produce the detailed metrics, macro/weighted scores, and sample predictions included in the analysis section.

All three commands were run with torchrun using distributed data parallel (DDP) with GPUs 0, 2, and 3, ensuring consistent reproducibility and efficient multi-GPU training. These commands allow anyone with access to the model and cluster to reproduce the fine-tuning, evaluation, and metrics reported in this project.

# Sample Outputs and Discussion – DBpedia14 Classification

## 1. Task and Models

**Task.**  
Classify short Wikipedia-style articles into one of 14 DBpedia14 categories:

> Company, EducationalInstitution, Artist, Athlete, OfficeHolder,  
> MeanOfTransportation, Building, NaturalPlace, Village, Animal,  
> Plant, Album, Film, WrittenWork.

**Base model**

- `meta-llama/Llama-3.2-1B-Instruct` (≤ 1.7B parameters).
- Used as-is (no fine-tuning).

**Fine-tuned model (LoRA)**

- Same base model with a LoRA adapter trained (using PEFT) on the **train** split of `fancyzhx/dbpedia_14`.
- Fine-tuning uses **only** train data.  
- All examples in this document come from the DBpedia **test** split.
- For the small confusion-matrix demo, indices used in fine-tuning (stored in `output/finetune_original_indices.txt`) are explicitly **excluded** before sampling test examples.

Both models are prompted with the same instruction-style prompt and their continuations are mapped back to one of the 14 DBpedia labels.

---

## 2. Prompt Used for Both Models

text
You are a classifier for the DBpedia14 dataset.
Your job is to assign exactly one category to the article.
Answer with only one category name, exactly as written in the list.

Categories: Company, EducationalInstitution, Artist, Athlete,
OfficeHolder, MeanOfTransportation, Building, NaturalPlace,
Village, Animal, Plant, Album, Film, WrittenWork

Title: (TITLE)

Content: (CONTENT)

Category:


## 3. Sample Outputs from `quick_demo.py

================ TWO SAMPLE OUTPUTS ================

--- Example with test index 2783 ---

Title: Great-West Lifeco

True label: Company

Base prediction: 
- Company

LoRA prediction: 
- Company

Content snippet:

 Great-West Lifeco is an insurance centered financial holding company (corporation) that operates in North America (USA and Canada) Europe and Asia through 5 wholly owned regionally focused subsidiaries. Many of the companies it has indirect control over are part of its largest subsidiary The Great- ...

### Discussion:
This article clearly describes a financial holding company. Words like “insurance”, “financial holding company”, “subsidiaries” strongly point to the Company label. Both the base and LoRA models classify it correctly, showing that the base model already handles many straightforward entity descriptions well and that LoRA fine-tuning does not harm performance on these easy cases.


----------------------------------------------------
### Example 2
--- Example with test index 4816 ---

Title: Elephant Stone Records

True label: 
- Company

Base prediction: 
- Artist

LoRA prediction: 
- Company

Content snippet:
 Elephant Stone Records is a US record label formed in 2002 in Los Angeles by music critic and former Dionysus Records publicist Ben Szporluk (Ben Vendetta) and his artist wife Arabella Proffer (Bella Vendetta). Named after a song by The Stone Roses the label focuses on new groups and reissue compil ...

### Discussion:
This entry describes a record label, which DBpedia categorizes as a Company, even though it is connected to music and artists. The base model focuses on the music context (“record label”, “artist wife”) and outputs Artist, which is semantically related but incorrect for the DBpedia14 label set.

fter LoRA fine-tuning on DBpedia14:
* The model has seen many examples where record labels belong to the Company class.
* It learns that phrases like “record label formed in 2002”, “US record label”, etc., typically map to Company, not Artist.
* As a result, the LoRA model answers Company, correctly aligning with the dataset’s label definition.

## 4. Small Confusion-Matrix Demo (30 Examples)

Evaluating on 30 examples from the DBpedia test split (skipping fine-tune indices).

Base accuracy on 30 examples:  
- 0.567

LoRA accuracy on 30 examples:  
- 1.000

# Confusion Matrices (Small Subset)

- **Subset size**: 100 examples (excluding fine-tune indices)
- **Base accuracy**: `0.610`
- **LoRA accuracy**: `0.990`

## Base Model Confusion Matrix

|                        |   Company |   EducationalInstitution |   Artist |   Athlete |   OfficeHolder |   MeanOfTransportation |   Building |   NaturalPlace |   Village |   Animal |   Plant |   Album |   Film |   WrittenWork |
|:-----------------------|----------:|-------------------------:|---------:|----------:|---------------:|-----------------------:|-----------:|---------------:|----------:|---------:|--------:|--------:|-------:|--------------:|
| Company                |         6 |                        0 |        0 |         0 |              0 |                      0 |          0 |              0 |         0 |        0 |       0 |       0 |      0 |             0 |
| EducationalInstitution |         1 |                        2 |        0 |         0 |              0 |                      0 |          0 |              0 |         0 |        0 |       0 |       0 |      0 |             0 |
| Artist                 |         0 |                        0 |        5 |         0 |              0 |                      0 |          0 |              0 |         0 |        0 |       0 |       0 |      0 |             0 |
| Athlete                |         0 |                        0 |        0 |         8 |              0 |                      0 |          0 |              0 |         0 |        0 |       0 |       0 |      0 |             0 |
| OfficeHolder           |         0 |                        0 |        2 |         0 |              2 |                      0 |          0 |              0 |         0 |        0 |       0 |       0 |      0 |             0 |
| MeanOfTransportation   |         2 |                        0 |        1 |         0 |              0 |                      0 |          0 |              0 |         0 |        0 |       0 |       0 |      1 |             0 |
| Building               |         0 |                        0 |        0 |         0 |              0 |                      0 |          4 |              0 |         0 |        0 |       0 |       0 |      0 |             0 |
| NaturalPlace           |         0 |                        0 |        0 |         0 |              0 |                      0 |          0 |              6 |         0 |        0 |       1 |       0 |      0 |             0 |
| Village                |         0 |                        0 |        0 |         0 |              0 |                      0 |          0 |              0 |         6 |        0 |       0 |       0 |      0 |             0 |
| Animal                 |         0 |                        0 |        0 |         0 |              0 |                      0 |          0 |              0 |         0 |        5 |       1 |       0 |      0 |             0 |
| Plant                  |         0 |                        0 |        0 |         0 |              0 |                      0 |          0 |              0 |         0 |        0 |       8 |       0 |      0 |             0 |
| Album                  |         0 |                        0 |        2 |         0 |              0 |                      0 |          0 |              0 |         0 |        0 |       0 |       5 |      0 |             0 |
| Film                   |         0 |                        0 |        0 |         0 |              0 |                      0 |          0 |              0 |         0 |        0 |       0 |       0 |      4 |             0 |
| WrittenWork            |         0 |                        0 |        1 |         0 |              0 |                      0 |          0 |              0 |         0 |        0 |       0 |       0 |      3 |             0 |

## LoRA Model Confusion Matrix

|                        |   Company |   EducationalInstitution |   Artist |   Athlete |   OfficeHolder |   MeanOfTransportation |   Building |   NaturalPlace |   Village |   Animal |   Plant |   Album |   Film |   WrittenWork |
|:-----------------------|----------:|-------------------------:|---------:|----------:|---------------:|-----------------------:|-----------:|---------------:|----------:|---------:|--------:|--------:|-------:|--------------:|
| Company                |         6 |                        0 |        0 |         0 |              0 |                      0 |          1 |              0 |         0 |        0 |       0 |       0 |      0 |             0 |
| EducationalInstitution |         0 |                        3 |        0 |         0 |              0 |                      0 |          0 |              0 |         0 |        0 |       0 |       0 |      0 |             0 |
| Artist                 |         0 |                        0 |        5 |         0 |              0 |                      0 |          0 |              0 |         0 |        0 |       0 |       0 |      0 |             0 |
| Athlete                |         0 |                        0 |        0 |         8 |              0 |                      0 |          0 |              0 |         0 |        0 |       0 |       0 |      0 |             0 |
| OfficeHolder           |         0 |                        0 |        0 |         0 |             11 |                      0 |          0 |              0 |         0 |        0 |       0 |       0 |      0 |             0 |
| MeanOfTransportation   |         0 |                        0 |        0 |         0 |              0 |                      9 |          0 |              0 |         0 |        0 |       0 |       0 |      0 |             0 |
| Building               |         0 |                        0 |        0 |         0 |              0 |                      0 |          6 |              0 |         0 |        0 |       0 |       0 |      0 |             0 |
| NaturalPlace           |         0 |                        0 |        0 |         0 |              0 |                      0 |          0 |              9 |         0 |        0 |       0 |       0 |      0 |             0 |
| Village                |         0 |                        0 |        0 |         0 |              0 |                      0 |          0 |              0 |         6 |        0 |       0 |       0 |      0 |             0 |
| Animal                 |         0 |                        0 |        0 |         0 |              0 |                      0 |          0 |              0 |         0 |        7 |       0 |       0 |      0 |             0 |
| Plant                  |         0 |                        0 |        0 |         0 |              0 |                      0 |          0 |              0 |         0 |        0 |       8 |       0 |      0 |             0 |
| Album                  |         0 |                        0 |        0 |         0 |              0 |                      0 |          0 |              0 |         0 |        0 |       0 |       8 |      0 |             0 |
| Film                   |         0 |                        0 |        0 |         0 |              0 |                      0 |          0 |              0 |         0 |        0 |       0 |       0 |      4 |             0 |
| WrittenWork            |         0 |                        0 |        0 |         0 |              0 |                      0 |          0 |              0 |         0 |        0 |       0 |       0 |      0 |             9 |



