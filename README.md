# Assignment 2 – LoRA Fine-Tuning with PEFT
## Access Requirement

This project uses the model meta-llama/Llama-3.2-1B-Instruct, which is gated on Hugging Face

To run this project, you must:
- Have a Hugging Face account.

- Visit the model page and click “Request Access” (approval may take a few minutes).

- Generate a personal access token at https://huggingface.co/settings/tokens


- The token should have “Read” permissions.

# 1) Set your Hugging Face token (replace with your own)
export HUGGING_FACE_HUB_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# 2) Build the Docker image (once)
cd assignment02
docker build -t lora-demo .

# 3) Run the project
 docker run --rm -e HUGGING_FACE_HUB_TOKEN=hf_xxx -e LORA_DIR=/app/lora_output/final_adapter -v /data/zhq7531/llm/assignment02_fdz/lora_output:/app/lora_output:ro --shm-size=2g lora-demo
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

```bash
CUDA_VISIBLE_DEVICES=0,2,3 torchrun --nproc_per_node=3 --master_port=29610 compare_base_vs_lora.py --model_id meta-llama/Llama-3.2-1B-Instruct --lora_dir ./lora_output/final_adapter --total_samples 500 --batch_size 16 --output_dir ./compare_out

CUDA_VISIBLE_DEVICES=0,2,3 torchrun --nproc_per_node=3 dbpedia_lora_finetune.py --total_samples 5000 --batch_size 4 --epochs 3 --output_dir ./lora_output

CUDA_VISIBLE_DEVICES=0,2,3 torchrun --nproc_per_node=3 --master_port=29600 dbpedia_llama_eval.py --total_samples 500 --batch_size 16 --output_dir output


