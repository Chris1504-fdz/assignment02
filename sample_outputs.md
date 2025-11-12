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


