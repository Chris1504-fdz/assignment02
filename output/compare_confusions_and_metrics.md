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
