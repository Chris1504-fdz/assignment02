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
