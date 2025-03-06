# sdg-classification

Use machine learning to classify the UN Sustainable Development Goals (SDGs) based on text data.

## Data

The data is a collection of text data from UPPA (Université de Pau et des Pays de l'Adour) and is not available for public use. The data is in French and contains summaries of thesis papers done by students at UPPA. The data is not labeled with the SDGs, so the goal is to use machine learning to classify the data into the 17 SDGs.

## Methodology

We used several machine learning models to classify the data into the 17 SDGs. The models we used are:
- BART large MNLI (Facebook)
- Aurora SDG mBERT (Aurora)
- 17 SDG mBERT (Aurora)


### References
https://huggingface.co/facebook/bart-large-mnli
https://aurora-universities.eu/sdg-research/classify/
https://zenodo.org/records/6487606

