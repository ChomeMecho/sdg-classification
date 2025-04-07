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

### Fichiers concernant aurora
server.py : Serveur Flask utilisant un modèle BERT pour prédire la probabilité qu'un texte corresponde à l'ODD 3 ("Bonne santé et bien-être").

server_lime.py : Serveur Flask qui interroge l'API aurora pour classer un texte selon les ODD et utilise LIME pour expliquer les prédictions.

server_shap.py : Serveur Flask similaire à server_lime.py, mais utilisant une approche simulée de SHAP pour expliquer les prédictions.

index_shap.html : Page HTML interactive pour tester le modèle avec des explications SHAP.

index.html : Page HTML pour classifier un texte selon les ODD en utilisant à la fois l'API externe et le modèle local (server.py).

api.html : Page HTML simple pour interroger l'API d'aurora et afficher les prédictions des ODD.
