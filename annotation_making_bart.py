import json
import nltk
import time
from transformers import pipeline
from nltk.tokenize import sent_tokenize

# Téléchargement du tokenizer NLTK
nltk.download('punkt')

# Initialisation du modèle de classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Liste des ODD
SDG_LABELS = [
    "Pauvreté",
    "Faim",
    "Bonne santé",
    "Éducation",
    "Égalité entre les sexes",
    "Eau propre",
    "Énergie coût abordable",
    "Travail et croissance économique",
    "Industrie, innovation et infrastructure",
    "Inégalités réduites",
    "Villes durables",
    "Consommation et production responsables",
    "changement climatique",
    "Vie aquatique",
    "Vie terrestre",
    "Paix, justice et institutions efficaces",
    "Partenariats pour la réalisation des objectifs"
]

# Seuil minimal pour considérer une prédiction
THRESHOLD = 0.7


def read_input_texts(filename):
    """Lit un fichier JSON contenant les textes à classifier."""
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)


def classify_text(text):
    """Classifie un texte en utilisant le modèle zero-shot."""
    sentences = sent_tokenize(text, language="french")
    results = [classifier(sentence, SDG_LABELS, multi_label=True) for sentence in sentences]
    return results


def aggregate_predictions(results):
    """Agrège les scores des prédictions et génère un vecteur binaire."""
    agg_scores = {label: 0.0 for label in SDG_LABELS}
    
    for res in results:
        for label, score in zip(res["labels"], res["scores"]):
            if score >= THRESHOLD:
                agg_scores[label] += score
    
    binary_vector = [1 if agg_scores[label] >= 1.0 else 0 for label in SDG_LABELS]
    
    return agg_scores, binary_vector


def process_texts(input_filename, output_filename):
    """Traite tous les textes et génère le fichier JSON de sortie."""
    input_data = read_input_texts(input_filename)["texts"]
    results = []
    
    for item in input_data:
        start_time = time.time()
        predictions = classify_text(item["text"])
        agg_scores, binary_vector = aggregate_predictions(predictions)
        end_time = time.time()
        
        result = {
            "id": item["id"],
            "binary_vector": binary_vector,
            "aggregated_scores": {label: round(score, 3) for label, score in agg_scores.items()},
            "inference_time": round(end_time - start_time, 2)
        }
        results.append(result)
    
    with open(output_filename, "w", encoding="utf-8") as json_file:
        json.dump({"sdg_labels": SDG_LABELS, "results": results}, json_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    input_filename = "input_texts.json"
    output_filename = "results_bart_0_7.json"
    process_texts(input_filename, output_filename)
