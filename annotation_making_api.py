import json
import requests
from transformers import pipeline
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
import time
from datasets import Dataset

SDG_ORDER = [
    "Pas de pauvreté",
    "Faim zéro",
    "Bonne santé et bien-être",
    "Éducation de qualité",
    "Égalité entre les sexes",
    "Eau propre et assainissement",
    "Énergie propre et d'un coût abordable",
    "Travail décent et croissance économique",
    "Industrie, innovation et infrastructure",
    "Inégalités réduites",
    "Villes et communautés durables",
    "Consommation et production responsables",
    "Mesures relatives à la lutte contre les changements climatiques",
    "Vie aquatique",
    "Vie terrestre",
    "Paix, justice et institutions efficaces",
    "Partenariats pour la réalisation des objectifs"
]


SDG_TRANSLATIONS = {
    "No poverty": "Pas de pauvreté",
    "Zero hunger": "Faim zéro",
    "Good health and well-being": "Bonne santé et bien-être",
    "Quality Education": "Éducation de qualité",
    "Gender equality": "Égalité entre les sexes",
    "Clean water and sanitation": "Eau propre et assainissement",
    "Affordable and clean energy": "Énergie propre et d'un coût abordable",
    "Decent work and economic growth": "Travail décent et croissance économique",
    "Industry, innovation and infrastructure": "Industrie, innovation et infrastructure",
    "Reduced inequalities": "Inégalités réduites",
    "Sustainable cities and communities": "Villes et communautés durables",
    "Responsible consumption and production": "Consommation et production responsables",
    "Climate action": "Mesures relatives à la lutte contre les changements climatiques",
    "Life below water": "Vie aquatique",
    "Life in Land": "Vie terrestre",
    "Peace, Justice and strong institutions": "Paix, justice et institutions efficaces",
    "Partnerships for the goals": "Partenariats pour la réalisation des objectifs"
}

def read_input_texts(filename):
    """Lit le fichier JSON contenant les textes à classifier."""
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

def classify_text(text):
    """Classifie un texte via l'API."""
    if not text.strip():
        return []

    response = requests.post(
        "https://aurora-sdg.labs.vu.nl/classifier/classify/aurora-sdg-multi",
        headers={"Content-Type": "application/json"},
        json={"text": text}
    )

    if response.status_code != 200:
        return []

    return response.json()["predictions"]



def create_binary_vector(predictions, threshold):
    """Crée un vecteur binaire basé sur les prédictions et un seuil."""

    sdg_dict = {sdg: 0 for sdg in SDG_ORDER}
    

    for pred in predictions:
        sdg_name = SDG_TRANSLATIONS.get(pred["sdg"]["name"])
        if sdg_name in sdg_dict and pred["prediction"] >= threshold:
            sdg_dict[sdg_name] = 1
    
 
    return [sdg_dict[sdg] for sdg in SDG_ORDER]

def process_texts(input_filename, threshold=0.05):
    """Traite tous les textes et génère les résultats."""
    input_data = read_input_texts(input_filename)["texts"]
    results = []

    for item in input_data:
        predictions = classify_text(item["text"])
        binary_vector = create_binary_vector(predictions, threshold)
        
        result = {
            "id": item["id"],
            "binary_vector": binary_vector,
            "predictions": [
                {
                    "sdg": SDG_TRANSLATIONS.get(p["sdg"]["name"]),
                    "confidence": f"{(p['prediction'] * 100):.2f}%"
                }
                for p in predictions
            ]
        }
        results.append(result)
    
    return results

def save_results(results, output_filename="results.json"):
    """Sauvegarde les résultats dans un fichier JSON."""
    with open(output_filename, "w", encoding="utf-8") as json_file:
        json.dump(
            {
                "sdg_order": SDG_ORDER,
                "results": results
            },
            json_file,
            ensure_ascii=False,
            indent=4
        )

if __name__ == "__main__":
    input_filename = "input_texts.json"  
    threshold = 0.1
    
    results = process_texts(input_filename, threshold)
    save_results(results)