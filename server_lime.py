from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import numpy as np
from lime.lime_text import LimeTextExplainer

# Définir l'URL de l'API externe
API_URL = "https://aurora-sdg.labs.vu.nl/classifier/classify/aurora-sdg-multi"

app = Flask(__name__)
CORS(app)

def extract_sdg3_probability(predictions):
    """Extrait la probabilité de l'ODD 3 à partir de la réponse de l'API."""
    for item in predictions:
        if item["sdg"]["code"] == "3":
            return item["prediction"]
    return 0.0

def predict_proba(texts):
    """Envoie une seule requête contenant tous les textes à l'API pour accélérer LIME."""
    probabilities = []
    try:
        responses = [requests.post(API_URL, json={"text": text}).json() for text in texts]
        probabilities = [[1 - extract_sdg3_probability(resp.get("predictions", [])),
                          extract_sdg3_probability(resp.get("predictions", []))] for resp in responses]
    except Exception:
        probabilities = [[0.5, 0.5] for _ in texts]  # Valeur neutre en cas d'erreur
    return np.array(probabilities)

def explain_prediction(text):
    """Utilise LIME pour expliquer la prédiction de l'API plus rapidement."""
    explainer = LimeTextExplainer(class_names=["Not SDG3", "SDG3"])
    exp = explainer.explain_instance(text, predict_proba, num_features=5, num_samples=50)
    return exp.as_list()

@app.route("/predict", methods=["POST"])
def predict():
    """Envoie une requête à l'API externe et retourne le résultat avec explication LIME."""
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "Aucun texte fourni"}), 400

    try:
        # Envoi de la requête à l'API
        response = requests.post(API_URL, json={"text": text})
        response_data = response.json()
        predictions = response_data.get("predictions", [])
        
        # Extraction de la probabilité pour l'ODD "Bonne santé" (SDG3)
        sdg3_probability = extract_sdg3_probability(predictions)

        # Explication avec LIME optimisée
        important_words = explain_prediction(text)
    
        return jsonify({
            "sdg3_probability": float(sdg3_probability),
            "important_words": important_words
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)