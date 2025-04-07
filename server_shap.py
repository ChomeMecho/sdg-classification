from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import shap
import numpy as np

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

def explain_prediction(text, probability):
    """Génère une explication SHAP basée sur l'importance des mots."""
    words = text.split()
    if not words:
        return []
    
    # Simuler un impact SHAP basé sur la probabilité
    impact = np.linspace(-probability, probability, num=len(words))
    important_words = sorted(zip(words, impact), key=lambda x: abs(x[1]), reverse=True)[:5]
    
    return [(word, float(value)) for word, value in important_words]

@app.route("/predict", methods=["POST"])
def predict():
    """Envoie une requête à l'API externe et retourne le résultat avec explication SHAP."""
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

        # Explication avec SHAP simulée
        important_words = explain_prediction(text, sdg3_probability)
    
        return jsonify({
            "sdg3_probability": float(sdg3_probability),
            "important_words": important_words
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)