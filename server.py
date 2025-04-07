from flask import Flask, request, jsonify
from flask_cors import CORS  # Ajout de CORS
import tensorflow as tf
from transformers import BertTokenizer
import numpy as np
from transformers import TFBertModel


# Charger le modèle
MODEL_PATH = "sdg3.h5" 
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"TFBertMainLayer": TFBertModel})

# Charger le tokenizer BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

app = Flask(__name__)
CORS(app)  # Autorise toutes les origines

def preprocess_text(text, max_len=512):
    """Prépare le texte pour le modèle BERT."""
    encoded = tokenizer.encode_plus(text, add_special_tokens=True, max_length=max_len, truncation=True, padding='max_length', return_tensors='tf')
    return np.array(encoded["input_ids"]), np.array(encoded["attention_mask"])

@app.route('/predict_sdg3', methods=['POST'])
def predict_sdg3():
    """Retourne la probabilité de l’ODD "Bonne santé"."""
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "Aucun texte fourni"}), 400

    input_ids, attention_mask = preprocess_text(text)
    prediction = model.predict([input_ids, attention_mask])[0][0]
    return jsonify({"sdg3_probability": float(prediction)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
