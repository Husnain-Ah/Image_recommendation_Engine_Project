from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2') #this model is used for semantic searching to get the meaning behind the highest predicted
#keyword and finding the closest match in the dataset.

@app.route('/embed', methods=['POST']) # Endpoint for embedding text
def embed():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']
    embedding = model.encode(text).tolist()
    return jsonify({'embedding': embedding})

if __name__ == '__main__':
    app.run(port=5001)
