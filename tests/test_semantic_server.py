import pytest
from flask import json
from semantic_server import app, model

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_embed_returns_embedding(client):
    text = "test sentence"
    response = client.post('/embed', json={'text': text})
    
    assert response.status_code == 200
    response_data = response.get_json()
    
    embedding = response_data.get('embedding')
    assert embedding is not None
    assert isinstance(embedding, list)
    assert len(embedding) == model.get_sentence_embedding_dimension()

def test_embed_missing_text_returns_error(client):
    response = client.post('/embed', json={})
    
    assert response.status_code == 400
    error_msg = response.get_json().get('error')
    assert error_msg == 'No text provided'

def test_embed_empty_payload(client):
    #Sending raw instead of JSON
    response = client.post('/embed', data='', content_type='application/json')
    assert response.status_code == 400

def test_embedding_is_consistent_for_same_input(client):
    text = "consistency test"
    
    resp1 = client.post('/embed', json={'text': text})
    resp2 = client.post('/embed', json={'text': text})
    
    emb1 = resp1.get_json().get('embedding')
    emb2 = resp2.get_json().get('embedding')
    
    assert emb1 == emb2

def test_different_texts_give_different_embeddings(client):
    first = "first sentence"
    second = "something else entirely"
    
    r1 = client.post('/embed', json={'text': first})
    r2 = client.post('/embed', json={'text': second})
    
    e1 = r1.get_json().get('embedding')
    e2 = r2.get_json().get('embedding')
    
    assert e1 != e2
