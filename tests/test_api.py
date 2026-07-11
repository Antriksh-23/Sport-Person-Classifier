import pytest
import sys
import os
import numpy as np
from unittest.mock import patch, MagicMock

# Add api path for import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'api'))
import app

@pytest.fixture
def client():
    app.app.config['TESTING'] = True
    with app.app.test_client() as client:
        yield client

def test_home_route(client):
    response = client.get('/')
    assert response.status_code == 200

@patch('predict_utils.classify_image')
def test_predict_route_no_data(mock_classify, client):
    response = client.post('/predict', json={})
    assert response.status_code == 400
    assert "error" in response.get_json()

@patch('predict_utils.classify_image')
def test_predict_route_success(mock_classify, client):
    mock_classify.return_value = {
        "athlete": "Roger Federer",
        "confidence": 98.5,
        "probabilities": {"Roger Federer": 98.5}
    }
    response = client.post('/predict', json={"image_data": "base64string"})
    assert response.status_code == 200
    data = response.get_json()
    assert data["athlete"] == "Roger Federer"
