import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Фикс path для импорта api.py на Windows

from flask.testing import FlaskClient
from flask import json
import pandas as pd
from api import app, engine  # Импорт из api.py

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_upload(client, mocker):
    # Тест POST /upload (mock XLSX — создайте tests/test.xlsx с sample row)
    mocker.patch('pandas.read_sql', return_value=pd.DataFrame())  # Mock для to_sql if needed
    with open('tests/test.xlsx', 'rb') as f:
        response = client.post('/upload', data={'file': (f, 'test.xlsx')})
    assert response.status_code in [200, 400]  # 200 если parsed, 400 если no flights
    data = json.loads(response.data)
    assert 'status' in data or 'error' in data

def test_metrics(client, mocker):
    # Тест /metrics (mock pd.read_sql для agg, повышает покрытие)
    mock_df = pd.DataFrame({'region': ['Test'], 'flight_id': [1], 'duration_min': [60], 'dep_date': ['2025-01-01'], 'dep_time': ['07:05:00'], 'area_km2': [1000]})
    mocker.patch('pandas.read_sql', return_value=mock_df)
    response = client.get('/metrics?year=2025')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert len(data) > 0
    assert data[0]['region'] == 'Test'
    assert data[0]['flight_count'] == 1
    assert data[0]['avg_duration_min'] == 60.0  # Для покрытия mean/agg

def test_graph_top_regions(client, mocker):
    # Тест /graph top (mock pd.read_sql)
    mock_df = pd.DataFrame({'region': ['Test'], 'flight_count': [1]})
    mocker.patch('pandas.read_sql', return_value=mock_df)
    response = client.get('/report/graph?year=2025&type=top_regions&format=png')
    assert response.status_code == 200
    assert 'image/png' in response.headers['Content-Type']

def test_graph_time_series(client, mocker):
    # Тест /graph time (mock pd.read_sql)
    mock_df = pd.DataFrame({'dep_date': ['2025-01-01'], 'flight_count': [1]})
    mocker.patch('pandas.read_sql', return_value=mock_df)
    response = client.get('/report/graph?year=2025&type=time_series&format=png')
    assert response.status_code == 200
    assert 'image/png' in response.headers['Content-Type']

def test_export(client, mocker):
    # Тест /export (mock pd.read_sql)
    mock_flights = pd.DataFrame({'flight_id': ['test1']})
    mock_metrics = pd.DataFrame({'region': ['Test']})
    mocker.patch('pandas.read_sql', side_effect=[mock_flights, mock_metrics])
    response = client.get('/report/export')
    assert response.status_code == 200
    assert 'application/json' in response.headers['Content-Type']
    data = json.loads(response.data)
    assert len(data['flights']) > 0
    assert data['flights'][0]['flight_id'] == 'test1'