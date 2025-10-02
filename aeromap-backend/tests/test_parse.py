import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Фикс path для импорта api.py на Windows

from datetime import datetime, time, date
from shapely.geometry import Point
import geopandas as gpd
import pandas as pd
from api import parse_flight_row, parse_coords, parse_time, parse_date, calculate_duration, get_region

def test_parse_coords():
    # Валидные форматы из вашего Excel
    assert parse_coords("5957N02905E") == (59.95, 29.083333333333332)
    assert parse_coords("4408N04308E") == (44.13333333333333, 43.13333333333333)
    assert parse_coords("5152N08600E") == (51.86666666666667, 86.0)
    # С секундами
    assert parse_coords("621900N0343500E") == (62.31666666666667, 34.583333333333336)
    # Инвалид
    assert parse_coords("invalid") == (None, None)
    assert parse_coords("") == (None, None)
    assert parse_coords("9999N99999E") == (None, None)  # Валидация диапазонов

def test_parse_time():
    assert parse_time("0705") == time(7, 5)
    assert parse_time("1647") == time(16, 47)
    assert parse_time("invalid") == None
    assert parse_time("") == None
    assert parse_time("2460") == None  # Инвалид минуты

def test_parse_date():
    assert parse_date("250201") == date(2025, 2, 1)
    assert parse_date("250124") == date(2025, 1, 24)
    assert parse_date("250731") == date(2025, 7, 31)
    assert parse_date("invalid") == None

def test_calculate_duration():
    dep_date = date(2025, 2, 1)
    dep_time = time(7, 5)
    arr_date = date(2025, 2, 1)
    arr_time = time(8, 5)
    assert calculate_duration(dep_date, dep_time, arr_date, arr_time) == 60.0  # 1 час
    # Кросс-день
    assert calculate_duration(dep_date, time(23, 0), date(2025, 2, 2), time(1, 0)) == 120.0
    assert calculate_duration(None, None, None, None) == None

def test_parse_flight_row():
    # Пример row из вашего 2025.xlsx (с запятой после центра, как в логе)
    row = "Санкт-Петербургский,(SHR-ZZZZZ -ZZZZ0705 -K0300M3000 -DEP/5957N02905E DOF/250201 OPR/МАЛИНОВСКИЙ НИКИТА АЛЕКСАНДРОВИ4 +79313215153 TYP/SHAR RMK/ОБОЛО4КА 300 ДЛЯ ЗОНДИРОВАНИЯ АТМОСФЕРЫ SID/7772187998),-TITLE IDEP -SID 7772187998 -ADD 250201 -ATD 0705 -ADEP ZZZZ -ADEPZ 5957N02905E -PAP 0"
    parsed = parse_flight_row(row)
    assert parsed['center'] == "Санкт-Петербургский,"  # Фикс: С запятой, как в вашем parsed (если хотите чистый, добавьте .strip(',') в api.py для center = center_match.group(1).strip().strip(','))
    assert parsed['flight_id'] == "7772187998"
    assert parsed['type'] == "SHAR"
    assert parsed['dep_lat'] == 59.95
    assert parsed['dep_lon'] == 29.083333333333332
    assert parsed['dep_date'] == date(2025, 2, 1)
    assert parsed['dep_time'] == time(7, 5)
    assert parsed['duration_min'] == 0.0  # Нет arr_time
    # С duration
    row_with_arr = row + ",-TITLE IARR -ADA 250201 -ATA 0805 -ADARR ZZZZ -ADARRZ 5957N02905E"
    parsed_arr = parse_flight_row(row_with_arr)
    assert parsed_arr['duration_min'] == 60.0
    assert parse_flight_row("invalid row") == None

@pytest.fixture
def mock_gdf():
    # Mock GeoDataFrame для теста (простой полигон вокруг точки из примера)
    data = {'name_ru': ['Ленинградская область'], 'geometry': [Point(29.083333333333332, 59.95).buffer(0.1)]}  # Буфер для contains
    return gpd.GeoDataFrame(data, crs="EPSG:4326")

def test_get_region(mock_gdf):
    assert get_region(59.95, 29.083333333333332, mock_gdf) == 'Ленинградская область'  # Внутри
    assert get_region(0, 0, mock_gdf) == 'Unknown'  # Снаружи
    assert get_region(None, None, mock_gdf) == 'Unknown'