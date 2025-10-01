import six  # Для совместимости
import re
from datetime import datetime, timedelta
import pandas as pd
from multiprocessing import Pool
import geopandas as gpd
from shapely.geometry import Point
from fiona import Env
from sqlalchemy import create_engine, text
from geoalchemy2 import Geometry
from flask import Flask, jsonify, send_file, make_response, request
from flask_swagger_ui import get_swaggerui_blueprint
# from flask_oidc import OpenIDConnect  # Закомментировано для dev
import matplotlib.pyplot as plt
import io
import json
import logging
from logging.handlers import RotatingFileHandler
import os

app = Flask(__name__)

# Настройка логирования
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

# DB_URL из env (для Docker) или fallback localhost
DB_URL = os.getenv('DB_URL', 'postgresql://aviation_user:aviation_pass@localhost:5432/aviation_db')
engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=3600)
SHAPEFILE_PATH = 'shapefiles/RF.shp'

# Swagger UI
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(SWAGGER_URL, API_URL, config={'app_name': "БПЛА Анализ"})
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# Функции парсинга (без изменений)
def parse_coords(coord_str):
    if not coord_str:
        return None, None
    coord_str = str(coord_str).upper().replace(' ', '').replace('С', 'N').replace('В', 'E').replace('C', 'N')
    try:
        match = re.match(r'(\d{2})(\d{2})([NS])(\d{3})(\d{2})([EW])', coord_str)
        if match:
            lat_deg = int(match.group(1))
            lat_min = int(match.group(2))
            lat_dir = match.group(3)
            lon_deg = int(match.group(4))
            lon_min = int(match.group(5))
            lon_dir = match.group(6)
            if not (0 <= lat_deg <= 90 and 0 <= lat_min < 60 and 0 <= lon_deg <= 180 and 0 <= lon_min < 60):
                return None, None  # Валидация диапазонов
            lat = lat_deg + lat_min / 60.0
            if lat_dir == 'S':
                lat = -lat
            lon = lon_deg + lon_min / 60.0
            if lon_dir == 'W':
                lon = -lon
            return lat, lon
        match_ss = re.match(r'(\d{2})(\d{2})(\d{2})([NS])(\d{3})(\d{2})(\d{2})([EW])', coord_str)
        if match_ss:
            lat_deg = int(match_ss.group(1))
            lat_min = int(match_ss.group(2))
            lat_sec = int(match_ss.group(3))
            lat_dir = match_ss.group(4)
            lon_deg = int(match_ss.group(5))
            lon_min = int(match_ss.group(6))
            lon_sec = int(match_ss.group(7))
            lon_dir = match_ss.group(8)
            if not (0 <= lat_deg <= 90 and 0 <= lat_min < 60 and 0 <= lat_sec < 60 and 0 <= lon_deg <= 180 and 0 <= lon_min < 60 and 0 <= lon_sec < 60):
                return None, None
            lat = lat_deg + lat_min / 60.0 + lat_sec / 3600.0
            if lat_dir == 'S':
                lat = -lat
            lon = lon_deg + lon_min / 60.0 + lon_sec / 3600.0
            if lon_dir == 'W':
                lon = -lon
            return lat, lon
        if 'ZONA' in coord_str:
            coord_match = re.search(r'(\d+[NS]\d+[EW])', coord_str)
            if coord_match:
                return parse_coords(coord_match.group(1))
    except ValueError:
        pass
    return None, None

def parse_time(time_str):
    time_str = str(time_str).strip()
    if len(time_str) >= 4 and time_str[:4].isdigit():
        try:
            hh, mm = int(time_str[:2]), int(time_str[2:4])
            if 0 <= hh < 24 and 0 <= mm < 60:
                return datetime.strptime(time_str[:4], '%H%M').time()
        except ValueError:
            pass
    return None

def parse_date(date_str):
    date_str = str(date_str).strip()
    if len(date_str) == 6 and date_str.isdigit():
        try:
            year = 2000 + int(date_str[0:2])
            month = int(date_str[2:4])
            day = int(date_str[4:6])
            return datetime(year, month, day).date()
        except ValueError:
            pass
    return None

def calculate_duration(dep_date, dep_time, arr_date, arr_time):
    if dep_date and dep_time and arr_date and arr_time:
        dep_dt = datetime.combine(dep_date, dep_time)
        arr_dt = datetime.combine(arr_date, arr_time)
        if arr_dt < dep_dt:
            arr_dt += timedelta(days=1)
        return (arr_dt - dep_dt).total_seconds() / 60
    return None

def parse_flight_row(row_str):
    if not row_str.strip():
        return None
    try:
        row_str = re.sub(r'row\d+:\s*', '', row_str).strip().replace('\n', ' ')
        center_match = re.match(r'(.*?)\s*\(', row_str)
        if center_match:
            center = center_match.group(1).strip()
            shr_full = row_str[center_match.end()-1:]
        else:
            center = ''
            shr_full = row_str
        sections = re.split(r'\),', shr_full)
        shr_section = next((s.strip() for s in sections if '(SHR' in s), '')
        if not shr_section:
            return None
        sid_match = re.search(r'SID/(\d+)', shr_section)
        flight_id = sid_match.group(1) if sid_match else None
        typ_match = re.search(r'TYP/([\w/]+)', shr_section)
        flight_type = typ_match.group(1) if typ_match else None
        dof_match = re.search(r'DOF/(\d{6})', shr_section)
        dof_date = parse_date(dof_match.group(1)) if dof_match else None
        dep_coord_str = ''
        dep_match = re.search(r'DEP/([\dNS EWСВ\d]+)', shr_section)
        if dep_match:
            dep_coord_str = dep_match.group(1)
        else:
            zona_match = re.search(r'/ZONA\s*(.+?)/', shr_section)
            if zona_match:
                dep_coord_str = zona_match.group(1)
        dep_lat, dep_lon = parse_coords(dep_coord_str)
        arr_coord_str = ''
        dest_match = re.search(r'DEST/([\dNS EWСВ\d]+)', shr_section)
        if dest_match:
            arr_coord_str = dest_match.group(1)
        else:
            arr_coord_str = dep_coord_str
        arr_lat, arr_lon = parse_coords(arr_coord_str)
        time_matches = re.findall(r'-ZZZZ(\d{4})', shr_section)
        plan_dep_time = parse_time(time_matches[0]) if len(time_matches) > 0 else None
        plan_arr_time = parse_time(time_matches[1]) if len(time_matches) > 1 else None
        dep_date = dof_date
        dep_time = plan_dep_time
        arr_date = dof_date
        arr_time = plan_arr_time
        for section in sections:
            section = section.strip()
            if '-TITLE IDEP' in section:
                add_match = re.search(r'-ADD (\d{6})', section)
                if add_match:
                    dep_date = parse_date(add_match.group(1))
                atd_match = re.search(r'-ATD (\d{4})', section)
                if atd_match:
                    dep_time = parse_time(atd_match.group(1))
                adepz_match = re.search(r'-ADEPZ ([\dNS EWСВ\d]+)', section)
                if adepz_match:
                    dep_lat, dep_lon = parse_coords(adepz_match.group(1))
            if '-TITLE IARR' in section:
                ada_match = re.search(r'-ADA (\d{6})', section)
                if ada_match:
                    arr_date = parse_date(ada_match.group(1))
                ata_match = re.search(r'-ATA (\d{4})', section)
                if ata_match:
                    arr_time = parse_time(ata_match.group(1))
                adarrz_match = re.search(r'-ADARRZ ([\dNS EWсВ\d]+)', section)
                if adarrz_match:
                    arr_lat, arr_lon = parse_coords(adarrz_match.group(1))
        duration = calculate_duration(dep_date, dep_time, arr_date, arr_time)
        if flight_id is None or dep_lat is None or dep_lon is None:
            app.logger.warning(f"Invalid flight data skipped: {row_str[:100]}...")
            return None
        if duration is None:
            duration = 0  # Default 0 if no arr_time
        return {
            'center': center,
            'flight_id': flight_id,
            'type': flight_type,
            'dep_lat': dep_lat,
            'dep_lon': dep_lon,
            'arr_lat': arr_lat,
            'arr_lon': arr_lon,
            'dep_date': dep_date,
            'dep_time': dep_time,
            'arr_date': arr_date,
            'arr_time': arr_time,
            'duration_min': duration
        }
    except Exception as e:
        app.logger.error(f"Error parsing row: {str(e)} - Row: {row_str[:100]}...")
        return None

def get_region(lat, lon, gdf):
    if pd.isna(lat) or pd.isna(lon):
        return 'Unknown'
    point = Point(lon, lat)
    matching = gdf[gdf.geometry.contains(point)]
    if not matching.empty:
        region = matching['name_ru'].iloc[0]
        app.logger.info(f"Found region for {lat}, {lon}: {region}")
        return region
    app.logger.warning(f"No region found for {lat}, {lon}")
    return 'Unknown'

# Новый эндпоинт для /api/regions/flights (без изменений)
@app.route('/api/regions/flights', methods=['GET'])
def get_regions_flights():
    try:
        from_str = request.args.get('from')
        to_str = request.args.get('to')
        metric = request.args.get('metric', 'count')
        base_query = "SELECT region, flight_id, duration_min FROM flights"
        where_clauses = []
        params = []
        if from_str:
            where_clauses.append("dep_date >= %s")
            params.append(from_str)
        if to_str:
            where_clauses.append("dep_date <= %s")
            params.append(to_str)
        if where_clauses:
            base_query += " WHERE " + " AND ".join(where_clauses)
        base_query += ";"
        with engine.connect() as conn:
            df = pd.read_sql(base_query, conn, params=tuple(params))
        if df.empty:
            return jsonify([])
        if metric == 'count':
            agg_df = df.groupby('region').size().reset_index(name='value')
        elif metric == 'avg_duration':
            agg_df = df.groupby('region')['duration_min'].mean().reset_index(name='value')
        else:
            return jsonify({"error": "Invalid metric (supported: count, avg_duration)"}), 400
        agg_df = agg_df.sort_values(by='value', ascending=False)
        agg_df['name'] = agg_df['region']
        response = jsonify(agg_df[['name', 'value']].to_dict(orient='records'))
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response
    except Exception as e:
        app.logger.error(f"Error in regions/flights: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    try:
        year = request.args.get('year')
        month = request.args.get('month')
        region = request.args.get('region')
        base_query = """
            SELECT f.region, f.flight_id, f.duration_min, f.dep_date, f.dep_time, m.area_km2
            FROM flights f
            JOIN metrics m ON f.region = m.region
        """
        where_parts = []
        params = {}
        if year:
            where_parts.append("f.dep_date::text LIKE :year")
            params['year'] = f"{year}%"
        if month:
            where_parts.append("f.dep_date::text LIKE :month")
            params['month'] = f"%-{month}-%"
        if region:
            where_parts.append("f.region = :region")
            params['region'] = region
        if where_parts:
            base_query += " WHERE " + " AND ".join(where_parts)
        base_query += ";"
        with engine.connect() as conn:
            df = pd.read_sql(text(base_query), conn, params=params)
        if df.empty:
            return jsonify({"error": "No metrics available"}), 404
        # Рассчитай метрики в Pandas (проще и стабильнее)
        metrics_df = df.groupby('region').agg(
            flight_count=('flight_id', 'count'),
            avg_duration_min=('duration_min', 'mean'),
            total_duration_min=('duration_min', 'sum'),
            area_km2=('area_km2', 'max')
        ).reset_index()
        metrics_df['flight_density'] = metrics_df['flight_count'] / metrics_df['area_km2']
        # Peak hourly (groupby hour from dep_time)
        df['dep_datetime'] = pd.to_datetime(df['dep_date'].astype(str) + ' ' + df['dep_time'].astype(str))
        df['hour'] = df['dep_datetime'].dt.floor('H')
        metrics_df['peak_load_hourly'] = df.groupby(['region', 'hour'])['flight_id'].count().groupby('region').max().values
        # Daily stats
        daily_counts = df.groupby(['region', 'dep_date'])['flight_id'].count().reset_index(name='daily_count')
        metrics_df = metrics_df.merge(daily_counts.groupby('region')['daily_count'].agg(['mean', 'median']).reset_index(), on='region')
        metrics_df = metrics_df.rename(columns={'mean': 'avg_daily_flights', 'median': 'median_daily_flights'})
        metrics_df = metrics_df.sort_values(by='flight_count', ascending=False)
        # Growth (separate conn)
        if month:
            prev_month = int(month) - 1 if int(month) > 1 else 12
            prev_year = year if prev_month != 12 else str(int(year) - 1)
            prev_params = {'prev_year': f"{prev_year}%", 'prev_month': f"%-{str(prev_month).zfill(2)}-%"}
            prev_base = """
                SELECT f.region, f.flight_id, f.duration_min, f.dep_date, f.dep_time, m.area_km2
                FROM flights f
                JOIN metrics m ON f.region = m.region
                WHERE f.dep_date::text LIKE :prev_year AND f.dep_date::text LIKE :prev_month;
            """
            with engine.connect() as conn:
                prev_df = pd.read_sql(text(prev_base), conn, params=prev_params)
            if not prev_df.empty:
                prev_counts = prev_df.groupby('region')['flight_id'].count().reset_index(name='flight_count_prev')
                metrics_df = metrics_df.merge(prev_counts, on='region', how='left')
                metrics_df['growth_percent'] = ((metrics_df['flight_count'] - metrics_df['flight_count_prev'].fillna(0)) / metrics_df['flight_count_prev'].fillna(1)) * 100
        # Hourly distribution (separate, with fix for NaN)
        hourly_query = "SELECT EXTRACT(HOUR FROM dep_time) as hour, COUNT(*) as count FROM flights WHERE dep_time IS NOT NULL GROUP BY hour;"
        with engine.connect() as conn:
            hourly_df = pd.read_sql(hourly_query, conn)
        metrics_df['hourly_distribution'] = [hourly_df.to_dict(orient='records')] * len(metrics_df)
        # Zero days (fixed for empty)
        zero_days_query = """
            SELECT m.region, COUNT(*) as zero_days
            FROM metrics m
            LEFT JOIN (
                SELECT generate_series(
                    COALESCE(MIN(dep_date), CURRENT_DATE),
                    COALESCE(MAX(dep_date), CURRENT_DATE),
                    '1 day'::interval
                ) as day
                FROM flights
                WHERE dep_date IS NOT NULL
            ) days ON TRUE
            LEFT JOIN flights f ON days.day = f.dep_date AND f.region = m.region
            WHERE f.flight_id IS NULL
            GROUP BY m.region;
        """
        with engine.connect() as conn:
            zero_df = pd.read_sql(zero_days_query, conn)
        metrics_df = metrics_df.merge(zero_df, on='region', how='left').fillna({'zero_days': 0})
        response = jsonify(metrics_df.to_dict(orient='records'))
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response
    except Exception as e:
        app.logger.error(f"Error in metrics: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/report/graph', methods=['GET'])
def get_graph():
    try:
        year = request.args.get('year')
        type_graph = request.args.get('type', 'top_regions')  # Новый параметр для типа графика
        format_type = request.args.get('format', 'png')
        if type_graph == 'top_regions':
            query = "SELECT region, flight_count FROM metrics ORDER BY flight_count DESC LIMIT 10;"
            params = {}
            if year:
                query = """
                    SELECT region, COUNT(flight_id) as flight_count
                    FROM flights
                    WHERE dep_date::text LIKE :year
                    GROUP BY region
                    ORDER BY flight_count DESC LIMIT 10;
                """
                params = {'year': f"{year}%"}
            with engine.connect() as conn:
                metrics_df = pd.read_sql(text(query), conn, params=params)
            if metrics_df.empty:
                return jsonify({"error": "No data for graph"}), 404
            plt.figure(figsize=(10, 6))
            plt.bar(metrics_df['region'], metrics_df['flight_count'])
            plt.title('Топ регионов по количеству полетов БПЛА')
            plt.xlabel('Регион')
            plt.ylabel('Количество полетов')
            plt.xticks(rotation=45)
            plt.tight_layout()
        elif type_graph == 'time_series':
            query = """
                SELECT dep_date, COUNT(flight_id) as flight_count
                FROM flights
                GROUP BY dep_date
                ORDER BY dep_date;
            """
            params = {}
            if year:
                query = """
                    SELECT dep_date, COUNT(flight_id) as flight_count
                    FROM flights
                    WHERE dep_date::text LIKE :year
                    GROUP BY dep_date
                    ORDER BY dep_date;
                """
                params = {'year': f"{year}%"}
            with engine.connect() as conn:
                ts_df = pd.read_sql(text(query), conn, params=params)
            if ts_df.empty:
                return jsonify({"error": "No data for time series"}), 404
            plt.figure(figsize=(12, 6))
            plt.plot(ts_df['dep_date'], ts_df['flight_count'], marker='o')
            plt.title('Временной ряд полетов БПЛА')
            plt.xlabel('Дата')
            plt.ylabel('Количество полетов')
            plt.grid(True)
            plt.tight_layout()
        else:
            return jsonify({"error": "Invalid graph type"}), 400
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format=format_type)
        img_buf.seek(0)
        mimetype = 'image/png' if format_type == 'png' else 'image/jpeg'
        response = make_response(send_file(img_buf, mimetype=mimetype))
        response.headers['Content-Disposition'] = f'attachment; filename=flight_graph_{type_graph}.{format_type}'
        return response
    except Exception as e:
        app.logger.error(f"Error in graph: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_data():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files['file']
        if not file.filename.endswith('.xlsx'):
            return jsonify({"error": "Only XLSX files supported"}), 400
        df_excel = pd.read_excel(file, sheet_name='Result_1', header=None)
        if df_excel.shape[1] > 1:
            rows = df_excel.apply(lambda x: ' '.join(x.fillna('').astype(str).str.strip().str.replace('nan', '').str.replace('NaN', '')), axis=1).tolist()
            rows = [re.sub(r'\s+', ' ', r).strip() for r in rows if r.strip()]
        else:
            rows = df_excel[0].fillna('').astype(str).str.strip().tolist()
            rows = [r for r in rows if r]
        with Pool(processes=4) as pool:
            parsed_flights = pool.map(parse_flight_row, rows)
        parsed_flights = [p for p in parsed_flights if p]
        if not parsed_flights:
            return jsonify({"error": "No valid flights parsed"}), 400
        df = pd.DataFrame(parsed_flights).drop_duplicates(subset=['flight_id', 'dep_date'])
        with Env(SHAPE_RESTORE_SHX='YES'):
            gdf = gpd.read_file(SHAPEFILE_PATH)
        if gdf.crs != 'EPSG:4326':
            gdf = gdf.to_crs('EPSG:4326')
        df['region'] = df.apply(lambda row: get_region(row['dep_lat'], row['dep_lon'], gdf), axis=1)
        df['dep_geom'] = df.apply(lambda row: f"SRID=4326;POINT({row['dep_lon']} {row['dep_lat']})" if pd.notna(row['dep_lat']) else None, axis=1)
        df.to_sql('flights', engine, if_exists='append', index=False, method='multi', dtype={'dep_geom': Geometry('POINT', srid=4326)})
        # Пересчет метрик (расширенные)
        metrics = df.groupby('region').agg({
            'flight_id': 'count',
            'duration_min': ['mean', 'sum']
        }).reset_index()
        metrics.columns = ['region', 'flight_count', 'avg_duration_min', 'total_duration_min']
        gdf_area = gdf.to_crs('EPSG:3395')
        gdf_area['area_km2'] = gdf_area.geometry.area / 10**6
        metrics = metrics.merge(gdf_area[['name_ru', 'area_km2']], left_on='region', right_on='name_ru', how='left')
        metrics['flight_density'] = metrics['flight_count'] / metrics['area_km2']
        metrics.to_sql('metrics', engine, if_exists='replace', index=False)
        app.logger.info(f"Processed {len(df)} flights")
        return jsonify({"status": "Data uploaded, parsed and metrics updated", "processed_count": len(df)}), 200
    except Exception as e:
        app.logger.error(f"Error in upload: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.json
        if not data or 'flights' not in data:
            return jsonify({"error": "Invalid payload"}), 400
        # Предполагаем, что payload - JSON с массивом raw строк полетов
        rows = data['flights']
        with Pool(processes=4) as pool:
            parsed_flights = pool.map(parse_flight_row, rows)
        parsed_flights = [p for p in parsed_flights if p]
        df = pd.DataFrame(parsed_flights).drop_duplicates(subset=['flight_id', 'dep_date'])
        # ... (аналогично upload: геопривязка, сохранение, метрики)
        # Для простоты, повтор кода из upload, но в prod рефакторить в функцию
        with Env(SHAPE_RESTORE_SHX='YES'):
            gdf = gpd.read_file(SHAPEFILE_PATH)
        if gdf.crs != 'EPSG:4326':
            gdf = gdf.to_crs('EPSG:4326')
        df['region'] = df.apply(lambda row: get_region(row['dep_lat'], row['dep_lon'], gdf), axis=1)
        df['dep_geom'] = df.apply(lambda row: f"SRID=4326;POINT({row['dep_lon']} {row['dep_lat']})" if pd.notna(row['dep_lat']) else None, axis=1)
        df.to_sql('flights', engine, if_exists='append', index=False, method='multi', dtype={'dep_geom': Geometry('POINT', srid=4326)})
        # Пересчет метрик (как выше)
        metrics = df.groupby('region').agg({
            'flight_id': 'count',
            'duration_min': ['mean', 'sum']
        }).reset_index()
        metrics.columns = ['region', 'flight_count', 'avg_duration_min', 'total_duration_min']
        gdf_area = gdf.to_crs('EPSG:3395')
        gdf_area['area_km2'] = gdf_area.geometry.area / 10**6
        metrics = metrics.merge(gdf_area[['name_ru', 'area_km2']], left_on='region', right_on='name_ru', how='left')
        metrics['flight_density'] = metrics['flight_count'] / metrics['area_km2']
        metrics.to_sql('metrics', engine, if_exists='replace', index=False)
        return jsonify({"status": "Webhook processed", "processed_count": len(df)}), 200
    except Exception as e:
        app.logger.error(f"Error in webhook: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Эндпоинт для экспорта полного отчета JSON
@app.route('/report/export', methods=['GET'])
def export_report():
    try:
        with engine.connect() as conn:
            flights_df = pd.read_sql("SELECT * FROM flights;", conn)
            metrics_df = pd.read_sql("SELECT * FROM metrics;", conn)
        # Фикс: convert date to str, time to 'HH:MM:SS'
        flights_df['dep_date'] = flights_df['dep_date'].astype(str)
        flights_df['arr_date'] = flights_df['arr_date'].astype(str)
        flights_df['dep_time'] = flights_df['dep_time'].apply(lambda x: x.strftime('%H:%M:%S') if pd.notnull(x) else None)
        flights_df['arr_time'] = flights_df['arr_time'].apply(lambda x: x.strftime('%H:%M:%S') if pd.notnull(x) else None)
        report = {
            "flights": flights_df.to_dict(orient='records'),
            "metrics": metrics_df.to_dict(orient='records')
        }
        # Фикс escaping: Используем json.dumps с ensure_ascii=False для чистого UTF-8
        json_content = json.dumps(report, ensure_ascii=False, indent=4)
        response = make_response(json_content)
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        response.headers['Content-Disposition'] = 'attachment; filename=full_report.json'
        return response
    except Exception as e:
        app.logger.error(f"Error in export: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Создать таблицы (теперь здесь, после wait_for_db)
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS flights (
                center TEXT,
                flight_id TEXT PRIMARY KEY,
                type TEXT,
                dep_lat FLOAT,
                dep_lon FLOAT,
                arr_lat FLOAT,
                arr_lon FLOAT,
                dep_date DATE,
                dep_time TIME,
                arr_date DATE,
                arr_time TIME,
                duration_min FLOAT,
                region TEXT,
                dep_geom GEOMETRY(POINT, 4326)
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS metrics (
                region TEXT PRIMARY KEY,
                flight_count INTEGER,
                avg_duration_min FLOAT,
                total_duration_min FLOAT,
                area_km2 FLOAT,
                flight_density FLOAT
            );
        """))
        conn.commit()
    app.run(debug=True, host='0.0.0.0', port=5000)
