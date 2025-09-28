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
# from flask_oidc import OpenIDConnect  # Закомментировано для пропуска аутентификации

app = Flask(__name__)

# Конфиг для Keycloak/OpenID Connect (закомментировано для dev)
# app.config.update({
#     'SECRET_KEY': 'your_secret_key_here',
#     'OIDC_CLIENT_SECRETS': 'client_secrets.json',
#     'OIDC_ID_TOKEN_COOKIE_SECURE': False,
#     'OIDC_USER_INFO_ENABLED': True,
#     'OIDC_OPENID_REALM': 'your_realm',
#     'OIDC_SCOPES': ['openid', 'email', 'profile'],
#     'OIDC_INTROSPECTION_AUTH_METHOD': 'client_secret_post'
# })
# oidc = OpenIDConnect(app)

# Swagger UI (документирование по ТЗ)
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(SWAGGER_URL, API_URL, config={'app_name': "БПЛА Анализ"})
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

DB_URL = 'postgresql://aviation_user:aviation_pass@localhost:5432/aviation_db'
engine = create_engine(DB_URL)
SHAPEFILE_PATH = 'shapefiles/RF.shp'  # Путь к shapefile

# Функции парсинга
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
        if flight_id is None or dep_lat is None:
            return None
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
        print(f"Error parsing row: {str(e)} - Row: {row_str[:100]}...")
        return None

def get_region(lat, lon, gdf):
    if pd.isna(lat) or pd.isna(lon):
        return 'Unknown'
    point = Point(lon, lat)
    matching = gdf[gdf.geometry.contains(point)]
    if not matching.empty:
        region = matching['name_ru'].iloc[0]
        print(f"Found region for {lat}, {lon}: {region}")
        return region
    print(f"No region found for {lat}, {lon}")
    return 'Unknown'

# Эндпоинты
@app.route('/metrics', methods=['GET'])
# @oidc.require_login  # Закомментировано для dev
def get_metrics():
    try:
        year = request.args.get('year')
        query = "SELECT * FROM metrics;"
        if year:
            query = f"""
                SELECT region, COUNT(flight_id) as flight_count, AVG(duration_min) as avg_duration_min, SUM(duration_min) as total_duration_min
                FROM flights
                WHERE dep_date LIKE '{year}%'
                GROUP BY region;
            """
        with engine.connect() as conn:
            metrics_df = pd.read_sql(query, conn)
        if metrics_df.empty:
            return jsonify({"error": "No metrics available"}), 404
        metrics_df = metrics_df.sort_values(by='flight_count', ascending=False)
        response = jsonify(metrics_df.to_dict(orient='records'))
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/report/graph', methods=['GET'])
# @oidc.require_login  # Закомментировано для dev
def get_graph():
    try:
        year = request.args.get('year')
        format_type = request.args.get('format', 'png')
        query = "SELECT region, flight_count FROM metrics ORDER BY flight_count DESC LIMIT 10;"
        if year:
            query = f"""
                SELECT region, COUNT(flight_id) as flight_count
                FROM flights
                WHERE dep_date LIKE '{year}%'
                GROUP BY region
                ORDER BY flight_count DESC LIMIT 10;
            """
        with engine.connect() as conn:
            metrics_df = pd.read_sql(query, conn)
        if metrics_df.empty:
            return jsonify({"error": "No data for graph"}), 404
        plt.figure(figsize=(10, 6))
        plt.bar(metrics_df['region'], metrics_df['flight_count'])
        plt.title('Топ регионов по количеству полетов БПЛА')
        plt.xlabel('Регион')
        plt.ylabel('Количество полетов')
        plt.xticks(rotation=45)
        plt.tight_layout()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format=format_type)
        img_buf.seek(0)
        mimetype = 'image/png' if format_type == 'png' else 'image/jpeg'
        response = make_response(send_file(img_buf, mimetype=mimetype))
        response.headers['Content-Disposition'] = f'attachment; filename=flight_graph.{format_type}'
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST'])
# @oidc.require_login  # Закомментировано для dev
def upload_data():
    try:
        file = request.files['file']
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
        df = pd.DataFrame(parsed_flights).drop_duplicates(subset=['flight_id', 'dep_date'])
        with Env(SHAPE_RESTORE_SHX='YES'):
            gdf = gpd.read_file(SHAPEFILE_PATH)
        if gdf.crs != 'EPSG:4326':
            gdf = gdf.to_crs('EPSG:4326')
        df['region'] = df.apply(lambda row: get_region(row['dep_lat'], row['dep_lon'], gdf), axis=1)
        df['dep_geom'] = df.apply(lambda row: f"SRID=4326;POINT({row['dep_lon']} {row['dep_lat']})" if pd.notna(row['dep_lat']) else None, axis=1)
        df.to_sql('flights', engine, if_exists='append', index=False, method='multi', dtype={'dep_geom': Geometry('POINT', srid=4326)})
        # Пересчет метрик
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
        return jsonify({"status": "Data uploaded, parsed and metrics updated"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)