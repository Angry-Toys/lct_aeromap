import six 
import re
from datetime import datetime, timedelta
import pandas as pd
from multiprocessing import Pool
import geopandas as gpd
from shapely.geometry import Point
from fiona import Env
from sqlalchemy import create_engine, text
from geoalchemy2 import Geometry
from flask import Flask, jsonify, send_file, make_response, request, redirect, g
from flask import session   
from flask_swagger_ui import get_swaggerui_blueprint
from flask_cors import CORS
from flask import Blueprint
from flask_oidc import OpenIDConnect 
import matplotlib.pyplot as plt
import io
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from sklearn.linear_model import LinearRegression
import numpy as np
from werkzeug.exceptions import Unauthorized
from werkzeug.middleware.proxy_fix import ProxyFix
from sklearn.cluster import DBSCAN
import numpy as np
from authlib.integrations.flask_client import OAuthError

app = Flask(__name__)
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False 
app.config['SESSION_COOKIE_DOMAIN'] = None
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
CORS(app, origins=['*']) 
OIDC_ISSUER_URL = os.getenv('OIDC_ISSUER', 'http://keycloak:8080/realms/aviation-realm')
HOST_MAPPING = 'keycloak:8080'
PUBLIC_HOST = 'localhost:8080'
app.config.update({
    'OIDC_CLIENT_SECRETS': {  
        'web': {
            'client_id': os.getenv('OIDC_CLIENT_ID', 'aviation-api'),
            'client_secret': os.getenv('OIDC_CLIENT_SECRET', '8HUnPbB26kdCImTFXLfN9scQLQmYLW4d'),
            'issuer': OIDC_ISSUER_URL, 
            'OIDC_VALID_ISSUERS': ['http://keycloak:8080/realms/aviation-realm', 'http://localhost:8080/realms/aviation-realm'],

            'leeway': 60
        }
    },
    'OVERWRITE_REDIRECT_URI': 'http://localhost:5000/oidc_callback',  
    'SECRET_KEY': os.getenv('SECRET_KEY', 'your-secret-key'),
    'SESSION_COOKIE_SAMESITE': 'Lax',
    'SESSION_COOKIE_SECURE': False,
})

oidc = OpenIDConnect(app)

bp = Blueprint('api', __name__, url_prefix='/api')


handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)


DB_URL = os.getenv('DB_URL', 'postgresql://aviation_user:aviation_pass@localhost:5432/aviation_db')
engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=3600)
SHAPEFILE_PATH = 'shapefiles/RF.shp'


SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(SWAGGER_URL, API_URL, config={'app_name': "БПЛА Анализ"})
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)


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
                return None, None  
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
            coord_matches = re.findall(r'(\d{4}[NS]\d{5}[EW])', coord_str) 
            if coord_matches:
                return parse_coords(coord_matches[0])
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
    if not row_str.strip() or 'truncated' in row_str:
        return None
    try:
        row_str = re.sub(r'row\d+:\s*', '', row_str).strip().replace('\n', ' ')
        app.logger.debug(f"Parsing row: {row_str[:100]}...") 
        center_match = re.match(r'(.*?)\s*\(', row_str)
        center = center_match.group(1).strip() if center_match else ''
        shr_full = row_str[center_match.end()-1:] if center_match else row_str
        sections = re.split(r'\),', shr_full)
        shr_section = next((s.strip() for s in sections if 'SHR' in s.upper()), '')
        if not shr_section:
            app.logger.debug(f"No SHR section in row: {row_str}")
            return None
        sid_match = re.search(r'SID/(\d+)', shr_section)
        flight_id = sid_match.group(1) if sid_match else None
        typ_match = re.search(r'TYP/([\w/]+)', shr_section)
        flight_type = typ_match.group(1) if typ_match else None
        customer_match = re.search(r'OPR/(.*?)( TYP/| RMK/| STS/| REG/| DEST/| $)', shr_section, re.DOTALL)
        customer = customer_match.group(1).strip() if customer_match else None
        operator = customer  
        dof_match = re.search(r'DOF/(\d{6})', shr_section)
        dof_date = parse_date(dof_match.group(1)) if dof_match else None
        dep_coord_str = re.search(r'DEP/([\dNS EWСВ\d]+)', shr_section).group(1) if re.search(r'DEP/([\dNS EWСВ\d]+)', shr_section) else ''
        if not dep_coord_str:
            zona_match = re.search(r'/ZONA\s*(.+?)/', shr_section)
            dep_coord_str = zona_match.group(1) if zona_match else ''
        dep_lat, dep_lon = parse_coords(dep_coord_str)
        arr_coord_str = re.search(r'DEST/([\dNS EWСВ\d]+)', shr_section).group(1) if re.search(r'DEST/([\dNS EWСВ\d]+)', shr_section) else dep_coord_str
        arr_lat, arr_lon = parse_coords(arr_coord_str)
        time_matches = re.findall(r'-Z{3,5}(\d{4})', shr_section)
        dep_time_str = time_matches[0] if time_matches else ''
        dep_time = parse_time(dep_time_str)
        arr_time_str = time_matches[1] if len(time_matches) > 1 else ''
        arr_time = parse_time(arr_time_str)
        dep_date = dof_date
        arr_date = dep_date if dep_time and arr_time and arr_time > dep_time else (dep_date + timedelta(days=1) if dep_date else None)
        duration_min = calculate_duration(dep_date, dep_time, arr_date, arr_time)
        parsed = {
            'center': center,
            'flight_id': flight_id,
            'type': flight_type,
            'operator': operator,
            'customer': customer,
            'dep_lat': dep_lat,
            'dep_lon': dep_lon,
            'arr_lat': arr_lat,
            'arr_lon': arr_lon,
            'dep_date': dep_date,
            'dep_time': dep_time,
            'arr_date': arr_date,
            'arr_time': arr_time,
            'duration_min': duration_min
        }
        app.logger.debug(f"Parsed row: {flight_id} - Customer: {customer} - Type: {flight_type} - Dep: {dep_lat},{dep_lon} - Dep time: {dep_time_str} - Arr time: {arr_time_str}")
        if not flight_id:
            return None
        return parsed
    except Exception as e:
        app.logger.error(f"Parse error in row: {row_str} - {str(e)}")
        return None

def get_region(lat, lon, gdf):
    if pd.isna(lat) or pd.isna(lon):
        return None
    point = Point(lon, lat)
    for idx, row in gdf.iterrows():
        if row['geometry'].contains(point):
            return row['name_ru']
    return None

@bp.route('/upload', methods=['POST'])
#@oidc.accept_token()
def upload_file():
    #if 'authlib_server_oauth2_token' not in g or g.authlib_server_oauth2_token is None:
        #raise Unauthorized("Missing or invalid authorization token")
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if file and file.filename.endswith('.xlsx'):
            xls = pd.ExcelFile(file)
            all_rows = []
            for sheet_name in xls.sheet_names:
                df_sheet = pd.read_excel(xls, sheet_name=sheet_name, header=None)
                if df_sheet.empty or df_sheet.shape[1] == 0:
                    continue
                for i in range(len(df_sheet)):
                    row_series = df_sheet.iloc[i].dropna().astype(str)
                    if row_series.empty:
                        continue
                    row_str = ','.join(row_series)
                    all_rows.append(row_str)
            if not all_rows:
                return jsonify({"error": "No valid data in any sheet"}), 400
            
            app.logger.info(f"Collected {len(all_rows)} raw rows from Excel")
            with Pool(processes=4) as pool:
                parsed_flights = pool.map(parse_flight_row, all_rows)
            parsed_flights = [p for p in parsed_flights if p]
            if not parsed_flights:
                return jsonify({"error": "No valid flights parsed"}), 400
            df = pd.DataFrame(parsed_flights).drop_duplicates(subset=['flight_id', 'dep_date'])
            app.logger.info(f"Parsed {len(df)} unique flights")
            with Env(SHAPE_RESTORE_SHX='YES'):
                gdf = gpd.read_file(SHAPEFILE_PATH)
            if gdf.crs != 'EPSG:4326':
                gdf = gdf.to_crs('EPSG:4326')
            df['region'] = df.apply(lambda row: get_region(row['dep_lat'], row['dep_lon'], gdf), axis=1)
            df['dep_geom'] = df.apply(lambda row: f"SRID=4326;POINT({row['dep_lon']} {row['dep_lat']})" if pd.notna(row['dep_lat']) else None, axis=1)
            df.to_sql('flights', engine, if_exists='append', index=False, method='multi', dtype={'dep_geom': Geometry('POINT', srid=4326)})
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

@bp.route('/webhook', methods=['POST'])
#@oidc.accept_token()
def webhook():
    #if 'authlib_server_oauth2_token' not in g or g.authlib_server_oauth2_token is None:
        #raise Unauthorized("Missing or invalid authorization token")
    try:
        data = request.json
        if not data or 'flights' not in data:
            return jsonify({"error": "Invalid payload"}), 400
        rows = data['flights']
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

@bp.route('/report/export', methods=['GET'])
#@oidc.accept_token()
def export_report():
    #if 'authlib_server_oauth2_token' not in g or g.authlib_server_oauth2_token is None:
        #raise Unauthorized("Missing or invalid authorization token")
    try:
        with engine.connect() as conn:
            flights_df = pd.read_sql("SELECT * FROM flights;", conn)
            metrics_df = pd.read_sql("SELECT * FROM metrics;", conn)

        flights_df['dep_date'] = flights_df['dep_date'].astype(str)
        flights_df['arr_date'] = flights_df['arr_date'].astype(str)
        flights_df['dep_time'] = flights_df['dep_time'].apply(lambda x: x.strftime('%H:%M:%S') if pd.notnull(x) else None)
        flights_df['arr_time'] = flights_df['arr_time'].apply(lambda x: x.strftime('%H:%M:%S') if pd.notnull(x) else None)
        report = {
            "flights": flights_df.to_dict(orient='records'),
            "metrics": metrics_df.to_dict(orient='records')
        }
        json_content = json.dumps(report, ensure_ascii=False, indent=4)
        response = make_response(json_content)
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        response.headers['Content-Disposition'] = 'attachment; filename=full_report.json'
        return response
    except Exception as e:
        app.logger.error(f"Error in export: {str(e)}")
        return jsonify({"error": str(e)}), 500

@bp.route('/metrics', methods=['GET'])
def get_metrics():
    try:
        year = request.args.get('year')
        month = request.args.get('month')
        region = request.args.get('region')
        group_by = request.args.get('group_by', 'regions')  # Default: regions; alternative: customers

        if group_by not in ['regions', 'customers']:
            return jsonify({"error": "Invalid group_by (regions or customers)"}), 400

        # Base query с фильтрами
        base_query = """
            SELECT f.region, f.customer, f.flight_id, f.duration_min, f.dep_date, f.dep_time, m.area_km2
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
        if group_by == 'customers':
            where_parts.append("f.customer IS NOT NULL")
        if where_parts:
            base_query += " WHERE " + " AND ".join(where_parts)
        base_query += ";"
        with engine.connect() as conn:
            df = pd.read_sql(text(base_query), conn, params=params)
        if df.empty:
            return jsonify({"error": "No metrics available"}), 404

        if group_by == 'customers':
            df['customer'] = df['customer'].apply(lambda x: re.sub(r'\+\d{10,}', '', x).strip() if x else x)
            metrics = df.groupby('customer').agg({
                'flight_id': 'count',
                'duration_min': ['mean', 'sum']
            }).reset_index()
            metrics.columns = ['customer', 'flight_count', 'avg_duration_min', 'total_duration_min']
            metrics = metrics.sort_values('flight_count', ascending=False)
            return jsonify(metrics.to_dict(orient='records')), 200

        # Original logic for regions
        metrics_df = df.groupby('region').agg(
            flight_count=('flight_id', 'count'),
            avg_duration_min=('duration_min', 'mean'),
            total_duration_min=('duration_min', 'sum'),
            area_km2=('area_km2', 'max')
        ).reset_index()
        metrics_df['flight_density'] = metrics_df['flight_count'] / metrics_df['area_km2']
        # Peak hourly: Фильтруем строки с dep_time IS NOT NULL
        df_time = df[df['dep_time'].notnull()].copy()
        if not df_time.empty:
            df_time['dep_datetime'] = pd.to_datetime(df_time['dep_date'].astype(str) + ' ' + df_time['dep_time'].astype(str))
            df_time['hour'] = df_time['dep_datetime'].dt.floor('H')
            metrics_df['peak_load_hourly'] = df_time.groupby(['region', 'hour'])['flight_id'].count().groupby('region').max().values
        else:
            metrics_df['peak_load_hourly'] = 0
        # Daily stats
        daily_counts = df.groupby(['region', 'dep_date'])['flight_id'].count().reset_index(name='daily_count')
        metrics_df = metrics_df.merge(daily_counts.groupby('region')['daily_count'].agg(['mean', 'median']).reset_index(), on='region')
        metrics_df = metrics_df.rename(columns={'mean': 'avg_daily_flights', 'median': 'median_daily_flights'})
        metrics_df = metrics_df.sort_values(by='flight_count', ascending=False)
        # Growth
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
        hourly_query = "SELECT EXTRACT(HOUR FROM dep_time) as hour, COUNT(*) as count FROM flights WHERE dep_time IS NOT NULL GROUP BY hour;"
        with engine.connect() as conn:
            hourly_df = pd.read_sql(hourly_query, conn)
        metrics_df['hourly_distribution'] = [hourly_df.to_dict(orient='records')] * len(metrics_df)
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

@bp.route('/metrics/operators', methods=['GET'])
#@oidc.accept_token()
def get_operators_metrics():
    #if 'authlib_server_oauth2_token' not in g or g.authlib_server_oauth2_token is None:
        #raise Unauthorized("Missing or invalid authorization token")
    try:
        with engine.connect() as conn:
            df = pd.read_sql("SELECT operator, flight_id, duration_min FROM flights WHERE operator IS NOT NULL;", conn)
        if df.empty:
            return jsonify({"error": "No data"}), 404
        metrics = df.groupby('operator').agg({
            'flight_id': 'count',
            'duration_min': ['mean', 'sum']
        }).reset_index()
        metrics.columns = ['operator', 'flight_count', 'avg_duration_min', 'total_duration_min']
        return jsonify(metrics.to_dict(orient='records')), 200
    except Exception as e:
        app.logger.error(f"Error in operators metrics: {str(e)}")
        return jsonify({"error": str(e)}), 500

@bp.route('/metrics/types', methods=['GET'])
#@oidc.accept_token()
def get_types_metrics():
    #if 'authlib_server_oauth2_token' not in g or g.authlib_server_oauth2_token is None:
        #raise Unauthorized("Missing or invalid authorization token")
    try:
        with engine.connect() as conn:
            df = pd.read_sql("SELECT type, flight_id, duration_min FROM flights WHERE type IS NOT NULL;", conn)
        if df.empty:
            return jsonify({"error": "No data"}), 404
        metrics = df.groupby('type').agg({
            'flight_id': 'count',
            'duration_min': ['mean', 'sum']
        }).reset_index()
        metrics.columns = ['type', 'flight_count', 'avg_duration_min', 'total_duration_min']
        return jsonify(metrics.to_dict(orient='records')), 200
    except Exception as e:
        app.logger.error(f"Error in types metrics: {str(e)}")
        return jsonify({"error": str(e)}), 500

@bp.route('/metrics/total', methods=['GET'])
#@oidc.accept_token()
def get_total_metrics():
    #if 'authlib_server_oauth2_token' not in g or g.authlib_server_oauth2_token is None:
        #raise Unauthorized("Missing or invalid authorization token")
    try:
        with engine.connect() as conn:
            df = pd.read_sql("SELECT flight_id, duration_min FROM flights;", conn)
            metrics_df = pd.read_sql("SELECT SUM(area_km2) as total_area_km2 FROM metrics;", conn)
        if df.empty:
            return jsonify({"error": "No data"}), 404
        total_flights = len(df)
        avg_duration = df['duration_min'].mean()
        total_duration = df['duration_min'].sum()
        total_area = metrics_df['total_area_km2'].iloc[0] if not metrics_df.empty else 0
        total_density = total_flights / total_area if total_area > 0 else 0
        return jsonify({
            "total_flights": total_flights,
            "avg_duration_min": avg_duration,
            "total_duration_min": total_duration,
            "total_flight_density": total_density
        }), 200
    except Exception as e:
        app.logger.error(f"Error in total metrics: {str(e)}")
        return jsonify({"error": str(e)}), 500

    
@bp.route('/metrics/customers', methods=['GET'])
@oidc.require_login
def get_customers_list():
    if 'authlib_server_oauth2_token' not in g or g.authlib_server_oauth2_token is None:
        raise Unauthorized("Missing or invalid authorization token")
    try:
        with engine.connect() as conn:
            df = pd.read_sql("SELECT DISTINCT customer FROM flights WHERE customer IS NOT NULL;", conn)
        if df.empty:
            return jsonify([]), 200
        df['customer'] = df['customer'].apply(lambda x: re.sub(r'\+\d{10,}', '', x).strip() if x else x)
        customers = df['customer'].tolist()
        return jsonify(customers), 200
    except Exception as e:
        app.logger.error(f"Error in customers list: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
@bp.route('/flights/coords', methods=['GET'])
#@oidc.accept_token()
def get_flights_coords():
    #if 'authlib_server_oauth2_token' not in g or g.authlib_server_oauth2_token is None:
        #raise Unauthorized("Missing or invalid authorization token")
    try:
        year = request.args.get('year')
        month = request.args.get('month')
        region = request.args.get('region')
        limit = int(request.args.get('limit', 5000)) 
        cluster_eps = float(request.args.get('cluster_eps', 0.05)) 

        sql = """
            SELECT dep_lat AS lat, dep_lon AS lon, 
                   COALESCE(duration_min, 1) AS intensity 
            FROM flights 
            WHERE dep_lat IS NOT NULL AND dep_lon IS NOT NULL
        """
        params = {}
        if year:
            sql += " AND EXTRACT(YEAR FROM dep_date) = :year"
            params['year'] = int(year)
        if month:
            sql += " AND EXTRACT(MONTH FROM dep_date) = :month"
            params['month'] = int(month)
        if region:
            sql += " AND region = :region"
            params['region'] = region
        sql += " LIMIT :limit;"
        params['limit'] = limit

        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn, params=params)

        if df.empty:
            return jsonify([])

        coords = df[['lat', 'lon']].values
        if len(coords) > 1:
            radians = np.deg2rad(coords)
            db = DBSCAN(eps=cluster_eps / 6371.0, min_samples=1, algorithm='ball_tree', metric='haversine') 
            labels = db.fit_predict(radians)

            aggregated = []
            for label in np.unique(labels):
                if label == -1: continue 
                cluster_df = df[labels == label]
                agg_lat = cluster_df['lat'].mean()
                agg_lon = cluster_df['lon'].mean()
                count = len(cluster_df)
                agg_intensity = cluster_df['intensity'].sum()
                aggregated.append({
                    'lat': agg_lat,
                    'lon': agg_lon,
                    'count': count,
                    'intensity': agg_intensity
                })
            return jsonify(aggregated), 200

        return jsonify(df.to_dict(orient='records')), 200
    except Exception as e:
        app.logger.error(f"Error in flights/coords: {str(e)}")
        return jsonify({"error": str(e)}), 500

@bp.route('/regions/flights', methods=['GET'])
#@oidc.accept_token()
def get_regions_flights():
    #if 'authlib_server_oauth2_token' not in g or g.authlib_server_oauth2_token is None:
        #raise Unauthorized("Missing or invalid authorization token")
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
    
@bp.route('/compare', methods=['GET'])
#@oidc.accept_token()
def compare_periods():
    #if 'authlib_server_oauth2_token' not in g or g.authlib_server_oauth2_token is None:
        #raise Unauthorized("Missing or invalid authorization token")
    try:
        year1 = request.args.get('year1')
        month1 = request.args.get('month1')
        year2 = request.args.get('year2')
        month2 = request.args.get('month2')
        metric = request.args.get('metric', 'count')  

        if not (year1 and year2):
            return jsonify({"error": "Required: year1 and year2"}), 400
        if metric not in ['count', 'avg_duration']:
            return jsonify({"error": "Invalid metric (count or avg_duration)"}), 400

        sql_base = "SELECT flight_id, duration_min FROM flights WHERE EXTRACT(YEAR FROM dep_date) = :year"
        params1 = {'year': int(year1)}
        params2 = {'year': int(year2)}

        if month1:
            sql_base += " AND EXTRACT(MONTH FROM dep_date) = :month"
            params1['month'] = int(month1)
        if month2:
            sql_base += " AND EXTRACT(MONTH FROM dep_date) = :month"
            params2['month'] = int(month2)

        with engine.connect() as conn:
            df1 = pd.read_sql(text(sql_base), conn, params=params1)
            df2 = pd.read_sql(text(sql_base), conn, params=params2)

        if metric == 'count':
            val1 = len(df1)
            val2 = len(df2)
        else:  
            val1 = df1['duration_min'].mean() if not df1.empty else 0
            val2 = df2['duration_min'].mean() if not df2.empty else 0

        diff_percent = ((val2 - val1) / val1 * 100) if val1 > 0 else None

        period1_label = f"{year1}-{month1:02d}" if month1 else year1
        period2_label = f"{year2}-{month2:02d}" if month2 else year2

        return jsonify({
            period1_label: {"value": val1},
            period2_label: {"value": val2},
            "diff_percent": diff_percent
        }), 200
    except Exception as e:
        app.logger.error(f"Error in compare: {str(e)}")
        return jsonify({"error": str(e)}), 500

@bp.route('/forecast', methods=['GET'])
#@oidc.accept_token()
def forecast_period():
    #if 'authlib_server_oauth2_token' not in g or g.authlib_server_oauth2_token is None:
        #raise Unauthorized("Missing or invalid authorization token")
    try:
        year = int(request.args.get('year', 2025))
        month = request.args.get('month')  
        metric = request.args.get('metric', 'count')  

        if metric not in ['count', 'avg_duration']:
            return jsonify({"error": "Invalid metric (count or avg_duration)"}), 400

        sql = """
            SELECT EXTRACT(MONTH FROM dep_date) AS month, flight_id, duration_min 
            FROM flights 
            WHERE EXTRACT(YEAR FROM dep_date) = :year AND dep_date <= CURRENT_DATE
            GROUP BY month, flight_id, duration_min
        """
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn, params={'year': year})

        if df.empty:
            return jsonify({"error": "No historical data for forecast"}), 404

        if metric == 'count':
            monthly = df.groupby('month').size().reset_index(name='value')
        else:
            monthly = df.groupby('month')['duration_min'].mean().reset_index(name='value')

        X = np.array(monthly['month']).reshape(-1, 1)
        y = monthly['value']
        model = LinearRegression()
        model.fit(X, y)

        if month:  
            pred_month = int(month)
            if pred_month <= max(monthly['month']):
                return jsonify({"error": "Month already passed, use actual data"}), 400
            pred = model.predict([[pred_month]])[0]
            return jsonify({f"{year}-{pred_month:02d}": {"predicted_value": pred}}), 200
        else:  
            current_max_month = max(monthly['month'])
            remaining_months = range(int(current_max_month) + 1, 13)
            preds = [model.predict([[m]])[0] for m in remaining_months]
            total_pred = monthly['value'].sum() + sum(preds)
            return jsonify({str(year): {"predicted_total": total_pred}}), 200
    except Exception as e:
        app.logger.error(f"Error in forecast: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
from authlib.integrations.flask_client import OAuthError  # Добавьте импорт если нужно

@app.route('/auth/login')
def login():
    try:
        redirect_uri = app.config.get('OVERWRITE_REDIRECT_URI', 'http://localhost:5000/oidc_callback')
        auth_uri, state = oidc.client.create_authorization_url(  # Изменено на oidc.client
            oidc.client.metadata['authorization_endpoint'],
            redirect_uri=redirect_uri,
            scope='openid profile email'  # Scopes по нужде, настройте в Keycloak
        )
        auth_uri = auth_uri.replace('keycloak:8080', 'localhost:8080')  # Замена host для browser
        session['oidc_state'] = state  # Защита от CSRF
        return redirect(auth_uri)
    except OAuthError as e:
        app.logger.error(f"OAuth error in login: {str(e)}")
        return jsonify({"error": "Authentication failed", "details": str(e)}), 500
    except Exception as e:
        app.logger.error(f"Unexpected error in login: {str(e)}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route('/oidc_callback')
def oidc_callback():
    if 'error' in request.args:
        return jsonify({"error": request.args['error']}), 400

    if 'code' not in request.args:
        return jsonify({"error": "No authorization code"}), 400

    # Проверка state против CSRF
    if session.get('oidc_state') != request.args.get('state'):
        return jsonify({"error": "Invalid state parameter"}), 403

    try:
        # Обмен code на token
        token = oidc.client.authorize_access_token(  # Изменено на oidc.client
            request.url,  # Полный callback URL
            code=request.args['code'],
            redirect_uri=app.config['OVERWRITE_REDIRECT_URI']
        )
        # Получение userinfo
        userinfo = oidc.client.userinfo(token=token['access_token'])  # Изменено на oidc.client
        session['oidc_token'] = token
        session['userinfo'] = userinfo
        session.pop('oidc_state', None)  # Очистка
        app.logger.info("Successful login")
        # Для теста API: Вернуть token (в prod — redirect('/'))
        return jsonify({
            "message": "Logged in",
            "access_token": token['access_token'],
            "refresh_token": token.get('refresh_token'),
            "userinfo": userinfo
        })
    except OAuthError as e:
        app.logger.error(f"OAuth error in callback: {str(e)}")
        return jsonify({"error": "Token exchange failed", "details": str(e)}), 500
    except Exception as e:
        app.logger.error(f"Unexpected error in callback: {str(e)}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route('/auth/logout')
def logout():
    oidc.logout()
    return redirect('/')

@app.route('/')
def index():
    if 'oidc_token' in session:
        return jsonify({"message": "Logged in", "userinfo": session.get('userinfo')})
    return jsonify({"message": "Welcome, please login"}), 200
        
app.register_blueprint(bp)

if __name__ == '__main__':
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
                dep_geom GEOMETRY(POINT, 4326),
                operator TEXT,
                customer TEXT
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