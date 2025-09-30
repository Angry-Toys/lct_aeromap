# Base на Python 3.12 slim (экономим размер, как в TZ)
FROM python:3.12-slim

# System deps для geopandas/shapely/fiona (GDAL, PROJ, etc.)
RUN apt-get update && apt-get install -y \
    libgdal-dev gdal-bin libspatialindex-dev libgeos-dev libproj-dev \
    postgresql-client && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Рабочая dir
WORKDIR /app

# Копируем requirements (сначала, для кэша)
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код и shapefile (shapefiles/ dir с RF.shp)
COPY . .
COPY shapefiles/ shapefiles/

# Expose port (как в app.py)
EXPOSE 5000

# Запуск
CMD ["python", "app.py"]