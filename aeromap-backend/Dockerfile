# Base на Python 3.12 slim
FROM python:3.12-slim

# System deps для geopandas (GDAL etc.)
RUN apt-get update && apt-get install -y \
    libgdal-dev gdal-bin libspatialindex-dev libgeos-dev libproj-dev python3-dev \
    postgresql-client && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Рабочая dir
WORKDIR /app

# Копируем requirements
COPY requirements.txt .

# Install deps
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код и shapefile
COPY . .
COPY shapefiles/ shapefiles/

# Expose port
EXPOSE 5000

# Запуск
CMD ["python", "api.py"]