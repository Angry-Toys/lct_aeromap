FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libgdal-dev gdal-bin libspatialindex-dev libgeos-dev libproj-dev python3-dev \
    postgresql-client dos2unix bash && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r requirements.txt
# Verify key deps
RUN python -c "from flask_cors import CORS; print('flask_cors OK')" || (echo "flask_cors install failed" && exit 1)

COPY . .
COPY shapefiles/ shapefiles/

COPY wait_for_db.sh .
RUN dos2unix wait_for_db.sh
RUN chmod +x wait_for_db.sh

EXPOSE 5000

CMD ["./wait_for_db.sh", "python", "api.py"]