#!/bin/bash
set -e

# Проверяем, что DB_URL задан
if [ -z "$DB_URL" ]; then
  >&2 echo "Error: DB_URL environment variable is not set"
  exit 1
fi

# Парсим DB_URL (ожидаем формат postgresql://user:password@host:port/db)
# Используем awk и sed для извлечения параметров
HOST=$(echo "$DB_URL" | sed -E 's/postgresql:\/\/[^:]+:[^@]+@([^:\/]+).*/\1/')
PORT=$(echo "$DB_URL" | sed -E 's/.*:([0-9]+)\/.*/\1/')
USER=$(echo "$DB_URL" | sed -E 's/postgresql:\/\/([^:]+):.*/\1/')
PASSWORD=$(echo "$DB_URL" | sed -E 's/postgresql:\/\/[^:]+:([^@]+)@.*/\1/')
DB=$(echo "$DB_URL" | sed -E 's/.*\/([^?]+).*/\1/')

# Проверяем доступность базы с помощью pg_isready
until pg_isready -h "$HOST" -p "$PORT" -U "$USER" -d "$DB"; do
  >&2 echo "Postgres is unavailable ($HOST:$PORT) - sleeping"
  sleep 2
done

>&2 echo "Postgres is up - executing command"
exec "$@"