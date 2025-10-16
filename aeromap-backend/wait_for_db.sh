#!/bin/sh
set -e

if [ -z "$DB_URL" ]; then
  >&2 echo "Error: DB_URL environment variable is not set"
  exit 1
fi

HOST=$(echo "$DB_URL" | sed -E 's/postgresql:\/\/[^:]+:[^@]+@([^:\/]+).*/\1/')
PORT=$(echo "$DB_URL" | sed -E 's/.*:([0-9]+)\/.*/\1/')
USER=$(echo "$DB_URL" | sed -E 's/postgresql:\/\/([^:]+):.*/\1/')
PASSWORD=$(echo "$DB_URL" | sed -E 's/postgresql:\/\/[^:]+:([^@]+)@.*/\1/')
DB=$(echo "$DB_URL" | sed -E 's/.*\/([^?]+).*/\1/')

until pg_isready -h "$HOST" -p "$PORT" -U "$USER" -d "$DB"; do
  >&2 echo "Postgres is unavailable ($HOST:$PORT) - sleeping"
  sleep 2
done

>&2 echo "Postgres is up - executing command"
exec "$@"