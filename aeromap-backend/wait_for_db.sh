#!/bin/bash
set -e

host="db"
port=5432
user="aviation_user"
password="aviation_pass"
db="aviation_db"

until PGPASSWORD=$password psql -h "$host" -p "$port" -U "$user" -d "$db" -c '\q'; do
  >&2 echo "Postgres is unavailable - sleeping"
  sleep 2
done

>&2 echo "Postgres is up - executing command"
exec "$@"