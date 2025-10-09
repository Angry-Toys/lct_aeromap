# Сопроводительная документация к решению задачи 1: Сервис для анализа количества и длительности полетов гражданских беспилотников в регионах Российской Федерации

**Команда:** Ферзь  
**Дата:** 02 октября 2025  
**Репозиторий:** [Ссылка на GitHub]  
**Прототип:** [Ссылка на развернутый сервис]  
**Контакты:** [Email/GitHub]  
Эта документация соответствует требованиям ТЗ (раздел 3.6.6): описывает используемые методы обработки данных, условия и ограничения, подробные инструкции по компиляции, сборке и установке. Код открыт, не обфусцирован, доступен в репозитории.

## 1. Используемые методы обработки данных

Сервис обрабатывает статистические данные Росавиации (формализованные сообщения по "Табелю сообщений" Минтранса России от 24.01.2013 №13). Методы:

### Импорт и парсинг (раздел 3.1 ТЗ)
- Прием пакетов через POST /upload (multipart XLSX) или /webhook (JSON с массивом строк).
- Парсер (parse_flight_row): Использует регулярные выражения (re) для извлечения ID полета (SID), типа БПЛА (TYP), координат взлета/посадки (DEP/DEST, ZONA, ADEPZ/ADARRZ), даты (DOF/ADD/ADA), времени (ZZZZ, ATD/ATA).
- Нормализация: Автоматическая валидация форматов (parse_coords для DDMMNS DDDMM EW, parse_time для HHMM, parse_date для YYMMDD), игнор лишних полей (e.g., OPR/RMK), отчет об ошибках в app.log (warnings/errors).
- Пример: Строка SHR/IDEP/IARR -> dict с flight_id, type, dep_lat/lon, duration_min (calculate_duration с timedelta для кросс-дней).

### Валидация и очистка (раздел 3.2 ТЗ)
- Проверка диапазонов (lat 0-90, lon 0-180, min/sec 0-59).
- Удаление дубликатов: pd.drop_duplicates по flight_id/dep_date.
- Очистка: strip(), upper() для coords, None для инвалид.

### Геопривязка (раздел 3.3 ТЗ)
- Привязка dep_lat/lon к регионам по официальным SHP-файлам (RF.shp от Росреестра/GIS-Lab, CRS EPSG:4326).
- Метод: get_region с Point.contains (geopandas, строго по границам без буферов).
- Обновление SHP: Ручное (wget от Росреестра), регламент — ежемесячно в cron-job (рекомендация: добавить script update_shp.sh в Docker).

### Хранение данных (раздел 3.4 ТЗ)
- Open Source СУБД: PostgreSQL с PostGIS (docker-compose).
- GiST-индексы: Для dep_geom (GEOMETRY POINT, SRID=4326).
- История: Append в flights/metrics, глубина > года (без удаления).

### Отчеты и визуализация (раздел 3.5 ТЗ)
- Метрики: Базовые (count, avg/sum duration) + расширенные (peak_load_hourly, avg/median_daily, growth_percent, flight_density, hourly_distribution, zero_days) в /metrics.
- Графики: PNG/JPEG в /report/graph (top_regions bar, time_series plot, matplotlib).
- Экспорт: JSON в /report/export (flights + metrics, on-demand).

### Дополнительные метрики (раздел 7 ТЗ)
Все реализованы в /metrics.

## 2. Условия и ограничения внутри решения

### Условия
- Входные данные: XLSX с sheet 'Result_1' (rows как SHR/IDEP/IARR), или JSON в webhook.
- Форматы: Coords DDMMNS DDDMM EW, time HHMM, date YYMMDD (по Табелю №13).
- Регионы: По SHP РФ (Росреестр), 'Unknown' если вне границ.
- Производительность: 75k полетов ~2-3 мин (Pool=4), <5 мин для 10k по ТЗ.
- Надежность: Pool_pre_ping/recycle для DB, logging в app.log.

### Ограничения
- Нет реал-тайм АФТН/ACARS (симуляция webhook).
- Нет XML (только JSON, добавить по запросу).
- Тесты: Покрытие 48% (добавляем до 80%).
- Обновление SHP: Ручное, без auto-cron.

## 3. Подробные инструкции по компиляции, сборке и установке

### Требования
Python 3.12, Docker, PostGIS, библиотеки в requirements.txt (pandas, geopandas, flask, etc.).

### Установка локально
- Клонируйте репозиторий: git clone [your-repo].
- Установите зависимости: pip install -r requirements.txt.
- Запустите DB: docker-compose up db.
- Запустите сервис: python api.py (host=0.0.0.0:5000, проверьте env DB_URL).

### Сборка в Docker
- Выполните: docker-compose up --build (собирает app, запускает db/app).

### Компиляция/тесты
- Выполните: pytest --cov=. (покрытие >=80%, отчет в htmlcov/).

### Развертывание
- **Локально**: Docker-compose.
- **Прод**: Heroku/AWS (настройте env DB_URL, e.g., postgresql://...).

### Инструкции по использованию
- Swagger UI: Откройте /swagger в браузере.
- Загрузка: POST /upload с XLSX.
- Метрики: GET /metrics?year=2025.
- Экспорт: GET /report/export.

#### 3.1 Развертывание в облаке (Yandex Cloud)
Сервис реализован с учетом облачной инфраструктуры Yandex Cloud для обеспечения масштабируемости и доступности, в соответствии с требованиями ТЗ к интеграции с внешними системами (раздел 3.6.3).

- **Подготовка аккаунта Yandex Cloud**: Создайте биллинг-аккаунт и папку в консоли Yandex Cloud[](https://console.cloud.yandex.ru). Установите yc CLI: `curl https://storage.yandexcloud.net/yandexcloud-yc/install.sh | bash` и авторизуйтесь (`yc init`). Включите API: `compute.api`, `managed-kubernetes.api`, `container-registry.api`, `managed-postgresql.api`.

- **Настройка базы данных**: Используется Managed PostgreSQL с PostGIS (версия 16+). Создайте кластер в консоли: имя `aviation-db`, пользователь `aviation_user`, пароль `aviation_pass`, база `aviation_db`. Включите extension PostGIS: подключитесь через psql и выполните `CREATE EXTENSION IF NOT EXISTS postgis;`. DB_URL: `postgresql://aviation_user:aviation_pass@<host>:6432/aviation_db`.

- **Контейнеризация и реестр образов**: Dockerfiles для бэкенда (Flask с geopandas) и фронтенда (Vue.js на Nginx). Создайте реестр: `yc cr registry create aviation-cr`. Постройте и загрузите образы: `docker build -t cr.yandex/<folder-id>/aviation-cr/backend:latest .` и аналогично для фронтенда, затем `yc cr push`.

- **Создание Kubernetes-кластера**: В консоли создайте MKS-кластер (`aviation-k8s`, версия 1.29+, 2 ноды). Получите kubeconfig: `yc managed-kubernetes cluster get-credentials aviation-k8s --external`.

- **Развертывание компонентов**: Примените YAML-манифесты для Deployment и Service (backend/frontend). Пример для бэкенда:
apiVersion: apps/v1
kind: Deployment
metadata:
name: backend
spec:
replicas: 2
template:
spec:
containers:

name: flask-app
image: cr.yandex/<folder-id>/aviation-cr/backend:latest
env:

name: DB_URL
valueFrom: { secretKeyRef: { name: db-secret } }




apiVersion: v1
kind: Service
metadata:
name: backend-svc
spec:
type: ClusterIP
ports:

port: 5000

Аналогично для фронтенда. Создайте Secret для DB_URL: `kubectl create secret generic db-secret --from-literal=DB_URL=...`.

- **Ingress и HTTPS**: Установите Nginx Ingress: `helm install nginx-ingress ingress-nginx/ingress-nginx --namespace ingress-nginx`. Настройте Ingress для роутинга (/ для фронтенда, /api для бэкенда). Выпустите сертификат в Certificate Manager Yandex (Let's Encrypt для <ip>.nip.io), импортируйте как Secret: `kubectl create secret tls aeromap-tls --cert=cert.pem --key=key.pem`. Добавьте в ingress.yaml:
annotations:
nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
tls:

hosts: [84.201.181.11.nip.io]
secretName: aeromap-tls

- **Тестирование**: Проверьте: `kubectl get pods`, `kubectl port-forward svc/backend-svc 5000:5000`. Доступ по `https://<ip>.n

ip.io`.

#### 3.2 Масштабируемость и отказоустойчивость
В соответствии с ТЗ (раздел 3.6.4), сервис спроектирован как масштабируемый микросервис с поддержкой горизонтального масштабирования.

- **Масштабируемость**: Horizontal Pod Autoscaler (HPA): `kubectl autoscale deployment backend --min=2 --max=10 --cpu-percent=50`. Managed PostgreSQL поддерживает auto-scaling. Для нагрузки >10k запросов/мин рекомендуется увеличить ноды в MKS.

- **Отказоустойчивость**: Managed DB с HA (shared), restarts в Docker-compose (`restart: always`), depends_on для последовательного запуска. Мониторинг: Логи в app.log, Prometheus в MKS. Бэкап: Автоматический в Managed PostgreSQL.

#### 3.3 Реализация как микросервиса
Сервис следует микросервисной архитектуре (раздел 3.6.3 ТЗ): Отдельные контейнеры для фронтенда (Vue.js/Nginx) и бэкенда (Flask/PostGIS). Взаимодействие через REST API (/api), CORS в Flask. Интеграция с внешними системами: Webhook для данных Росавиации, возможное расширение на ACARS/AFTN через JSON. Архитектура в Archi/C4: [добавьте описание от товарища].

#### 3.4 Настройка параметров сервиса
Сервис конфигурируется через переменные окружения (env-файлы) или Kubernetes Secrets в соответствии с ТЗ (раздел 3.6.2). Ключевые параметры:

- **DB_URL**: Строка подключения к PostgreSQL (пример: `postgresql://user:pass@host:port/db`). Обязательный для хранения данных.
- **SHP_PATH**: Путь к файлу SHP для геопривязки (по умолчанию: `./data/rf.shp`). Рекомендуется обновлять ежемесячно.
- **LOG_LEVEL**: Уровень логирования (DEBUG, INFO, ERROR; по умолчанию: INFO). Логи хранятся в app.log.
- **MAX_WORKERS**: Количество потоков для обработки данных (по умолчанию: 4). Для больших наборов данных (>50k записей) увеличить до 8.
- **API_HOST/PORT**: Хост и порт для Flask (по умолчанию: 0.0.0.0:5000).
- **FRONTEND_URL**: URL фронтенда для CORS (пример: `https://<ip>.nip.io`).

Для локальной настройки создайте .env файл в корне репозитория и загрузите его в `docker-compose.yml` или Kubernetes manifests. При изменении параметров перезапустите контейнеры.

#### 3.5 Фронтенд: Установка и настройка
Фронтенд реализован на Vue.js с использованием Vue Router и Axios для взаимодействия с бэкендом, в соответствии с требованиями ТЗ к UX/UI (раздел 3.6.5). Он предоставляет интуитивный интерфейс для визуализации метрик, графиков и экспорта данных.

- **Требования**: Node.js 18+, npm/yarn.
- **Установка локально**:
- Перейдите в директорию frontend: `cd frontend`.
- Установите зависимости: `npm install`.
- Запустите в режиме разработки: `npm run serve` (доступ по http://localhost:8080).
- **Сборка для продакшена**: `npm run build` (результат в dist/).
- **Интеграция с бэкендом**: В src/config.js укажите API_BASE_URL (пример: `http://localhost:5000/api`). Для продакшена используйте переменные окружения в .env.vue.
- **Docker-сборка**: Dockerfile в frontend/ собирает статические файлы и обслуживает через Nginx. В docker-compose.yml фронтенд запускается как отдельный сервис на порту 80.
- **Функциональность**: Дашборд с таблицами метрик, интерактивными графиками (Chart.js для bar/line charts), формой загрузки XLSX/JSON, фильтрами по году/региону. Авторизация через Keycloak (OpenID Connect, настройка в auth.js).

Для развертывания в Yandex Cloud примените YAML для frontend Deployment/Service, аналогично бэкенду, с типом Service LoadBalancer для внешнего доступа.

## 4. Приложения

### Скриншоты
- /metrics output (JSON с flight_count).
- /report/graph?type=top_regions (PNG).
- Swagger UI (/swagger).

### Тест-отчет
- htmlcov/index.html (покрытие 80% после добавления тестов).

### Исходный код
- api.py (core).
- docker-compose.yml (настройка).
- tests/ (pytest).

## 5. Функциональная и компонентная архитектура в Archi
Микросервисы (frontend, backend, db), потоки данных (upload → parse → store → metrics), компоненты (Ingress → Services → Pods). Диаграммы в формате ArchiMate/C4 Model:

- **Контекстный уровень (C1)**: Система взаимодействует с пользователем через фронтенд, получает данные от Росавиации via webhook/upload, хранит в DB, генерирует отчеты.
- **Контейнерный уровень (C2)**: Фронтенд (Vue.js/Nginx), Бэкенд (Flask), БД (PostgreSQL/PostGIS).
- **Компонентный уровень (C3)**: В бэкенде - модули парсинга (parser.py), геопривязки (geo.py), метрик (metrics.py); во фронтенде - компоненты Dashboard.vue, Charts.vue.
- **Кодовый уровень (C4)**: Классы и функции, такие как FlightParser, RegionMapper.

Диаграммы доступны в ./archi/ (файлы .archimate и экспорт в PNG).

## 6. Руководство пользователя
Сервис предназначен для аналитиков и операторов Росавиации, предоставляя удобный доступ к данным о полетах БПЛА. В соответствии с ТЗ (раздел 3.5), интерфейс интуитивен, с фокусом на визуализацию и экспорт.

### Авторизация
- Перейдите на главную страницу фронтенда[](https://<ip>.nip.io).
- Войдите через Keycloak (логин/пароль или OpenID Connect). Доступны роли: admin (полный доступ), analyst (только просмотр).

### Загрузка данных
- В разделе "Загрузка" выберите XLSX-файл (формат как в ТЗ: sheet 'Result_1' с SHR/IDEP/IARR) или укажите JSON в webhook-форме.
- Нажмите "Загрузить". Процесс парсинга отображается в прогресс-баре; ошибки логируются и показываются в алертах.

### Просмотр метрик и графиков
- В дашборде выберите период (год/месяц) и регион через фильтры.
- Базовые метрики: Количество полетов, суммарная/средняя длительность (таблица с сортировкой).
- Расширенные: Плотность полетов, почасовая распределение, дни без полетов (визуализация в Chart.js: бар-чарты для топ-регионов, линейные графики для трендов).
- Экспорт: Кнопка "Экспорт JSON/PNG" скачивает отчеты on-demand.

### API-доступ (для интеграции)
- Используйте Swagger UI (/api/swagger) для тестирования эндпоинтов.
- Примеры: GET /metrics?year=2025&region=Москва (возвращает JSON с метриками); POST /upload (multipart/form-data с файлом).

Рекомендации: Для больших наборов данных (>10k) используйте асинхронный webhook. Мониторьте логи для отладки. Обновляйте SHP-файлы вручную для точной геопривязки. Если возникнут вопросы, обращайтесь по контактам команды.