#!/bin/bash

# Скрипт для деплоя GCP Cost Agent в Cloud Run
# Script to deploy GCP Cost Agent to Cloud Run

set -e

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Деплой GCP Cost Agent в Cloud Run ===${NC}"
echo -e "${GREEN}=== Deploying GCP Cost Agent to Cloud Run ===${NC}"
echo ""

# Проверка переменных окружения
if [ -z "$GCP_PROJECT_ID" ]; then
    echo -e "${RED}Ошибка: GCP_PROJECT_ID не установлен${NC}"
    echo -e "${RED}Error: GCP_PROJECT_ID is not set${NC}"
    echo "Использование / Usage: GCP_PROJECT_ID=your-project-id BILLING_TABLE=your.table ./deploy.sh"
    exit 1
fi

if [ -z "$BILLING_TABLE" ]; then
    echo -e "${RED}Ошибка: BILLING_TABLE не установлен${NC}"
    echo -e "${RED}Error: BILLING_TABLE is not set${NC}"
    echo "Использование / Usage: GCP_PROJECT_ID=your-project-id BILLING_TABLE=your.table ./deploy.sh"
    exit 1
fi

# Название сервиса
SERVICE_NAME="gcp-cost-agent"
REGION="${REGION:-us-central1}"

echo -e "${YELLOW}Проект / Project: $GCP_PROJECT_ID${NC}"
echo -e "${YELLOW}Регион / Region: $REGION${NC}"
echo -e "${YELLOW}Таблица биллинга / Billing table: $BILLING_TABLE${NC}"
echo ""

# 1. Включаем необходимые API
echo -e "${GREEN}[1/5] Включение необходимых API...${NC}"
echo -e "${GREEN}[1/5] Enabling required APIs...${NC}"
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    bigquery.googleapis.com \
    --project=$GCP_PROJECT_ID

# 2. Собираем Docker образ с помощью Cloud Build
echo -e "${GREEN}[2/5] Сборка Docker образа...${NC}"
echo -e "${GREEN}[2/5] Building Docker image...${NC}"

# Собираем с API ключом если он есть
if [ -n "$GOOGLE_API_KEY" ]; then
    gcloud builds submit \
        --config cloudbuild.yaml \
        --substitutions _GOOGLE_API_KEY="$GOOGLE_API_KEY" \
        --project=$GCP_PROJECT_ID \
        .
else
    gcloud builds submit \
        --config cloudbuild.yaml \
        --project=$GCP_PROJECT_ID \
        .
fi

# 3. Деплоим в Cloud Run
echo -e "${GREEN}[3/5] Деплой в Cloud Run...${NC}"
echo -e "${GREEN}[3/5] Deploying to Cloud Run...${NC}"

# Проверяем наличие GOOGLE_API_KEY
if [ -z "$GOOGLE_API_KEY" ]; then
    echo -e "${YELLOW}Предупреждение: GOOGLE_API_KEY не установлен${NC}"
    echo -e "${YELLOW}Warning: GOOGLE_API_KEY is not set${NC}"
    echo "Установите переменную: export GOOGLE_API_KEY=your-api-key"
fi

gcloud run deploy $SERVICE_NAME \
    --image gcr.io/$GCP_PROJECT_ID/$SERVICE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --set-env-vars "GCP_PROJECT_ID=$GCP_PROJECT_ID,BILLING_TABLE=$BILLING_TABLE,GOOGLE_API_KEY=${GOOGLE_API_KEY:-}" \
    --memory 1Gi \
    --cpu 1 \
    --timeout 300 \
    --min-instances 0 \
    --max-instances 10 \
    --project=$GCP_PROJECT_ID

# 4. Получаем URL сервиса
echo -e "${GREEN}[4/5] Получение URL сервиса...${NC}"
echo -e "${GREEN}[4/5] Getting service URL...${NC}"
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
    --region $REGION \
    --project=$GCP_PROJECT_ID \
    --format='value(status.url)')

# 5. Тестируем сервис
echo -e "${GREEN}[5/5] Тестирование сервиса...${NC}"
echo -e "${GREEN}[5/5] Testing service...${NC}"
sleep 10  # Ждем пока сервис полностью запустится

HEALTH_CHECK=$(curl -s "$SERVICE_URL/health")
echo -e "${YELLOW}Health check: $HEALTH_CHECK${NC}"

echo ""
echo -e "${GREEN}=== Деплой завершен успешно! ===${NC}"
echo -e "${GREEN}=== Deployment completed successfully! ===${NC}"
echo ""
echo -e "${YELLOW}URL сервиса / Service URL: $SERVICE_URL${NC}"
echo ""
echo -e "${YELLOW}Примеры использования / Usage examples:${NC}"
echo ""
echo "# Проверка здоровья / Health check:"
echo "curl $SERVICE_URL/health"
echo ""
echo "# Запрос о затратах (русский) / Cost query (Russian):"
echo "curl -X POST $SERVICE_URL/chat \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"question\": \"Какие были общие затраты в сентябре 2025?\", \"session_id\": \"test\"}'"
echo ""
echo "# Запрос о затратах (English) / Cost query (English):"
echo "curl -X POST $SERVICE_URL/chat \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"question\": \"What was my total bill in August 2025?\", \"session_id\": \"test\"}'"
echo ""
