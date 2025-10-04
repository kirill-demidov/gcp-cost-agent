# GCP Cost Agent 💰

Agent for analyzing Google Cloud Platform costs with Russian and English language support.

## Features

- 📊 **Cost Analysis**
  - Monthly total costs
  - Project breakdown
  - Service breakdown
  - Period comparison
  - Trend analysis
  - Seasonality analysis

- 🌍 **Language Support**
  - Russian
  - English

- 🤖 **Advanced Analytics**
  - Anomaly detection
  - Cost optimization
  - Benchmarking
  - Forecasting
  - Efficiency analysis

- ☁️ **Deployment Ready**
  - Cloud Run deployment
  - Docker containerization
  - Health checks
  - Auto-scaling

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI       │    │   MCP Toolbox   │
│   (React)       │◄──►│   Backend       │◄──►│   + ADK         │
│                 │    │   (Python)      │    │   (Tools)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Cloud Run     │    │   BigQuery      │
                       │   (Container)   │    │   (Billing)     │
                       └─────────────────┘    └─────────────────┘
```

### Architecture Diagrams

The project includes PlantUML sequence diagrams in the `docs/` folder:

- `docs/architecture.puml` - Overall system architecture and flows
- `docs/detailed-flow.puml` - Detailed cost analysis request flow
- `docs/deployment.puml` - Cloud Run deployment process

To view the diagrams:
1. Install PlantUML: `brew install plantuml` (Mac) or download from [plantuml.com](https://plantuml.com/)
2. Generate images: `plantuml docs/*.puml`
3. Or use online viewer: [PlantUML Online Server](http://www.plantuml.com/plantuml/uml/)

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Docker
- gcloud CLI
- GCP project with enabled APIs:
  - BigQuery API
  - Vertex AI API
  - Cloud Build API
  - Cloud Run API
- Billing data export to BigQuery

### 2. Installation

```bash
# Clone repository
git clone <repository-url>
cd gcp-cost-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Edit `tools.yaml` and replace:

- `${GCP_PROJECT_ID}` → your GCP Project ID
- `${BILLING_TABLE}` → full billing table name

Table format: `project-id.dataset.gcp_billing_export_v1_XXXXX`

### 4. Install MCP Toolbox

Download the MCP Toolbox binary:

```bash
# Download toolbox (replace with latest version)
curl -L https://github.com/modelcontextprotocol/toolbox/releases/latest/download/toolbox-darwin-arm64 -o toolbox
chmod +x toolbox
```

### 5. Local Run

```bash
# Start MCP Toolbox
./toolbox

# In separate terminal
python api/main.py
```

Web interface will be available at `http://localhost:8080`

## Deploy to Cloud Run

### Automatic Deployment

```bash
# Set Google API Key
export GOOGLE_API_KEY=your-google-api-key

# Run deployment
GCP_PROJECT_ID=your-project-id \
BILLING_TABLE=your-project.billing.gcp_billing_export_v1_XXXXX \
./deploy.sh
```

### Manual Deployment

```bash
# 1. Build Docker image
gcloud builds submit --tag gcr.io/$GCP_PROJECT_ID/gcp-cost-agent

# 2. Deploy to Cloud Run
gcloud run deploy gcp-cost-agent \
  --image gcr.io/$GCP_PROJECT_ID/gcp-cost-agent \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "GCP_PROJECT_ID=$GCP_PROJECT_ID,BILLING_TABLE=$BILLING_TABLE,GOOGLE_API_KEY=$GOOGLE_API_KEY" \
  --memory 1Gi \
  --cpu 1 \
  --timeout 300 \
  --min-instances 0 \
  --max-instances 10
```

## API Endpoints

### Health Check
```bash
GET /health
```

### Chat Endpoint
```bash
POST /chat
Content-Type: application/json

{
  "question": "What were my total costs in September 2025?",
  "session_id": "unique-session-id"
}
```

## Usage Examples

### English Language

```bash
# Total costs
curl -X POST https://your-service-url/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "What were my total costs in September 2025?", "session_id": "test"}'

# By projects
curl -X POST https://your-service-url/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "Show me project spending for August 2025", "session_id": "test"}'

# By services
curl -X POST https://your-service-url/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "Which services were most expensive in September 2025?", "session_id": "test"}'

# Comparison
curl -X POST https://your-service-url/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "Compare costs between July and August 2025", "session_id": "test"}'

# Trends
curl -X POST https://your-service-url/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "Show me spending trends for 2025", "session_id": "test"}'

# Optimization
curl -X POST https://your-service-url/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "Where can I optimize my costs?", "session_id": "test"}'

# Anomalies
curl -X POST https://your-service-url/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "Find anomalies in my spending", "session_id": "test"}'

# Seasonality
curl -X POST https://your-service-url/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "Is there seasonality in my costs?", "session_id": "test"}'

# Forecasting
curl -X POST https://your-service-url/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "Predict my costs for next month", "session_id": "test"}'
```

### Russian Language

```bash
# Общие затраты / Total costs
curl -X POST https://your-service-url/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "Какие были общие затраты в сентябре 2025?", "session_id": "test"}'

# По проектам / By projects
curl -X POST https://your-service-url/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "Покажи разбивку затрат по проектам за август 2025", "session_id": "test"}'

# По сервисам / By services
curl -X POST https://your-service-url/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "Какие сервисы были самыми дорогими в сентябре 2025?", "session_id": "test"}'

# Сравнение / Comparison
curl -X POST https://your-service-url/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "Сравни расходы за июль и август 2025", "session_id": "test"}'

# Тренды / Trends
curl -X POST https://your-service-url/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "Покажи тренд расходов за 2025 год", "session_id": "test"}'

# Оптимизация / Optimization
curl -X POST https://your-service-url/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "Где можно сэкономить на расходах?", "session_id": "test"}'

# Аномалии / Anomalies
curl -X POST https://your-service-url/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "Найди аномалии в расходах", "session_id": "test"}'
```

## Project Structure

```
gcp-cost-agent/
├── docs/                   # Documentation and diagrams
│   ├── architecture.puml   # System architecture diagram
│   ├── detailed-flow.puml  # Detailed request flow
│   └── deployment.puml     # Deployment process
├── api/
│   ├── main.py             # FastAPI backend (local)
│   └── cloud_api.py        # FastAPI backend (Cloud Run)
├── agents/
│   └── gcp_cost_agent/
│       └── agent.py        # ADK Agent logic
├── frontend/
│   └── index.html          # React frontend
├── tools.yaml              # MCP Toolbox configuration
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker image
├── cloudbuild.yaml        # Cloud Build configuration
├── deploy.sh              # Deployment script
├── .gitignore             # Git ignore rules
├── env.example            # Environment variables example
└── README.md              # This file
```

## Environment Variables

Copy `env.example` to `.env` and configure:

```bash
# GCP Configuration
GCP_PROJECT_ID=your-project-id
BILLING_TABLE=your-project.billing.gcp_billing_export_v1_XXXXX

# Google API Key for Gemini
GOOGLE_API_KEY=your-google-api-key

# Toolbox Configuration
TOOLBOX_URL=http://127.0.0.1:5001

# Vertex AI Configuration
VERTEXAI_PROJECT=your-project-id
VERTEXAI_LOCATION=us-central1

# Server Configuration
PORT=8080
```

## Troubleshooting

### Problem: Agent cannot connect to BigQuery

**Solution:**
- Ensure BigQuery API is enabled
- Check `BILLING_TABLE` is correct
- Verify access permissions

### Problem: Empty query results

**Solution:**
- Check `invoice_month` format (should be YYYYMM)
- Ensure data exists for the requested period
- Verify data exists in BigQuery table

### Problem: Cloud Run deployment errors

**Solution:**
- Ensure all APIs are enabled
- Check project quotas
- Check logs: `gcloud run logs read gcp-cost-agent --region=us-central1`

### Problem: Queries not recognized

**Solution:**
- Use GCP-related keywords
- Rephrase the query
- Check examples above

## Security

- All API keys are passed via environment variables
- Sensitive data is not stored in code
- Use IAM roles with minimal permissions

## License

MIT

## Author

Based on the article by Aryan Irani  
Adapted for Cloud Run deployment with Russian language support and advanced analytics