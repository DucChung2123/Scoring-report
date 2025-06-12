# ESG Scoring API System

A comprehensive ESG (Environmental, Social, Governance) text analysis system that provides both RESTful API endpoints and an interactive Streamlit UI for analyzing documents and text content.

## 🌟 Features

- **ESG Factor Scoring**: Score text content for Environmental, Social, and Governance relevance
- **Sub-factor Classification**: Classify text into specific ESG sub-categories
- **Batch Processing**: Process multiple texts efficiently
- **Interactive UI**: Streamlit web interface for PDF document analysis
- **Modular Architecture**: Clean, maintainable codebase with separated concerns
- **Docker Support**: Easy deployment with Docker and Docker Compose

## 📋 Table of Contents

- [System Requirements](#system-requirements)
- [Installation](#installation)
  - [Option 1: Docker Deployment (Recommended)](#option-1-docker-deployment-recommended)
  - [Option 2: Python Development Setup](#option-2-python-development-setup)
- [Usage](#usage)
  - [API Endpoints](#api-endpoints)
  - [Streamlit UI](#streamlit-ui)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

## 🛠️ System Requirements

- **Docker & Docker Compose** (for containerized deployment)
- **Python 3.9+** (for development setup)
- **CUDA-compatible GPU** (optional, for faster inference)
- **4GB+ RAM** (for model loading)

## 🚀 Installation

### Option 1: Docker Deployment (Recommended)

This is the easiest way to get started. Both API and UI will be available with a single command.

#### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Scoring-report
   ```

2. **Ensure model files are in place**
   ```bash
   # Models should be in the models/ directory
   ls models/
   ```

3. **Start all services**
   ```bash
   docker-compose up --build
   ```

4. **Access the services**
   - **API**: http://localhost:8000
   - **Interactive UI**: http://localhost:8501
   - **API Documentation**: http://localhost:8000/docs

#### Docker Commands

```bash
# Build and start services
docker-compose up --build

# Start services in background
docker-compose up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs -f

# Rebuild specific service
docker-compose build esg-api
docker-compose build esg-ui
```

### Option 2: Python Development Setup

For development and customization, you can run the services directly with Python.

#### Prerequisites

1. **Install Python dependencies**
   ```bash
   # Install API dependencies
   pip install -r requirements.txt
   
   ```

2. **Set up environment**
   ```bash
   # Copy environment file
   cp .env.example .env  # if available
   
   # Ensure models directory exists
   mkdir -p models/
   ```

#### Running the API Server
Select Python path

```bash
export PYTHONPATH=$(pwd)
python src/api/app.py
```

#### Running the Streamlit UI

```bash
# Start the Streamlit interface
streamlit run src/UI/demo_streamlit_new_api_fixed.py 

```

## 📖 Usage

### API Endpoints

Once the API is running, you can interact with it using HTTP requests or the interactive documentation.

#### Health Check

```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "status": "alive",
  "message": "ESG Model API is running!"
}
```

#### Score Single Text

```bash
curl -X POST "http://localhost:8000/score" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Our company reduced carbon emissions by 25% this year",
    "factor": "E"
  }'
```

**Response:**
```json
{
  "score": 0.85,
}
```

#### Classify Sub-factor

```bash
curl -X POST "http://localhost:8000/classify_sub_factor" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "We implemented new diversity and inclusion programs"
  }'
```

**Response:**
```json
{
  "factor": "S",
  "sub_factor": "Diversity",
}
```

#### Batch Processing

```bash
curl -X POST "http://localhost:8000/score_batch" \
  -H "Content-Type: application/json" \
  -d '[
    {"text": "Renewable energy initiatives", "factor": "E"},
    {"text": "Employee training programs", "factor": "S"}
  ]'
```

### Streamlit UI

The Streamlit interface provides an intuitive way to analyze PDF documents:

1. **Access the UI** at http://localhost:8501
2. **Upload a PDF** document using the sidebar
3. **Configure API URL** (default: http://localhost:8000)
4. **Click "Analyze ESG Data"** to start processing
5. **View results** organized by E, S, G categories
6. **Download results** as JSON for further analysis

#### UI Features

- **PDF Text Extraction**: Automatically extracts text from uploaded PDFs
- **Real-time Progress**: Shows classification and scoring progress
- **Categorized Results**: Organizes findings by ESG factors and sub-factors
- **Score Visualization**: Color-coded scoring with relevance metrics
- **Export Functionality**: Download analysis results as JSON

## 📚 API Documentation

### Endpoints Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/score` | POST | Score single text for ESG factor |
| `/classify_sub_factor` | POST | Classify text into ESG sub-factors |
| `/score_batch` | POST | Score multiple texts |
| `/classify_sub_factor_batch` | POST | Classify multiple texts |

### Request/Response Schemas

#### ScoreRequest
```json
{
  "text": "string",
  "factor": "E" | "S" | "G"
}
```

#### ClassifyRequest
```json
{
  "text": "string"
}
```

#### ScoreResponse
```json
{
  "score": 0.85,
  "factor": "E",
  "text": "input text"
}
```

#### ClassifyResponse
```json
{
  "factor": "S",
  "sub_factor": "Diversity",
  "text": "input text"
}
```

### ESG Categories

#### Environmental (E)
- **Emission**: Climate change, carbon footprint
- **Resource Use**: Water, energy, waste management
- **Product Innovation**: Sustainable products, green technology

#### Social (S)
- **Community**: Community relations, social impact
- **Diversity**: Diversity and inclusion initiatives
- **Employment**: Employment practices, workplace conditions
- **HS**: Health and safety programs
- **HR**: Human rights policies
- **PR**: Product responsibility, customer safety
- **Training**: Employee development, education

#### Governance (G)
- **BFunction**: Board function and effectiveness
- **BStructure**: Board structure and composition
- **Compensation**: Executive compensation policies
- **Shareholder**: Shareholder rights and relations
- **Vision**: Corporate vision and strategy

## 🏗️ Project Structure

```
Scoring-report/
├── src/
│   ├── api/                    # FastAPI application
│   │   ├── app.py             # Main application entry
│   │   ├── ai_models/         # Model management
│   │   ├── routers/           # API endpoints
│   │   ├── schemas/           # Request/response models
│   │   ├── services/          # Business logic
│   │   └── utils/             # Utility functions
│   ├── UI/                    # Streamlit interface
│   │   ├── demo_streamlit_new_api_fixed.py
│   └── finetune/              # Model training utilities
├── models/                    # cached models
├── settings/                  # Configuration files
├── Dockerfile                 # Container definition
├── docker-compose.yml         # Service orchestration
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```


## 🛠️ Development

### Adding New Features

1. **API Endpoints**: Add new routers in `src/api/routers/`
2. **Business Logic**: Implement services in `src/api/services/`
3. **Data Models**: Define schemas in `src/api/schemas/`
4. **UI Components**: Modify `src/UI/demo_streamlit_new_api_fixed.py`

### Model Management

Models are loaded automatically when the API starts. Ensure model files are present in the `models/` directory with the expected structure.

- **GPU Usage**: Ensure CUDA is properly installed for GPU acceleration
- **Batch Size**: Adjust batch sizes based on available memory
- **Concurrent Requests**: Configure uvicorn workers for higher throughput

```bash
# Run with multiple workers
uvicorn src.api.app:app --workers 4 --host 0.0.0.0 --port 8000
```

**Created by**: HDChung, AI engineer
**Last Updated**: June 2025
