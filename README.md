# Truth Shield 🛡️

**Truth Shield** is an advanced **Agentic Threat Intelligence & Fake News Detection platform**, initially developed under the codename *VIGIL-AI*. It features a multi-layered verification system that seamlessly balances fine-tuned Machine Learning models with live web verification provided by AI Agents.

Our ecosystem is designed for structural linguistic analysis of text (fake news detection) through state-of-the-art machine learning algorithms and reasoning engines.

---

## 🔥 Key Features & Capabilities

### 1. Fake News & Threat Intelligence Analysis
- **RoBERTa Linguistic Backbone**: A custom-trained Natural Language Processing pipeline that captures nuances in fake news and malicious text.
- **Agentic Live Web Verification**: A dedicated AI agent actively queries live sources to verify facts, balancing synthetic reasoning (40%) with live web verification (60%).
- **Granular Controls**: Granular chat analysis interface allowing users to dynamically toggle between the custom-trained static model, the live web-aware AI agent, or a combination of both.

---

## 🏗️ Technology Stack

| Component         | Technology |
|-------------------|------------|
| **Frontend**      | Svelte, Vite, Vercel/Node |
| **Backend**       | Python, FastAPI, Uvicorn, WebSockets |
| **Machine Learning** | PyTorch, HuggingFace (RoBERTa), NLTK |
| **LLMs / Agents** | Ollama, Nvidia Gemma API |
| **Data Processing**| Pandas, Scikit-Learn |

---

## 📂 Project Structure

```bash
AIML-PROJECT-CSET312/
│
├── backend/                  # FastAPI backend server
│   ├── main.py               # Main API & WebSocket endpoints
│   ├── schema.py             # Pydantic validation models
│   └── requirements.txt      # Python dependencies
│
├── frontend/                 # Svelte-based Web Application
│   ├── src/                  # Svelte components & logic
│   ├── public/               # Static assets (including Truth Shield logo)
│   └── package.json          # Node.js dependencies
│
├── Datasets/                 # Local data sources (Kaggle, FakeNewsCorpus)
├── Notebooks/                # Jupyter Notebooks for data gathering & preprocessing
├── Model_desgin_vX/          # Historical architecture design prototypes
└── project_structure.txt     # Complete directory manifest
```

---

## ⚙️ Detailed Installation & Setup

### 1. Prerequisites
Ensure you have the following installed on your machine:
- **Python 3.10+**
- **Node.js 18+** & NPM
- **Git**

### 2. Backend Environment (FastAPI)

We recommend using `uv` or `venv` to isolate the backend environment. 

```bash
# Clone the repository
git clone <your-repo-url>
cd AIML-PROJECT-CSET312

# Navigate to the backend directory
cd backend

# Create and activate a Virtual Environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

# Install strictly required dependencies
pip install -r requirements.txt
```

#### Environment Variables Config (`.env`)
You must configure your `.env` file in the `/backend` directory before running. Do not commit credentials to Git.
```env
# Example .env configuration
NVIDIA_API_KEY=your_gemma_api_key
OLLAMA_HOST=http://localhost:11434
```

#### Run the Backend Server
```bash
uvicorn main:app --reload --port 8000
```
*The FastAPI backend will now be actively listening on `http://localhost:8000`. WebSocket streams connect on `ws://localhost:8000`.*

### 3. Frontend Environment (Svelte)

The frontend handles live video rendering, real-time object detection bounding boxes, and the chat analysis interface.

```bash
# Open a new terminal and navigate to the frontend
cd AIML-PROJECT-CSET312/frontend

# Install Node dependencies
npm install

# Start the Vite development server
npm run dev
```
*The dashboard will be accessible via your browser at the URL provided by Vite (usually `http://localhost:5173` or similar).*

---

## 📓 Model Training & Data Preparation

If you intend to adjust the base models, work happens within the `Notebooks/` directory.

- **`data_gathering.ipynb`**: Responsible for API connections and raw data ingestion.
- **`Preprocessing.ipynb`**: Responsible for NLTK tokenization, removing stop-words, scaling, and preparing unified CSVS for custom training pipelines.

---

## 📜 Legal & License

This project is licensed under the **MIT License** (see `LICENSE` file for details). Please note that datasets obtained from Kaggle or external sources are subject to their respective proprietary licensing.
