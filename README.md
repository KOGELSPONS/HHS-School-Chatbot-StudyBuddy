# Minor Choice Chatbot

An interactive AI assistant that helps students choose an academic minor based on their interests, goals, and course preferences. The project provides a **Python backend** (chat logic) and a **Streamlit frontend**.

---

## Project structure

```
.
├── test_ai.py               #Backend entry point
├── front-end/
│   └── app.py           # Streamlit UI entry point
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Prerequisites

* **Python 3.10+** (Python 3.8+ may work, but 3.10+ recommended)

---

## Setup & Installation

### 1) Create and activate a virtual environment

```bash
python3 -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1
```

### 2) Upgrade pip and install dependencies

```bash
python3 -m pip install --upgrade pip

python3 -m pip install -r requirements.txt

```

---

## Backend: run & test

### Run the backend (if `app.py` is a script)

```bash
python3 test_ai.py
```

---

## Frontend (Streamlit)

Run the Streamlit application from the **project root**:

```bash
streamlit run front-end/app.py
```

* Streamlit will print a local URL (usually `http://localhost:8501`).
* If running on a remote server, pass `--server.address 0.0.0.0` and optionally `--server.port 8501`.

Examples:

```bash
# Local default
streamlit run front/end/app.py

# Remote-friendly
streamlit run front/end/app.py --server.address 0.0.0.0 --server.port 8501
```

---

## Configuration (optional)

If your chatbot uses API keys or model configs, set them via environment variables. For example:

```bash
AUTH_TOKEN="XXXXXXXXX"
```

On Windows (PowerShell):

```powershell
$env:OPENAI_API_KEY="your_key_here"
$env:MODEL_NAME="gpt-4o-mini"
```


---

## Quick Start (copy/paste)

```bash
# 1) Create venv
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1

# 2) Install
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt  # or minimally: python3 -m pip install streamlit

# 3) Backend (optional run/tests)
python3 test_ai.py           # run backend script

# 4) Frontend (Streamlit)
streamlit run front-end/app.py
```
