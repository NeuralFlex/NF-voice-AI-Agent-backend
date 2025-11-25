
# 🧠 Interviewer Chatbot (Web Version)

An **AI-driven technical interviewer** powered by **LangGraph**, served through a **Streamlit web interface** and a **FastAPI backend**.  
It dynamically generates and evaluates interview questions using **Gemini** and **Tavily APIs**, manages conversation flow with **LangGraph**, and persists state using a **SQLite-based checkpoint system**, allowing interviews to resume seamlessly between interactions.


---

## 📘 Table of Contents

1. [Core Components](#1-core-components)  
2. [Installation](#2-installation)  
   - [2.1 Prerequisites](#21-prerequisites)  
   - [2.2 Environment Configuration](#22-environment-configuration)  
   - [2.3 Setup Steps](#23-setup-steps)  
3. [Running the Application](#3-running-the-application)  
   - [3.1 Start the FastAPI Backend](#31-start-the-fastapi-backend)  
   - [3.2 Start the Streamlit Frontend](#32-start-the-streamlit-frontend)  
   - [3.3 Workflow Summary](#33-workflow-summary)

---

## 1. Core Components

### 1.1 Interview Graph
Implements the main interview logic as a **directed LangGraph**, where each node represents a stage of the interview process — setup, question generation, answer evaluation, retrieval, and final summary.

### 1.2 Stateful Interview Management
Maintains all session variables such as topic, current step, generated questions, and recorded responses throughout the interview session.  
State is **persisted using `SqliteSaver`**, allowing the interview to **resume exactly where it left off**.


### 1.3 AI & Knowledge Services
- **Gemini** powers question generation and response evaluation.  
- **Tavily** enriches reasoning with background knowledge and contextual references.

### 1.4 Vector Database & Embeddings
Integrates **vector similarity search** to enhance interview personalization:

- The candidate’s **CV is split into text chunks**.  
- Each chunk is **converted into an embedding vector** using an embedding model.  
- The candidate’s **answers are also embedded**, and their similarity to CV chunks is computed.  
- If the **distance is below a threshold** (e.g., 0.55), relevant CV chunks are retrieved.  
- Retrieved context is injected into the **question generation** and **evaluation** steps for more personalized reasoning.

This enables the chatbot to:
- Ask **personalized follow-up questions** grounded in the candidate’s experience.  
- **Evaluate answers** more accurately with background context.  
- Maintain **contextual coherence** throughout the interview.

### 1.5 Frontend (Streamlit)
The **Streamlit web app (`app.py`)** handles:
- CV upload  
- Job title input  
- Question type selection (`broad`, `narrow_up`, `follow_up`)  
- Real-time question and answer display  

It communicates with the backend through HTTP requests to FastAPI endpoints.

### 1.6 Backend (FastAPI)
The **FastAPI server (`main.py`)** handles:
- Interview initialization and continuation  
- LangGraph orchestration  
- State persistence using SQLite checkpoints  
- Integration with Gemini and Tavily services  

---

## 2. Installation

### 2.1 Prerequisites

Ensure the following are installed:

- Python 3.10+
- pip
- Virtual environment tool (`venv` or `virtualenv`)

---

### 2.2 Environment Configuration

Create a `.env` file in the project root containing your API keys and configuration values.  
Refer to the `.env_example` file included in the repository for required variables.

---

### 2.3 Setup Steps

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # On Linux/macOS
venv\Scripts\activate      # On Windows

# Install dependencies
pip install -r requirements.txt

3. Running the Application

You need to start both the FastAPI backend and the Streamlit frontend.
3.1 Start the FastAPI Backend

Run the backend, which hosts the interview logic and LangGraph state management:

```bash
python main.py
```

Or, using `uvicorn`:

```bash
uvicorn main:app --reload
```

This will start the backend at [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

### 3.2 Start the Streamlit Frontend

Run the Streamlit web application:

```bash
streamlit run app.py
```

This will open the UI in your browser at [http://localhost:8501](http://localhost:8501).

---

### 3.3 Workflow Summary

    User interacts with the Streamlit UI (uploads CV, selects topic/type).

    Streamlit sends data to the FastAPI backend.

    FastAPI invokes the LangGraph interview graph, managing flow and checkpoints via SqliteSaver.

    Updated interview state is saved and can be resumed later.

    Responses and new questions are streamed back to the Streamlit UI in real-time.

✅ With this setup, you have a fully persistent, web-based AI interviewer that uses LangGraph, Gemini, Tavily, and SQLite checkpoints to conduct context-aware technical interviews.
