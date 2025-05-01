# Backend (FastAPI - ScriptChain)

This directory contains the Python backend for the AI graph execution engine, built with FastAPI.

It provides an API for:
*   Defining graph nodes (`/add_node`)
*   Connecting nodes with edges (`/add_edge`)
*   Executing a single text generation node (`/generate_text_node`)
*   Executing an entire defined graph (`/execute` - potentially less used with the new UI flow)

## Setup

1.  **Navigate to this directory**:
    ```bash
    cd backend
    ```

2.  **Create Virtual Environment** (Recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables**:
    *   Create a `.env` file in this directory (you can copy `.env.example` if it exists, or create it manually).
    *   Add your OpenAI API key to the `.env` file:
      ```env
      OPENAI_API_KEY='sk-...' 
      ```

## Running the Server

Once setup is complete, run the development server from within the `backend` directory:

```bash
uvicorn main:app --reload --port 8000
```

*   `uvicorn`: The ASGI server.
*   `main:app`: Tells Uvicorn to find the `app` object inside the `main.py` file.
*   `--reload`: Automatically restarts the server when code changes are detected.
*   `--port 8000`: Specifies the port to run on.

The API server should now be running at `http://127.0.0.1:8000`.
You can access the interactive API documentation (Swagger UI) at `http://127.0.0.1:8000/docs`.

## Core Components

*   **`main.py`**: Contains the FastAPI application, API endpoints, and core logic.
*   **`LLMConfig`**: Class for configuring AI model parameters.
*   **`MessageTemplate`, `PromptTemplate`**: Classes for handling prompt structures.
*   **`Node`**: Class representing a single processing step in the graph.
*   **`ScriptChain`**: Class using `networkx` to manage and execute the graph of nodes.
*   **`Callback`, `LoggingCallback`**: System for observing graph execution events.
*   **`track_token_usage`**: Context manager for logging OpenAI token usage.
*   **AI Functions (`generate_text`, etc.)**: Functions performing specific AI tasks called by `Node.process`. 