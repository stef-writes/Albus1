# Backend (FastAPI - ScriptChain)

This directory contains the Python backend for the AI graph execution engine, built with FastAPI.

It provides an API for:
*   Defining graph nodes (`/add_node`)
*   Connecting nodes with edges (`/add_edge`)
*   Executing a single text generation node (`/generate_text_node`)
*   Executing an entire defined graph (`/execute`)
*   Managing and retrieving node/chain data from MongoDB.

## Setup

1.  **Navigate to this directory**:
    ```bash
    cd Backend
    ```

2.  **Create Virtual Environment** (Recommended):
    ```bash
    python3 -m venv venv  # Or python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables**:
    *   Create a `.env` file in this `Backend` directory (you can copy `.env.example` if it exists, or create it manually).
    *   Add your OpenAI API key and MongoDB URL (if not localhost) to the `.env` file:
      ```env
      OPENAI_API_KEY='your_openai_api_key_here'
      MONGODB_URL='mongodb://localhost:27017' # Or your remote MongoDB URL
      ```

## Running the Server

Once setup is complete, ensure you are in the `Backend` directory, then run the development server:

```bash
uvicorn main:app --reload --port 8000
```

*   `uvicorn`: The ASGI server.
*   `main:app`: Tells Uvicorn to find the `app` object (FastAPI instance) inside the `main.py` file.
*   `--reload`: Automatically restarts the server when code changes are detected (useful for development).
*   `--port 8000`: Specifies the port to run on (default is 8000).

The API server should now be running at `http://127.0.0.1:8000`.
You can access the interactive API documentation (Swagger UI) at `http://127.0.0.1:8000/docs`.

## Core Components & Structure

The backend follows a modular structure:

*   **`main.py`**: Contains the FastAPI application instance, API route definitions, and glues the components together.
*   **`llm.py`**: Manages LLM (Large Language Model) configurations (like `LLMConfig`), the OpenAI client setup, and token usage tracking (`track_token_usage`).
*   **`models.py`**: Defines core Pydantic models used across the application for data validation and serialization, especially for API request/response bodies related to defining nodes, edges, etc.
*   **`script_chain.py`**: Implements the core graph execution logic. Includes the `Node` class (representing a single processing step) and the `ScriptChain` class (using `networkx` to manage and execute the graph of nodes).
*   **`callbacks.py`**: Provides a callback system (`Callback`, `LoggingCallback`) for observing and reacting to events during graph execution (e.g., node start/complete).
*   **`templates.py`**: Handles prompt templating logic (`TemplateProcessor`) for dynamic prompt construction using context data.
*   **`utils.py`**: Contains utility classes and functions, such as `ContentParser` for extracting structured data from LLM outputs and `InputValidator`.
*   **`database.py`**: Manages interactions with MongoDB using `motor` for asynchronous operations. Includes functions for saving and retrieving nodes and chains.

This modular design aims for better maintainability, testability, and separation of concerns. 