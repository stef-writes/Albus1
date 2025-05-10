# Basic-LLM-Chain2 - AI Workflow Orchestration Tool

Basic-LLM-Chain2 is a powerful, node-based AI workflow tool designed to help you create, visualize, and execute complex AI processing chains with a user-friendly interface. It allows for modular construction of AI tasks, making it easier to experiment with and deploy sophisticated AI pipelines.

## Project Structure

The project is organized into two main components:

*   **`Frontend/`**: Contains the React-based user interface. Users interact with this part to build and manage their AI workflows. It utilizes Vite for a fast development experience and React Flow for the graph visualization.
*   **`Backend/`**: A FastAPI server that powers the AI graph execution engine. It handles node processing, LLM interactions, context management between nodes, and data persistence using MongoDB.

For detailed setup, development, and running instructions for each component, please refer to their respective README files:

*   [Frontend README](./Frontend/README.md)
*   [Backend README](./Backend/README.md)

## Quick Start

To get Basic-LLM-Chain2 up and running quickly, follow these general steps:

1.  **Prerequisites:**
    *   Node.js and npm (for the Frontend)
    *   Python 3.8+ and pip (for the Backend)
    *   MongoDB instance running (locally or accessible via URL)
    *   OpenAI API Key

2.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd Basic-LLM-Chain2  # Or your repository's root directory name
    ```

3.  **Setup & Run Backend:**
    *   Navigate to the backend: `cd Backend`
    *   Follow instructions in `Backend/README.md` (create virtual environment, install requirements, set up `.env` file with API keys and MongoDB URL).
    *   Run the backend server: `uvicorn main:app --reload --port 8000` (from within the `Backend` directory).

4.  **Setup & Run Frontend:**
    *   Open a new terminal.
    *   Navigate to the frontend: `cd Frontend`
    *   Follow instructions in `Frontend/README.md` (install dependencies).
    *   Run the frontend development server: `npm run dev`.
    *   Open the URL provided by Vite (usually `http://localhost:5173`) in your browser.

## Core Features

*   **Visual Node-Based Workflow:** Intuitively design complex AI pipelines.
*   **Modular AI Components:** Define and reuse different types of processing nodes (LLM calls, data retrieval, custom logic).
*   **FastAPI Backend:** Robust and efficient server for managing and executing workflows.
*   **React Frontend:** Modern and responsive user interface.
*   **MongoDB Integration:** Persist your workflow designs and node configurations.
*   **Real-time Interaction:** (Depending on specific frontend features) Immediate feedback during workflow construction.

## License

MIT 