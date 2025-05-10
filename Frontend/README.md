# Albus Frontend

This directory contains the frontend application for Albus, a node-based AI workflow tool. It provides a user-friendly interface to create, connect, and manage AI processing nodes.

## Technologies Used

*   **React**: For building the user interface.
*   **Vite**: As the build tool and development server, providing a fast development experience.
*   **React Flow**: For rendering and managing the interactive node graph.
*   **JavaScript (ESM)**: With ESLint for code quality.

## Setup

1.  **Navigate to this directory**:
    ```bash
    cd Frontend
    ```

2.  **Install Dependencies**:
    Ensure you have Node.js and npm installed. Then, run:
    ```bash
    npm install
    ```
    This will install all the necessary project dependencies defined in `package.json`.

## Running the Development Server

Once setup is complete, run the development server:

```bash
npm run dev
```

This will start the Vite development server, typically available at:
*   **Local:** `http://localhost:5173` (or the next available port if 5173 is in use).

The server supports Hot Module Replacement (HMR) for a fast development feedback loop.

## Connecting to the Backend

The frontend application is configured to connect to the backend API, which is expected to be running at `http://localhost:8000` by default. This URL is typically defined in service files like `src/services/api.js` or `src/services/NodeService.js`.

Ensure the backend server is running for full application functionality.

## Building for Production

To create a production build of the frontend, run:

```bash
npm run build
```

This will generate optimized static assets in the `dist` directory, which can then be deployed to a static hosting service.

## Linting

To check for code quality and style issues, run:

```bash
npm run lint
```
