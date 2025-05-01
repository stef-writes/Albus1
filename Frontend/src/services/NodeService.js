// NodeService.js - Handles API calls to the backend

const API_URL = 'http://localhost:8000';

// Create a unique session ID for this browser session
// In a production app, this might come from authentication or be stored in localStorage
const SESSION_ID = 'session_' + Math.random().toString(36).substring(2, 15);
console.log(`Using session ID: ${SESSION_ID}`);

/**
 * Sends a prompt to the backend API to generate text
 * @param {string} prompt - The prompt text
 * @param {object} config - Optional LLM config parameters
 * @param {object} contextData - Optional context data for the prompt (node outputs)
 * @returns {Promise<object>} - The API response with generated text
 */
export const generateText = async (prompt, config = null, contextData = null) => {
  try {
    console.log('Sending to backend - Prompt:', prompt);
    console.log('Sending to backend - Context Data:', contextData);
    
    const payload = {
      prompt_text: prompt,
      llm_config: config,
    };

    // If contextData is provided, add it to the payload
    if (contextData) {
      payload.context_data = contextData;
    }

    const response = await fetch(`${API_URL}/generate_text_node?session_id=${SESSION_ID}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Error generating text');
    }

    return response.json();
  } catch (error) {
    console.error('Error calling generateText API:', error);
    throw error;
  }
};

/**
 * Adds a node to the backend script chain
 * @param {string} nodeId - Unique identifier for the node
 * @param {string} nodeType - Type of node (e.g., 'text_generation')
 * @param {Array<string>} inputKeys - Input keys required by the node
 * @param {Array<string>} outputKeys - Output keys produced by the node
 * @param {object} config - Optional LLM config parameters
 * @returns {Promise<object>} - The API response
 */
export const addNode = async (nodeId, nodeType, inputKeys = [], outputKeys = [], config = null) => {
  try {
    const response = await fetch(`${API_URL}/add_node?session_id=${SESSION_ID}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        node_id: nodeId,
        node_type: nodeType,
        input_keys: inputKeys,
        output_keys: outputKeys,
        llm_config: config,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Error adding node');
    }

    return response.json();
  } catch (error) {
    console.error('Error calling addNode API:', error);
    throw error;
  }
};

/**
 * Adds an edge between two nodes in the backend script chain
 * @param {string} fromNode - Source node ID
 * @param {string} toNode - Target node ID
 * @returns {Promise<object>} - The API response
 */
export const addEdge = async (fromNode, toNode) => {
  try {
    const response = await fetch(`${API_URL}/add_edge?session_id=${SESSION_ID}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        from_node: fromNode,
        to_node: toNode,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Error adding edge');
    }

    return response.json();
  } catch (error) {
    console.error('Error calling addEdge API:', error);
    throw error;
  }
};

/**
 * Executes the script chain in the backend
 * @param {object} initialInputs - Optional initial inputs for the chain
 * @returns {Promise<object>} - The execution results
 */
export const executeChain = async (initialInputs = null) => {
  try {
    console.log('Executing chain with initial inputs:', initialInputs);
    
    const response = await fetch(`${API_URL}/execute?session_id=${SESSION_ID}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(initialInputs),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Error executing chain');
    }

    return response.json();
  } catch (error) {
    console.error('Error calling executeChain API:', error);
    throw error;
  }
};

/**
 * Validates a template against available nodes
 * @param {string} promptText - The template text to validate
 * @param {Array<string>} availableNodes - List of available node names
 * @returns {Promise<object>} - Validation results
 */
export const validateTemplate = async (promptText, availableNodes = []) => {
  try {
    const response = await fetch(`${API_URL}/validate_template?session_id=${SESSION_ID}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        prompt_text: promptText,
        available_nodes: availableNodes,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Error validating template');
    }

    return response.json();
  } catch (error) {
    console.error('Error calling validateTemplate API:', error);
    throw error;
  }
};

/**
 * Retrieves the latest output values for specified nodes
 * @param {Array<string>} nodeIds - Array of node IDs to fetch outputs for
 * @returns {Promise<object>} - Map of node IDs to their output values
 */
export const getNodeOutputs = async (nodeIds) => {
  try {
    // Since we don't have a dedicated endpoint for this yet,
    // this is a workaround to get node outputs from the backend's storage
    // In a production system, you'd implement a proper API endpoint
    
    // First try to get from the backend's storage using execute with empty input
    const response = await fetch(`${API_URL}/get_node_outputs?session_id=${SESSION_ID}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ node_ids: nodeIds }),
    });

    if (!response.ok) {
      // If the endpoint doesn't exist or fails, return empty object
      console.warn("get_node_outputs endpoint failed or doesn't exist, using local values");
      return {};
    }

    return response.json();
  } catch (error) {
    console.error('Error fetching node outputs:', error);
    return {}; // Return empty object on error
  }
};

// Export all functions as a service object
const NodeService = {
  generateText,
  addNode,
  addEdge,
  executeChain,
  validateTemplate,
  getNodeOutputs,
};

export default NodeService; 