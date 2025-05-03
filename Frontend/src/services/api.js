const API_BASE_URL = 'http://127.0.0.1:8000'; // Your backend URL

// Function to search nodes by name
export const searchNodes = async (query) => {
  try {
    const response = await fetch(`${API_BASE_URL}/nodes/?q=${encodeURIComponent(query)}`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const nodes = await response.json();
    console.log('API searchNodes response:', nodes);
    return nodes; // Expects an array of nodes, e.g., [{ node_id: '...', ...}]
  } catch (error) {
    console.error('Error searching nodes:', error);
    return []; // Return empty array on error
  }
};

// Function to get full details of a single node by ID
export const getNodeDetails = async (nodeId) => {
  try {
    const response = await fetch(`${API_BASE_URL}/nodes/${encodeURIComponent(nodeId)}`);
    if (!response.ok) {
      if (response.status === 404) {
        console.warn(`Node not found: ${nodeId}`);
        return null; // Node not found is not necessarily a hard error
      }
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const node = await response.json();
    console.log('API getNodeDetails response:', node);
    return node; // Expects a single node object
  } catch (error) {
    console.error(`Error getting node details for ${nodeId}:`, error);
    return null; // Return null on error
  }
};

// Function to update the name of a node
export const updateNodeName = async (nodeId, newName) => {
  try {
    const response = await fetch(`${API_BASE_URL}/nodes/${encodeURIComponent(nodeId)}/name`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ name: newName }),
    });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const result = await response.json();
    console.log('API updateNodeName response:', result);
    return true; // Indicate success
  } catch (error) {
    console.error(`Error updating node name for ${nodeId}:`, error);
    return false; // Indicate failure
  }
}; 