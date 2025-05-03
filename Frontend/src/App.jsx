import React, { useState, useCallback, useRef } from 'react';
import ReactFlow, {
  Controls, // Adds zoom/pan controls
  Background, // Adds a background pattern
  applyNodeChanges, // Helper for node state updates
  applyEdgeChanges, // Helper for edge state updates
  addEdge, // Helper for adding edges
  ReactFlowProvider, // Import the provider
  Panel,
  useReactFlow,
} from 'reactflow';
import { v4 as uuidv4 } from 'uuid';

// Import React Flow CSS
import 'reactflow/dist/style.css';

// Import our App specific CSS
import './App.css';

// Import custom node types
import nodeTypes from './components/CustomNodeTypes';

// Import API service functions
import { searchNodes, getNodeDetails, updateNodeName } from './services/api'; // Import getNodeDetails and updateNodeName
import NodeService from './services/NodeService'; // Keep NodeService for other calls

// Import our components
import Node from './components/Node';
import DocumentPanel from './components/DocumentPanel';

function Flow() {
  const reactFlowWrapper = useRef(null);
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [reactFlowInstance, setReactFlowInstance] = useState(null);
  const [nodeOutputs, setNodeOutputs] = useState({});
  const [isConnecting, setIsConnecting] = useState(false);
  
  // Document panel state
  const [isDocumentPanelOpen, setIsDocumentPanelOpen] = useState(false);
  const [documents, setDocuments] = useState([
    { id: uuidv4(), name: 'Document 1', content: '' }
  ]);

  // Document handlers
  const handleDocumentSave = (id, updatedDoc) => {
    setDocuments(docs => 
      docs.map(doc => doc.id === id ? {...doc, ...updatedDoc} : doc)
    );
  };

  const handleAddDocument = () => {
    const newDoc = { 
      id: uuidv4(), 
      name: `Document ${documents.length + 1}`, 
      content: '' 
    };
    setDocuments([...documents, newDoc]);
  };

  // Handle selecting a node from search results
  const handleNodeSelect = async (nodeId) => {
    console.log(`Node selected from search: ${nodeId}`);
    try {
      const nodeDetails = await getNodeDetails(nodeId);
      if (nodeDetails) {
        console.log('Fetched node details:', nodeDetails);
        // Check if a document for this node already exists
        const existingDocIndex = documents.findIndex(doc => doc.id === nodeId);
        const nodeContent = nodeDetails.output || ''; // Use output field, default to empty

        if (existingDocIndex !== -1) {
          // Update existing document
          setDocuments(docs => 
            docs.map((doc, index) => 
              index === existingDocIndex ? { ...doc, name: nodeDetails.node_id, content: nodeContent } : doc
            )
          );
          console.log(`Updated existing document for node: ${nodeId}`);
        } else {
          // Add new document
          const newDocFromNode = { 
            id: nodeId, // Use nodeId as the document ID
            name: nodeDetails.node_id, 
            content: nodeContent
          };
          setDocuments(docs => [...docs, newDocFromNode]);
          console.log(`Added new document for node: ${nodeId}`);
        }

        // TODO: Optionally set this new/updated doc as active in DocumentPanel

        // Ensure panel is open
        setIsDocumentPanelOpen(true);

      } else {
        console.warn(`Could not find details for node: ${nodeId}`);
        // Handle node not found (e.g., show a message)
      }
    } catch (error) {
      console.error(`Error handling node selection for ${nodeId}:`, error);
      // Handle error (e.g., show a message)
    }
  };

  // Handler for node changes (drag, select, remove)
  const onNodesChange = useCallback(
    (changes) => setNodes((nds) => applyNodeChanges(changes, nds)),
    []
  );

  // Handler for edge changes (select, remove)
  const onEdgesChange = useCallback(
    (changes) => setEdges((eds) => applyEdgeChanges(changes, eds)),
    []
  );

  // Handler for connecting nodes
  const onConnect = useCallback(
    async (connection) => {
      try {
        setIsConnecting(true);
        // Create the edge in the UI
        setEdges((eds) => addEdge(connection, eds));
        const sourceNodeId = connection.source;
        const targetNodeId = connection.target;
        // Add the edge to the backend
        await NodeService.addEdge(sourceNodeId, targetNodeId);
        // Update available variables for the target node (multi-input support)
        setNodes((nds) => 
          nds.map((node) => {
            if (node.id === targetNodeId) {
              const sourceNode = nds.find(n => n.id === sourceNodeId);
              const sourceNodeName = sourceNode?.data?.nodeName || 'Unnamed Node';
              const updatedVariables = [
                ...(node.data.variables || []),
                {
                  id: sourceNodeId,
                  name: sourceNodeName
                }
              ];
              // Remove duplicates
              const uniqueVariables = updatedVariables.filter(
                (variable, index, self) => 
                  index === self.findIndex(v => v.id === variable.id)
              );
              return {
                ...node,
                data: {
                  ...node.data,
                  variables: uniqueVariables
                }
              };
            }
            return node;
          })
        );
      } catch (error) {
        // Show backend error (cycle prevention, etc.)
        alert(error.message || 'Failed to connect nodes.');
        // Revert the UI change on error
        setEdges((eds) => eds.filter(e => 
          !(e.source === connection.source && e.target === connection.target)
        ));
      } finally {
        setIsConnecting(false);
      }
    },
    []
  );

  // Add a new node when the button is clicked
  const onAddNode = useCallback(async () => {
    if (!reactFlowInstance) return;
    
    try {
      // Get the center position of the viewport
      const { x, y, zoom } = reactFlowInstance.getViewport();
      const centerX = (window.innerWidth / 2 - x) / zoom;
      const centerY = (window.innerHeight / 2 - y) / zoom;
      
      const id = uuidv4();
      const nodeName = `Node ${nodes.length + 1}`;
      
      // Add the node to the backend first
      await NodeService.addNode(
        id,
        'text_generation', // Default node type
        ['context', 'query'], // Default input keys
        ['generated_text'], // Default output keys
      );
      
      // Then add to the UI if successful
      const newNode = {
        id,
        type: 'llmNode',
        position: { x: centerX, y: centerY },
        data: {
          nodeName,
          prompt: '',
          output: '',
          variables: [],
          onNameChange: (name) => {
            // Update node name in local React state
            setNodes((nds) =>
              nds.map((node) => {
                if (node.id === id) {
                  return {
                    ...node,
                    data: {
                      ...node.data,
                      nodeName: name,
                    },
                  };
                }
                return node;
              })
            );
            
            // Update node name in other nodes' variable lists (local state)
            setNodes((nds) =>
              nds.map((node) => {
                if (node.id !== id && node.data.variables) {
                  const updatedVariables = node.data.variables.map(variable => {
                    if (variable.id === id) {
                      return {
                        ...variable,
                        name: name
                      };
                    }
                    return variable;
                  });
                  
                  return {
                    ...node,
                    data: {
                      ...node.data,
                      variables: updatedVariables
                    }
                  };
                }
                return node;
              })
            );
            
            // --- Call backend to update name in database --- 
            updateNodeName(id, name).then(success => {
              if (success) {
                console.log(`Successfully updated node ${id} name to "${name}" in database.`);
              } else {
                console.error(`Failed to update node ${id} name in database.`);
                // Optionally: Add user feedback or rollback logic here
              }
            });
            // --- End backend call --- 
          },
          onPromptChange: (prompt) => {
            setNodes((nds) =>
              nds.map((node) => {
                if (node.id === id) {
                  return {
                    ...node,
                    data: {
                      ...node.data,
                      prompt,
                    },
                  };
                }
                return node;
              })
            );
          },
          onOutputChange: (output) => {
            setNodeOutputs(prev => ({
              ...prev,
              [id]: output
            }));
          },
          onVariableSelect: (variableIds) => {
            // Store the selected variable IDs for state management
            setNodes((nds) =>
              nds.map((node) => {
                if (node.id === id) {
                  return {
                    ...node,
                    data: {
                      ...node.data,
                      selectedVariableIds: variableIds, // Store which are selected
                    },
                  };
                }
                return node;
              })
            );
          },
          onRun: async (prompt, activeInputNodeIds) => {
            // Get the CURRENT name of the node being run, not from closure
            const currentNode = nodes.find(n => n.id === id);
            const currentRunnerNodeName = currentNode?.data?.nodeName || id;
            console.log(`Running node ${id} (current name: ${currentRunnerNodeName}) with active inputs:`, activeInputNodeIds);
            
            // First, fetch the latest data from the backend for all input nodes
            let latestNodeOutputs = {};
            if (activeInputNodeIds && activeInputNodeIds.length > 0) {
              try {
                console.log("Fetching latest node outputs from backend...");
                latestNodeOutputs = await NodeService.getNodeOutputs(activeInputNodeIds);
                console.log("Latest node outputs from backend:", latestNodeOutputs);
              } catch (error) {
                console.warn("Error fetching node outputs:", error);
                // If the fetch fails, we'll use the local state (already in nodeOutputs)
                latestNodeOutputs = {};
              }
            }
            
            // Prepare context data based ONLY on actively selected nodes (checked in UI)
            let contextData = {};
            
            // Always use the __node_mapping to help backend understand node name -> ID relationships
            contextData['__node_mapping'] = {};
            
            if (activeInputNodeIds && activeInputNodeIds.length > 0) {
              console.log("Building context data using latest node outputs:");
              
              activeInputNodeIds.forEach(inputId => {
                // 1. First determine the value to use - prefer the latest from backend
                let inputNodeOutput;
                
                // Use the latest value from backend if available
                if (latestNodeOutputs[inputId]) {
                  inputNodeOutput = latestNodeOutputs[inputId];
                  console.log(`Using latest value from backend for ${inputId}: ${inputNodeOutput.substring(0, 30)}...`);
                } 
                // Fall back to local state if backend value not available
                else if (nodeOutputs[inputId]) {
                  inputNodeOutput = nodeOutputs[inputId];
                  console.log(`Using local value for ${inputId}: ${inputNodeOutput.substring(0, 30)}...`);
                }
                // Default to empty string if no value is available
                else {
                  inputNodeOutput = '';
                  console.log(`No value available for ${inputId}, using empty string`);
                }
                
                // 2. Get the current SOURCE node name from the live nodes state
                const sourceNode = nodes.find(n => n.id === inputId);
                const sourceNodeName = sourceNode?.data?.nodeName || inputId;
                
                // 3. Store the mapping from node name to ID for template processing
                contextData['__node_mapping'][sourceNodeName] = inputId;
                
                // 4. PRIMARY DATA STORAGE: Always store by ID (with 'id:' prefix)
                //    This ensures data is always retrievable by a stable ID
                contextData[`id:${inputId}`] = inputNodeOutput;
                
                // 5. SECONDARY DATA STORAGE: Also store by name for backward compatibility
                //    and easier template use, but this is not the primary source
                contextData[sourceNodeName] = inputNodeOutput;
                
                console.log(`  Added context for node ${sourceNodeName} (ID: ${inputId})`);
              });
            }
            
            // Add the current node's ID to the context
            contextData['__current_node'] = id;
            
            console.log("Final context data being sent to backend:", contextData);
            
            try {
              // Send the prompt text (with {NodeName} templates) and context data separately
              // The backend can use contextData to replace templates or as additional context
              const response = await NodeService.generateText(prompt, null, contextData);
              
              // Update this node's output in local state
              const newOutput = response.generated_text;
              
              // Update the local state for future use
              setNodeOutputs(prev => ({
                ...prev,
                [id]: newOutput
              }));
              
              // Update the visible output in the node
              setNodes(prevNodes => 
                prevNodes.map(node => {
                  if (node.id === id) {
                    return {
                      ...node,
                      data: {
                        ...node.data,
                        output: newOutput // Update visible output
                      }
                    };
                  }
                  return node;
                })
              );
              
              // If this node is an input to other nodes, log that those nodes
              // will need fresh data next time they run
              const affectedNodes = edges
                .filter(edge => edge.source === id)
                .map(edge => edge.target);
                
              if (affectedNodes.length > 0) {
                console.log(`Node ${id} output changed, affected nodes: `, affectedNodes);
                
                // Visual indicator or other action for dependent nodes
                // This could be expanded to do more than logging
                setNodes(prevNodes => 
                  prevNodes.map(node => {
                    // If this node depends on the changed node, mark it
                    if (affectedNodes.includes(node.id)) {
                      return {
                        ...node,
                        data: {
                          ...node.data,
                          needsRefresh: true // Flag that this node should be refreshed
                        }
                      };
                    }
                    return node;
                  })
                );
              }
              
              return newOutput;
            } catch (error) {
              console.error('Caught Error:', error);
              console.log('Error type:', typeof error);
              console.log('Error message:', error?.message);
              console.log('Error stringified:', JSON.stringify(error));
              
              // Ensure we return a string message
              const errorMessage = error?.message || 'An unknown error occurred';
              return `Error: ${errorMessage}`;
            }
          },
        },
      };
      
      setNodes((nds) => [...nds, newNode]);
    } catch (error) {
      console.error('Failed to add node:', error);
    }
  }, [reactFlowInstance, nodes, nodeOutputs]);

  const onDragOver = useCallback((event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  // Execute the entire flow
  const executeFlow = useCallback(async () => {
    try {
      // Prepare initial data for the chain execution
      const initialInputs = {};
      
      // For each node, find if it has any connected input nodes
      // and prepare the context accordingly
      edges.forEach(edge => {
        const sourceNode = nodes.find(n => n.id === edge.source);
        const targetNode = nodes.find(n => n.id === edge.target);
        
        if (sourceNode && targetNode) {
          const sourceOutput = nodeOutputs[sourceNode.id] || '';
          
          // Add this as input for the ScriptChain execution
          // Using the proper input keys that the ScriptChain expects
          initialInputs[`context_${targetNode.id}`] = sourceOutput;
        }
      });
      
      // Execute the entire flow with these inputs
      const result = await NodeService.executeChain(initialInputs);
      console.log('Flow execution results:', result);
      
      // Update node outputs based on results
      if (result.results) {
        const newNodeOutputs = { ...nodeOutputs };
        
        Object.entries(result.results).forEach(([nodeId, nodeResult]) => {
          // Only update if the node exists in our flow
          if (nodes.some(node => node.id === nodeId)) {
            // Update the output in the state
            const outputText = nodeResult.generated_text || 
                              nodeResult.decision_output || 
                              nodeResult.reasoning_result || 
                              nodeResult.retrieved_data || 
                              JSON.stringify(nodeResult);
            
            newNodeOutputs[nodeId] = outputText;
            
            // Update node display
            setNodes(prevNodes => 
              prevNodes.map(node => {
                if (node.id === nodeId) {
                  return {
                    ...node,
                    data: {
                      ...node.data,
                      output: outputText
                    }
                  };
                }
                return node;
              })
            );
          }
        });
        
        setNodeOutputs(newNodeOutputs);
      }
    } catch (error) {
      console.error('Error executing flow:', error);
    }
  }, [nodes, edges, nodeOutputs]);

  // Helper to find DIRECTLY connected input nodes
  const findDirectInputNodes = (nodeId, nodes, edges) => {
    const inputEdges = edges.filter(edge => edge.target === nodeId);
    const inputNodeIds = inputEdges.map(edge => edge.source);
    return nodes
      .filter(node => inputNodeIds.includes(node.id))
      .map(node => ({ id: node.id, name: node.data.nodeName || 'Unnamed Node' }));
  };

  // Helper: Find all downstream nodes from a given node (DFS) - Keep for potential future use or validation
  const findDownstreamNodes = (nodeId, edges) => {
    const visited = new Set();
    const stack = [nodeId];
    while (stack.length > 0) {
      const current = stack.pop();
      edges.forEach(edge => {
        if (edge.source === current && !visited.has(edge.target)) {
          visited.add(edge.target);
          stack.push(edge.target);
        }
      });
    }
    return visited;
  };

  return (
    <div className="reactflow-wrapper" ref={reactFlowWrapper}>
      <ReactFlow
        nodes={nodes.map(node => {
          // *** CHANGE HERE: Use findDirectInputNodes ***
          const directInputNodes = findDirectInputNodes(node.id, nodes, edges);
          // Compute valid input nodes for this node (all nodes not self, not downstream) - No longer used for dropdown, kept for reference
          // const downstream = findDownstreamNodes(node.id, edges);
          // const potentialUpstreamNodes = nodes
          //   .filter(n => n.id !== node.id && !downstream.has(n.id))
          //   .map(n => ({ id: n.id, name: n.data.nodeName || 'Unnamed Node' }));
          return {
            ...node,
            data: {
              ...node.data,
              validInputNodes: directInputNodes, // Pass only DIRECT inputs
            },
            type: node.type,
          };
        })}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onInit={setReactFlowInstance}
        onDragOver={onDragOver}
        nodeTypes={nodeTypes}
        fitView
      >
        <Background />
        <Controls />
        <Panel position="top-right">
          <button className="add-node-button" onClick={onAddNode}>
            Add Node
          </button>
          {nodes.length > 1 && (
            <button 
              className="execute-flow-button" 
              onClick={executeFlow}
            >
              Execute Flow
            </button>
          )}
          <button 
            className={`document-button ${isDocumentPanelOpen ? 'active' : ''}`}
            onClick={() => setIsDocumentPanelOpen(!isDocumentPanelOpen)}
          >
            {isDocumentPanelOpen ? 'Close Docs' : 'Open Docs'}
          </button>
        </Panel>
      </ReactFlow>

      {/* Document Panel */}
      <DocumentPanel 
        isOpen={isDocumentPanelOpen}
        onClose={() => setIsDocumentPanelOpen(false)}
        documents={documents}
        onDocumentSave={handleDocumentSave}
        onAddDocument={handleAddDocument}
      />
    </div>
  );
}

function App() {
  return (
    <div className="app">
      <ReactFlowProvider>
        <Flow />
      </ReactFlowProvider>
    </div>
  );
}

export default App;
