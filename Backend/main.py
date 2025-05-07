import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional, Tuple

# Import modular components
from llm import LLMConfig, default_llm_config, track_token_usage, client
from models import (
    Message, ModelConfigInput, NodeInput, EdgeInput, GenerateTextNodeRequest, 
    GenerateTextNodeResponse, NodeNameUpdate, TemplateValidationRequest, TemplateValidationResponse
)
from script_chain import ScriptChain, Node
from callbacks import Callback, LoggingCallback
from templates import TemplateProcessor, template_processor
from utils import ContentParser, DataAccessor
from database import save_node, get_node, save_chain, get_chain, get_all_chains, get_all_nodes, update_node, delete_node, update_node_name

# Create FastAPI app
app = FastAPI()

# Configure CORS
origins = [
    "http://localhost",        
    "http://localhost:5173",  
    "http://127.0.0.1",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       
    allow_credentials=True,     
    allow_methods=["GET", "POST", "OPTIONS", "PUT"], 
    allow_headers=["*"],          
)

# --- ScriptChain Storage ---
# We'll use a dictionary to store separate ScriptChain instances for each session
script_chain_store = {}

# Helper function to get or create a ScriptChain for a session
def get_script_chain(session_id):
    """Get or create a ScriptChain instance for the given session ID."""
    if session_id not in script_chain_store:
        print(f"Creating new ScriptChain for session {session_id}")
        chain = ScriptChain()
        chain.add_callback(LoggingCallback())
        script_chain_store[session_id] = chain
    return script_chain_store[session_id]

# --- API Routes ---
@app.get("/")
def read_root():
    return {"message": "ScriptChain Backend Running"}

@app.post("/add_node", status_code=201)
async def add_node_api(node: NodeInput, session_id: str = "default"):
    """Adds a node to the script chain via API and saves it to the database."""
    script_chain = get_script_chain(session_id)
    llm_config_for_node = default_llm_config
    # Use the renamed field 'llm_config' here
    if node.llm_config:
        llm_config_for_node = LLMConfig(
            model=node.llm_config.model,
            temperature=node.llm_config.temperature,
            max_tokens=node.llm_config.max_tokens
        )
    try:
        script_chain.add_node(
            node_id=node.node_id,
            node_type=node.node_type,
            input_keys=node.input_keys,
            output_keys=node.output_keys,
            model_config=llm_config_for_node # This maps to the Node class __init__ param
        )
        print(f"Added node: {node.node_id} to in-memory chain for session {session_id}")
        
        # --- Add node to the database --- 
        try:
            node_db_data = {
                "node_id": node.node_id,
                "node_type": node.node_type,
                "input_keys": node.input_keys,
                "output_keys": node.output_keys,
                "output": None, # Initialize output as None
                "name": node.node_id # Use node_id as the initial default name
            }
            await save_node(node_db_data) # Save basic node info to MongoDB
            print(f"Saved node: {node.node_id} to database with initial name.")
            return {"message": f"Node '{node.node_id}' added successfully to chain and database."}
        except Exception as db_e:
            print(f"Error saving node {node.node_id} to database: {db_e}")
            # Note: Node is added to memory chain but not DB. Consider if rollback is needed.
            raise HTTPException(status_code=500, detail=f"Node added to chain but failed to save to database: {str(db_e)}")
            
    except Exception as chain_e:
        print(f"Error adding node {node.node_id} to chain: {chain_e}")
        raise HTTPException(status_code=400, detail=f"Failed to add node to chain: {str(chain_e)}")

@app.post("/add_edge", status_code=201)
async def add_edge_api(edge: EdgeInput, session_id: str = "default"):
    """Adds an edge to the script chain via API."""
    script_chain = get_script_chain(session_id)
    # Basic validation: Check if nodes exist before adding edge
    if edge.from_node not in script_chain.graph or edge.to_node not in script_chain.graph:
        raise HTTPException(status_code=404, detail=f"Node(s) not found: '{edge.from_node}' or '{edge.to_node}'")
    
    # --- CYCLE PREVENTION ---
    # Temporarily add the edge and check for cycles
    script_chain.graph.add_edge(edge.from_node, edge.to_node)
    try:
        import networkx as nx
        if not nx.is_directed_acyclic_graph(script_chain.graph):
            script_chain.graph.remove_edge(edge.from_node, edge.to_node)
            raise HTTPException(status_code=400, detail="Adding this edge would create a cycle. Please check your node connections.")
    except ImportError:
        # Handle case where networkx is not imported in this file
        script_chain.graph.remove_edge(edge.from_node, edge.to_node)
        script_chain.add_edge(edge.from_node, edge.to_node)
    
    print(f"Added edge: {edge.from_node} -> {edge.to_node} for session {session_id}")
    return {"message": f"Edge from '{edge.from_node}' to '{edge.to_node}' added successfully."}

# --- Generate Text Node API Route ---
@app.post("/generate_text_node", response_model=GenerateTextNodeResponse)
async def generate_text_node_api(request: GenerateTextNodeRequest, session_id: str = "default"):
    """Executes a single text generation call based on provided prompt text."""
    script_chain = get_script_chain(session_id)
    
    # --- Log the raw incoming request data --- 
    try:
        print(f"\n=== RAW generate_text_node_api Request (Session: {session_id}) ===")
        print(f"Prompt Text: {request.prompt_text}")
        print(f"Context Data Received: {request.context_data}")
        print(f"LLM Config Received: {request.llm_config}")
        print("=== END RAW Request ===\n")
    except Exception as log_e:
        print(f"Error logging raw request: {log_e}")
    # --- End Logging --- 

    # Extract node mapping information from context data
    node_mapping = request.context_data.get('__node_mapping', {}) if request.context_data else {}
    current_node_id = None  # The ID of the node being executed
    input_node_id = None    # The ID of a node that provides input
    
    # Identify the current node (if this is a node execution)
    if request.context_data and '__current_node' in request.context_data:
        current_node_id = request.context_data['__current_node']
        print(f"Explicit current node ID: {current_node_id}")
        
    # Detect if this is a simple node output update request
    # (i.e., user is re-generating content for an existing node)
    if current_node_id is None and request.context_data and 'node_id' in request.context_data:
        current_node_id = request.context_data['node_id']
        print(f"Found node_id in context: {current_node_id}")
    
    # Look for template references in the prompt to identify dependencies
    if '{' in request.prompt_text:
        import re
        dependency_pattern = r'\{([^}:]+)(?:\[.*\])?\}'
        referenced_nodes = re.findall(dependency_pattern, request.prompt_text)
        for node_name in referenced_nodes:
            if node_name in node_mapping:
                input_node_id = node_mapping[node_name]
                print(f"Found input node: {node_name} with ID: {input_node_id}")
            
            # If this is not a node with ID in the mapping, check if we're supposed to
            # generate new content for a dependency node
            elif request.context_data and node_name in request.context_data:
                content = request.context_data[node_name]
                print(f"Found input node {node_name} with content: {content}")

    # For each node in the node_mapping, check if its value has changed
    # by comparing to what's in storage
    nodes_with_updated_content = []
    if node_mapping:
        for node_name, node_id in node_mapping.items():
            # Skip the node that is currently being executed
            if node_id == current_node_id:
                continue
                
            # Compare provided content with stored content
            if request.context_data and node_name in request.context_data:
                provided_content = request.context_data[node_name]
                stored_content = None
                
                # Get the stored content if the node exists in storage
                if script_chain.storage.has_node(node_id):
                    # Get the first value in the node's data (simplified assumption)
                    node_data = script_chain.storage.get_node_output(node_id)
                    if node_data:
                        first_key = next(iter(node_data))
                        stored_content = node_data.get(first_key)
                
                # If content has changed, update the storage
                if provided_content != stored_content:
                    print(f"Node {node_name} (ID: {node_id}) content has changed:")
                    print(f"  Old: {stored_content}")
                    print(f"  New: {provided_content}")
                    
                    # Update the storage directly
                    script_chain.storage.store(node_id, {"generated_text": provided_content})
                    
                    # Increment the node version to trigger dependency updates
                    script_chain.increment_node_version(node_id)
                    
                    nodes_with_updated_content.append(node_id)
    
    if nodes_with_updated_content:
        print(f"Updated content for nodes: {nodes_with_updated_content}")
        
        # For any node that depends on the updated nodes, reset its version
        for dependent_node_id, dependencies in script_chain.node_dependencies.items():
            for updated_node_id in nodes_with_updated_content:
                if updated_node_id in dependencies:
                    print(f"Marking node {dependent_node_id} for refresh due to dependency changes")
                    # Clear any existing results
                    if script_chain.storage.has_node(dependent_node_id):
                        script_chain.storage.data[dependent_node_id] = {}

    # --- Process the template ---
    # Now we process the template to replace node references with actual content
    processed_prompt, processed_node_values = template_processor.process_node_references(
        request.prompt_text,
        request.context_data or {}
    )
    
    # Configure model to use
    node_config = default_llm_config
    if request.llm_config:
        node_config = LLMConfig(
            model=request.llm_config.model,
            temperature=request.llm_config.temperature,
            max_tokens=request.llm_config.max_tokens
        )
    
    print(f"Using model config: {node_config.model}, temp={node_config.temperature}, max_tokens={node_config.max_tokens}")
    
    # Build enhanced system content
    system_content = "You are an expert AI assistant helping with data analysis and text generation."
    
    # Add reference to mapping between node names and IDs if present
    if request.context_data and '__node_mapping' in request.context_data:
        mapping_info = request.context_data['__node_mapping']
        system_content += f"\n\nYou have access to a graph of connected nodes with the following name-to-ID mapping: {mapping_info}"
    
    # Add reference to all keys in context_data except for special system keys
    if request.context_data:
        context_keys = [k for k in request.context_data.keys() if k != '__node_mapping' and k != '__current_node']
        if context_keys:
            system_content += f"\n\nYou have access to information from the following nodes: {', '.join(context_keys)}."
            system_content += "\nUse this information to inform your response."

    print(f"\n=== FULL SYSTEM CONTENT ===\n{system_content}\n=== END SYSTEM CONTENT ===\n")
    
    # Prepare messages with enhanced system content
    messages_payload = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": processed_prompt}
    ]

    print(f"\n=== FULL MESSAGE PAYLOAD ===")
    for idx, msg in enumerate(messages_payload):
        print(f"Message {idx+1} ({msg['role']}):\n{msg['content']}\n---")
    print(f"=== END MESSAGE PAYLOAD ===\n")

    response_content = None
    tracker_instance = None

    try:
        # Use the token tracker context manager
        with track_token_usage() as tracker:
            response = client.chat.completions.create(
                model=node_config.model,
                messages=messages_payload,
                temperature=node_config.temperature,
                max_tokens=node_config.max_tokens
            )
            # Update the tracker with the response
            response_dict = response.model_dump()
            tracker.update(response_dict)
            response_content = response.choices[0].message.content
            tracker_instance = tracker # Store the tracker instance
            
            # Log the response content
            print(f"\n=== RESPONSE CONTENT ===\n{response_content}\n=== END RESPONSE CONTENT ===\n")
            
            # If we identified the current node, store the result and increment version
            if current_node_id:
                # Store the result in the global chain's storage
                # Store with BOTH "generated_text" key (API convention) and just "output" key
                # for easier access by other parts of the system
                result_data = {
                    "generated_text": response_content,  # Standard API response key
                    "output": response_content,          # Simpler key for general use
                    "content": response_content          # Alternative key some components might use
                }
                script_chain.storage.store(current_node_id, result_data)
                
                # Update the node version
                script_chain.increment_node_version(current_node_id)
                print(f"Updated node {current_node_id} with new content and incremented version")
                
                # Output what's actually in storage for debugging
                stored_data = script_chain.storage.get_node_output(current_node_id)
                print(f"Stored data for node {current_node_id}: {stored_data}")

        if response_content is None:
            raise ValueError("Received no content from OpenAI.")

        # Prepare the response using the Pydantic model
        return GenerateTextNodeResponse(
            generated_text=response_content,
            prompt_tokens=getattr(tracker_instance, 'prompt_tokens', None),
            completion_tokens=getattr(tracker_instance, 'completion_tokens', None),
            total_tokens=getattr(tracker_instance, 'total_tokens', None),
            cost=getattr(tracker_instance, 'cost', None),
            duration=round(tracker_instance.end_time - tracker_instance.start_time, 2) if tracker_instance and tracker_instance.end_time else None
        )

    except Exception as e:
        print(f"Error during single text generation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed text generation: {str(e)}")

@app.post("/get_node_outputs")
async def get_node_outputs(request: Dict[str, List[str]], session_id: str = "default"):
    """Retrieves the current output values for specified nodes from the script chain."""
    script_chain = get_script_chain(session_id)
    try:
        node_ids = request.get("node_ids", [])
        if not node_ids:
            return {}
            
        result = {}
        print(f"Fetching outputs for nodes: {node_ids} (Session: {session_id})")
        
        # For debugging: show all storage
        print(f"Current storage state for session {session_id}:")
        for node_id, data in script_chain.storage.data.items():
            print(f"  Node {node_id}: {data}")
        
        for node_id in node_ids:
            if script_chain.storage.has_node(node_id):
                node_data = script_chain.storage.get_node_output(node_id)
                print(f"Node {node_id} data: {node_data}")
                
                if node_data:
                    # Try several output key patterns in priority order
                    # This makes the endpoint more robust to different node types
                    output_keys = ["generated_text", "output", "content", "result"]
                    
                    # First try the standard output keys
                    found_output = False
                    for key in output_keys:
                        if key in node_data:
                            result[node_id] = node_data[key]
                            found_output = True
                            print(f"Node {node_id}: found output under key '{key}'")
                            break
                    
                    # If no standard key found, use the first available key
                    if not found_output and node_data:
                        first_key = next(iter(node_data))
                        result[node_id] = node_data[first_key]
                        print(f"Node {node_id}: used first available key '{first_key}'")
                    
                    # Convert None to empty string for consistency
                    if result.get(node_id) is None:
                        result[node_id] = ""
                else:
                    print(f"Node {node_id} exists but has no data")
            else:
                print(f"Node {node_id} not found in storage")
                    
        print(f"Returning outputs for {len(result)} nodes: {result}")
        return result
    except Exception as e:
        print(f"Error retrieving node outputs: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to retrieve node outputs: {str(e)}")

@app.post("/execute")
async def execute_api(initial_inputs: Optional[Dict[str, Any]] = None, session_id: str = "default"):
    """Executes the AI-driven node chain."""
    script_chain = get_script_chain(session_id)
    print(f"--- Received /execute request (Session: {session_id}) ---")
    if initial_inputs:
        for key, value in initial_inputs.items():
            script_chain.storage.data[key] = value
        print(f"Initial storage set to: {script_chain.storage.data}")

    try:
        results = await script_chain.execute()
        if results and "error" in results:
             raise HTTPException(status_code=400, detail=results["error"])
        return results
    except Exception as e:
        print(f"Error during chain execution: {e}")
        raise HTTPException(status_code=500, detail=f"Chain execution failed: {str(e)}")

# --- Debug Endpoints ---
@app.get("/debug/node_content")
async def debug_node_content(node_content: str):
    """Debug endpoint to test content parsing functionality."""
    result = {
        "original_content": node_content,
        "analysis": {},
        "parsed_data": {}
    }
    
    # Analyze using ContentParser
    parser = ContentParser()
    numbered_items = parser.parse_numbered_list(node_content)
    json_data = parser.try_parse_json(node_content)
    table_data = parser.extract_table(node_content)
    
    # Build analysis
    result["analysis"] = {
        "has_numbered_list": bool(numbered_items),
        "numbered_items_count": len(numbered_items) if numbered_items else 0,
        "has_json": json_data is not None,
        "has_table": table_data is not None
    }
    
    # Add parsed data
    result["parsed_data"] = {
        "numbered_items": numbered_items,
        "json_data": json_data,
        "table_data": table_data
    }
    
    return result

@app.post("/debug/process_template")
async def debug_process_template(request: dict):
    """Debug endpoint to test template processing directly."""
    if "prompt" not in request or "context_data" not in request:
        raise HTTPException(status_code=400, detail="Request must include 'prompt' and 'context_data' fields")
    
    prompt = request["prompt"]
    context_data = request["context_data"]
    
    print(f"Debug process template request:")
    print(f"Prompt: {prompt}")
    print(f"Context data: {context_data}")
    
    # Process the template using our unified processor
    processed_prompt, processed_node_values = template_processor.process_node_references(
        prompt, context_data
    )
    
    # Return detailed results for debugging
    return {
        "original_prompt": prompt,
        "context_data": context_data,
        "processed_prompt": processed_prompt,
        "processed_node_values": processed_node_values,
        "validation": {
            "is_valid": len(template_processor.validate_node_references(prompt, context_data.keys())[1]) == 0,
            "missing_nodes": template_processor.validate_node_references(prompt, context_data.keys())[1],
            "found_nodes": template_processor.validate_node_references(prompt, context_data.keys())[2],
        }
    }

@app.post("/validate_template", response_model=TemplateValidationResponse)
async def validate_template_api(request: TemplateValidationRequest):
    """Validates that all node references in a template exist in the available nodes."""
    is_valid, missing_nodes, found_nodes = template_processor.validate_node_references(
        request.prompt_text, set(request.available_nodes)
    )
    
    warnings = []
    if not is_valid:
        for node in missing_nodes:
            warnings.append(f"Node reference '{node}' not found in available nodes.")
    
    return TemplateValidationResponse(
        is_valid=is_valid,
        missing_nodes=missing_nodes,
        found_nodes=found_nodes,
        warnings=warnings
    )

@app.post("/debug/test_reference")
async def debug_test_reference(request: dict):
    """Test endpoint for reference extraction."""
    if "content" not in request or "reference" not in request:
        raise HTTPException(status_code=400, detail="Request must include 'content' and 'reference' fields")
    
    content = request["content"]
    reference = request["reference"]
    
    # Create fake context with Node1 containing the content
    fake_context = {"Node1": content}
    data_accessor = DataAccessor(fake_context)
    
    # Try to parse the reference
    result = {
        "original_content": content,
        "reference": reference,
        "parsed_result": None,
        "details": {}
    }
    
    # Check if it's an item reference
    import re
    item_ref_pattern = r'\{([^:\}\[]+)(?:\[(\d+)\]|\:item\((\d+)\))\}'
    match = re.search(item_ref_pattern, reference)
    
    if match:
        node_name = match.group(1)
        item_num_str = match.group(2) or match.group(3)
        
        result["details"] = {
            "node_name": node_name,
            "item_num": item_num_str,
            "is_valid_node": data_accessor.has_node(node_name)
        }
        
        try:
            item_num = int(item_num_str)
            result["details"]["valid_item_num"] = True
            
            # Get the specific item
            item_content = data_accessor.get_item(node_name, item_num)
            result["parsed_result"] = item_content
            
        except ValueError:
            result["details"]["valid_item_num"] = False
    else:
        result["details"]["is_reference_pattern"] = False
    
    return result

# --- Database API endpoints ---
@app.post("/nodes/")
async def create_node(node_id: str, output: Any):
    """Create a new node with ID and output"""
    try:
        node_id = await save_node({"node_id": node_id, "output": output})
        return {"id": node_id, "message": "Node created successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/nodes/{node_id}")
async def read_node(node_id: str):
    """Get a node by ID"""
    node = await get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")
    return node

@app.put("/nodes/{node_id}")
async def update_node_output(node_id: str, output: Any):
    """Update a node's output"""
    success = await update_node(node_id, output)
    if not success:
        raise HTTPException(status_code=404, detail="Node not found")
    return {"message": "Node updated successfully"}

@app.delete("/nodes/{node_id}")
async def remove_node(node_id: str):
    """Delete a node"""
    success = await delete_node(node_id)
    if not success:
        raise HTTPException(status_code=404, detail="Node not found")
    return {"message": "Node deleted successfully"}

@app.get("/nodes/")
async def list_nodes(q: Optional[str] = None): # Add optional query parameter 'q'
    """List all nodes, optionally filtering by name (node_id) via query param 'q'."""
    nodes = await get_all_nodes(name_query=q) # Pass the query to the database function
    return nodes

@app.post("/chains/")
async def create_chain(chain_data: Dict[str, Any]):
    """Create a new chain"""
    chain_id = await save_chain(chain_data)
    return {"id": chain_id, "message": "Chain created successfully"}

@app.get("/chains/{chain_id}")
async def read_chain(chain_id: str):
    """Get a chain by ID"""
    chain = await get_chain(chain_id)
    if chain is None:
        raise HTTPException(status_code=404, detail="Chain not found")
    return chain

@app.get("/chains/")
async def list_chains():
    """List all chains"""
    chains = await get_all_chains()
    return chains

@app.put("/nodes/{node_id}/name")
async def update_node_name_api(node_id: str, update_data: NodeNameUpdate):
    """Update the name of a node."""
    success = await update_node_name(node_id, update_data.name)
    if not success:
        raise HTTPException(status_code=404, detail="Node not found or name could not be updated")
    return {"message": f"Node '{node_id}' name updated successfully to '{update_data.name}'"}

# --- Run Server ---
if __name__ == "__main__":
    print("Starting backend server at http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)