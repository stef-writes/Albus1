import networkx as nx
from typing import Dict, Any, List, Optional, Tuple
from utils import InputValidator
from llm import client, track_token_usage, LLMConfig, default_llm_config
from callbacks import Callback
from models import MessageTemplate, PromptTemplate
from templates import template_processor
import re
import traceback

# --- Node Structure ---
class Node:
    def __init__(self, node_id, node_type, input_keys=None, output_keys=None, model_config=None, template=None):
        self.node_id = node_id        # Unique name for this node
        self.node_type = node_type      # Type of operation (e.g., "text_generation")
        self.input_keys = input_keys or [] # List of data keys this node needs from storage
        self.output_keys = output_keys or []# List of data keys this node will produce
        self.data = {}                  # Internal data for the node (not currently used)
        self.model_config = model_config or default_llm_config # Use node-specific or default LLM config
        self.token_usage = None         # To store token usage from the process method
        self.template = template        # Optional template configuration for the node

    async def process(self, inputs):
        """Processes input data based on node type. Calls specific AI functions."""
        print(f"--- Processing Node: {self.node_id} ({self.node_type}) ---")
        result = None
        api_response_for_tracking = None # Store API response here for the tracker

        # Apply template if available
        processed_inputs = self._apply_template(inputs)

        # The token tracker context manager is now placed *inside* relevant node types
        # to ensure it only runs when an actual API call is made.
        if self.node_type == "text_generation":
            with track_token_usage() as usage:
                result_data, api_response_for_tracking = generate_text(processed_inputs, self.model_config)
                self.token_usage = usage # Store usage info
            result = result_data # Assign the content result

        elif self.node_type == "decision_making":
            with track_token_usage() as usage:
                result_data, api_response_for_tracking = process_decision(processed_inputs, self.model_config)
                self.token_usage = usage
            result = result_data

        elif self.node_type == "retrieval":
            # Retrieval doesn't use LLM/tokens, so no tracking here
            result = retrieve_data(processed_inputs)

        elif self.node_type == "logic_chain":
            with track_token_usage() as usage:
                result_data, api_response_for_tracking = logical_reasoning(processed_inputs, self.model_config)
                self.token_usage = usage
            result = result_data

        else:
            print(f"Warning: Unknown node type '{self.node_type}' for node '{self.node_id}'")
            result = None

        # If an API call was made, update the tracker manually (since it's yielded)
        # The tracker printed automatically in its __exit__ method
        if self.token_usage and api_response_for_tracking:
             try:
                 # Ensure we have the dictionary form for update method
                 response_dict = api_response_for_tracking.model_dump()
                 self.token_usage.update(response_dict)
             except Exception as e:
                 print(f"Error updating token tracker: {e}")

        print(f"--- Finished Node: {self.node_id} ---")
        
        # Save the node's output to the database
        if result is not None:
            try:
                # Ensure the result is serializable (e.g., convert complex objects if needed)
                # For now, assume result is a dict or basic type
                from database import update_node
                await update_node(self.node_id, result) 
                print(f"Saved output for node {self.node_id} to database.")
            except Exception as e:
                print(f"Error saving node {self.node_id} output to database: {e}")

        return result
    
    def _apply_template(self, inputs):
        """Apply node template if defined, otherwise return original inputs."""
        if not self.template:
            return inputs
            
        # Use the global template processor for consistent processing
        return template_processor.process_node_template(self.template, inputs, self.node_id)

# --- AI Functions ---
def generate_text(inputs: Dict[str, Any], config: LLMConfig) -> Tuple[Dict[str, Any], Any]:
    """Uses OpenAI to generate structured text based on inputs using templates."""
    if not client: raise ValueError("OpenAI client not initialized")
    context = inputs.get('context', '')
    query = inputs.get('query', '')

    system_message = MessageTemplate(role="system", template="You are an expert AI assistant. {context}")
    user_message = MessageTemplate(role="user", template="{query}")

    prompt = PromptTemplate([system_message, user_message])
    formatted_messages = prompt.format_messages(context=context, query=query)

    response = client.chat.completions.create(
        model=config.model,
        messages=formatted_messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens
    )
    # Return the content dict AND the raw response object
    return {"generated_text": response.choices[0].message.content}, response

def process_decision(inputs: Dict[str, Any], config: LLMConfig) -> Tuple[Dict[str, Any], Any]:
    """AI-powered ethical decision-making based on inputs."""
    if not client: raise ValueError("OpenAI client not initialized")
    scenario = inputs.get("situation", "")
    company_value = inputs.get("value", "")

    system_message = MessageTemplate(role="system", template="Analyze this scenario based on ethical and company values.")
    user_message = MessageTemplate(role="user", template="In the given scenario: {scenario}, how does it align with the value: {company_value}?")

    prompt = PromptTemplate([system_message, user_message])
    formatted_messages = prompt.format_messages(scenario=scenario, company_value=company_value)

    response = client.chat.completions.create(
        model=config.model,
        messages=formatted_messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens
    )
    # Return the content dict AND the raw response object
    return {"decision_output": response.choices[0].message.content}, response

def retrieve_data(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieves stored data from previous nodes."""
    # Expects 'storage' (the graph's data store) and 'key' (which data to get)
    storage = inputs.get("storage", {})
    key_to_retrieve = inputs.get("key", "")
    retrieved_value = storage.get(key_to_retrieve, "No data found.")
    # Return value using a standard output key for retrieval nodes
    return {"retrieved_data": retrieved_value}

def logical_reasoning(inputs: Dict[str, Any], config: LLMConfig) -> Tuple[Dict[str, Any], Any]:
    """Processes multi-step logical AI reasoning chains."""
    if not client: raise ValueError("OpenAI client not initialized")
    premise = inputs.get("premise", "")
    supporting_evidence = inputs.get("supporting_evidence", "")

    system_message = MessageTemplate(role="system", template="Perform structured logical reasoning.")
    user_message = MessageTemplate(role="user", template="Given the premise: {premise}, and supporting evidence: {supporting_evidence}, logically conclude the next step.")

    prompt = PromptTemplate([system_message, user_message])
    formatted_messages = prompt.format_messages(premise=premise, supporting_evidence=supporting_evidence)

    response = client.chat.completions.create(
        model=config.model,
        messages=formatted_messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens
    )
    # Return the content dict AND the raw response object
    return {"reasoning_result": response.choices[0].message.content}, response

# --- Enhanced Storage and Context Management ---
class ContextVersion:
    """Tracks versions of context data for each node."""
    def __init__(self):
        self.versions = {}  # {node_id: version_number}
    
    def update(self, node_id):
        """Increment the version number for a node."""
        self.versions[node_id] = self.versions.get(node_id, 0) + 1
        return self.versions[node_id]
    
    def get(self, node_id):
        """Get the current version number for a node."""
        return self.versions.get(node_id, 0)

class NamespacedStorage:
    """
    A storage system that namespaces data by node ID to prevent key collisions.
    Allows for retrieving outputs from specific nodes or by key across all nodes.
    """
    
    def __init__(self):
        self.data = {}  # Main storage: {node_id: {key: value}}
        
    def store(self, node_id, data):
        """Store data dictionary under node_id"""
        if not isinstance(data, dict):
            raise ValueError(f"Data must be a dictionary, got {type(data)}")
        
        if node_id not in self.data:
            self.data[node_id] = {}
            
        # Store all key-value pairs from the data dict
        for key, value in data.items():
            self.data[node_id][key] = value
        
    def get(self, node_id, key=None):
        """
        Get a value from storage.
        If key is None, return all data for the node.
        If key is provided, return the specific value.
        """
        if node_id not in self.data:
            return None if key else {}
            
        if key is None:
            return self.data[node_id]
        
        return self.data[node_id].get(key)
    
    def get_all_data(self):
        """Return a flat dictionary with node_id:key as the keys"""
        flat_data = {}
        for node_id, node_data in self.data.items():
            for key, value in node_data.items():
                flat_data[f"{node_id}:{key}"] = value
        return flat_data
        
    def has_node(self, node_id):
        """Check if a node has any data stored"""
        return node_id in self.data
    
    def get_node_output(self, node_id, key=None):
        """Helper method to get output from a specific node"""
        return self.get(node_id, key)
        
    def get_by_key(self, key):
        """
        Scan all nodes for a key and return the first value found.
        This is used for backward compatibility with non-namespaced keys.
        """
        for node_data in self.data.values():
            if key in node_data:
                return node_data[key]
        return None
        
    def get_flattened(self):
        """
        Return a flattened view of all data without namespacing.
        Used for backward compatibility.
        If there are key collisions, the last value encountered wins.
        """
        flat_data = {}
        for node_data in self.data.values():
            flat_data.update(node_data)
        return flat_data

# --- ScriptChain class ---
class ScriptChain:
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.graph = nx.DiGraph()  # Directed graph to hold nodes and connections
        self.storage = NamespacedStorage()  # Namespaced storage to prevent key collisions
        self.callbacks = callbacks or []  # List of callback objects to notify
        self.node_versions = {}  # Track versions of node outputs
        self.node_dependencies = {}  # Track which nodes have been used as inputs for others

    def add_node(self, node_id: str, node_type: str, input_keys: Optional[List[str]] = None, output_keys: Optional[List[str]] = None, model_config: Optional[LLMConfig] = None):
        """Adds a node (processing step) to the graph."""
        # Associates a Node object with the node_id in the networkx graph
        node_instance = Node(node_id, node_type, input_keys, output_keys, model_config)
        self.graph.add_node(node_id, node=node_instance)
        # Initialize version tracking for this node
        if node_id not in self.node_versions:
            self.node_versions[node_id] = 0

    def add_edge(self, from_node: str, to_node: str):
        """Adds a directed connection (dependency) between two nodes."""
        # Ensures that 'from_node' must execute before 'to_node'
        self.graph.add_edge(from_node, to_node)
        
        # Track this dependency for manual execution as well
        if to_node not in self.node_dependencies:
            self.node_dependencies[to_node] = set()
        self.node_dependencies[to_node].add(from_node)

    def add_callback(self, callback: Callback):
        """Registers a callback object to receive execution events."""
        if isinstance(callback, Callback):
            self.callbacks.append(callback)
        else:
            print(f"Warning: Attempted to add non-Callback object: {callback}")
            
    def node_needs_update(self, node_id):
        """Check if a node needs updating based on dependency changes."""
        if node_id not in self.node_dependencies:
            return False  # No dependencies to check
            
        # Get last execution version of this node (0 if never executed)
        node_last_version = self.node_versions.get(node_id, 0)
        
        # Check if any dependencies have been updated since this node was last executed
        for dep_node in self.node_dependencies[node_id]:
            dep_version = self.node_versions.get(dep_node, 0)
            # If dependency version is higher than when this node was last executed, update needed
            if dep_version > node_last_version:
                return True
                
        return False

    def increment_node_version(self, node_id):
        """Increment the version of a node, indicating its output has changed."""
        self.node_versions[node_id] = self.node_versions.get(node_id, 0) + 1
        print(f"Node {node_id} version incremented to {self.node_versions[node_id]}")
        
        # Don't clear the node's own data - that would erase what we just generated!
        # Only clear data for nodes that depend on this one
        if node_id in self.node_dependencies:
            # Find all nodes that have this node as a dependency
            for dependent_node_id, dependencies in self.node_dependencies.items():
                if node_id in dependencies and dependent_node_id != node_id:
                    # Only clear data for dependent nodes, not the node itself
                    if self.storage.has_node(dependent_node_id):
                        print(f"Clearing stored results for dependent node {dependent_node_id}")
                        self.storage.data[dependent_node_id] = {}

    async def execute(self):
        """Executes the graph nodes in topological (dependency) order."""
        try:
            # Calculate the order nodes must run based on edges
            execution_order = list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            print("Error: Graph contains a cycle, cannot determine execution order.")
            return {"error": "Graph contains a cycle"}
        except Exception as e:
             print(f"Error during topological sort: {e}")
             return {"error": f"Failed to determine execution order: {e}"}

        results = {}  # Stores the final output of each node by node_id
        total_tokens = 0
        total_cost = 0.0  # Use float for cost

        print(f"--- Executing Chain (Order: {execution_order}) ---")
        
        # Before execution, check which nodes need updates due to dependency changes
        nodes_needing_updates = []
        for node_id in execution_order:
            if self.node_needs_update(node_id):
                nodes_needing_updates.append(node_id)
                # Clear any existing results for this node
                if self.storage.has_node(node_id):
                    print(f"Clearing cached results for node {node_id} due to dependency changes")
                    self.storage.data[node_id] = {}
                    
        if nodes_needing_updates:
            print(f"Nodes needing updates due to dependency changes: {nodes_needing_updates}")

        for node_id in execution_order:
            if node_id not in self.graph:
                print(f"Error: Node '{node_id}' found in execution order but not in graph.")
                continue  # Or handle error more formally

            node_instance = self.graph.nodes[node_id].get("node")
            if not isinstance(node_instance, Node):
                print(f"Error: Node '{node_id}' in graph does not contain a valid Node object.")
                continue  # Or handle error more formally
                
            # Record dependency relationship for future reference
            upstream_nodes = list(self.graph.predecessors(node_id))
            if upstream_nodes:
                if node_id not in self.node_dependencies:
                    self.node_dependencies[node_id] = set()
                for upstream_id in upstream_nodes:
                    self.node_dependencies[node_id].add(upstream_id)
                print(f"Node {node_id} depends on: {self.node_dependencies[node_id]}")

            # --- Prepare Inputs for Node --- 
            # Get required inputs that aren't node-specific
            inputs_for_node = {}
            
            # Collect outputs from each upstream node
            for upstream_id in upstream_nodes:
                node_outputs = self.storage.get_node_output(upstream_id)
                if node_outputs:
                    # Add each output with a namespaced key: {node_id}:{output_key}
                    for output_key, output_value in node_outputs.items():
                        namespaced_key = f"{upstream_id}:{output_key}"
                        inputs_for_node[namespaced_key] = output_value
                        
                        # Also provide direct access to keys specified in input_keys
                        if output_key in node_instance.input_keys:
                            inputs_for_node[output_key] = output_value
            
            # Add non-namespaced keys for backward compatibility
            for key in node_instance.input_keys:
                if key not in inputs_for_node:
                    # Try to find from any node via flattened view
                    value = self.storage.get_by_key(key)
                    if value is not None:
                        inputs_for_node[key] = value
            
            # Provide the flattened storage for backward compatibility
            inputs_for_node["storage"] = self.storage.get_flattened()
            
            # Add namespaced accessors for more precision
            inputs_for_node["get_node_output"] = self.storage.get_node_output
            
            # --- Validate inputs ---
            try:
                InputValidator.validate(node_instance, inputs_for_node)
            except ValueError as e:
                print(f"Input validation error for node {node_id}: {e}")
                results[node_id] = {"error": str(e)}
                self.storage.store(node_id, {"error": str(e)})
                continue  # Skip processing this node

            # --- Trigger on_node_start Callbacks --- 
            for callback in self.callbacks:
                try:
                    callback.on_node_start(node_id, node_instance.node_type, inputs_for_node)
                except Exception as e:
                    print(f"Error in callback {type(callback).__name__}.on_node_start for node {node_id}: {e}")

            # --- Process the Node --- 
            try:
                # Call the now async process method
                node_result = await node_instance.process(inputs_for_node) 
            except Exception as e:
                print(f"Error processing node {node_id}: {e}")
                traceback.print_exc()
                node_result = None # Indicate failure
                results[node_id] = {"error": str(e)}
                self.storage.store(node_id, {"error": str(e)})
                continue  # Skip storing result and callbacks for this failed node

            # --- Store Result --- 
            if isinstance(node_result, dict):
                results[node_id] = node_result
                self.storage.store(node_id, node_result)  # Store with namespace
            else:
                # Handle non-dict results
                results[node_id] = {"output": node_result}  # Wrap non-dict result
                self.storage.store(node_id, {"output": node_result})
                
            # --- Update node version after successful execution ---
            # Each time a node is successfully processed, increment its version
            # This signals to downstream nodes that they need to re-execute
            self.increment_node_version(node_id)
            print(f"Node {node_id} execution complete, version incremented to {self.node_versions[node_id]}")

            # --- Aggregate Token Stats --- 
            if node_instance.token_usage:
                try:
                    total_tokens += getattr(node_instance.token_usage, 'total_tokens', 0)
                    total_cost += getattr(node_instance.token_usage, 'cost', 0.0)
                except AttributeError:
                     print(f"Warning: token_usage object for node {node_id} missing expected attributes.")

            # --- Trigger on_node_complete Callbacks --- 
            for callback in self.callbacks:
                try:
                    callback.on_node_complete(node_id, node_instance.node_type, results.get(node_id), node_instance.token_usage)
                except Exception as e:
                    print(f"Error in callback {type(callback).__name__}.on_node_complete for node {node_id}: {e}")

        # --- Trigger on_chain_complete Callbacks --- 
        print("--- Chain Execution Finished ---")
        for callback in self.callbacks:
             try:
                 callback.on_chain_complete(results, total_tokens, total_cost)
             except Exception as e:
                 print(f"Error in callback {type(callback).__name__}.on_chain_complete: {e}")

        # Return final results and aggregated stats
        return {
            "results": results,  # Dictionary mapping node_id to its result dictionary
            "stats": {
                "total_tokens": total_tokens,
                "total_cost": total_cost
            }
        }

# --- ScriptChain Storage ---
# We'll use a dictionary to store separate ScriptChain instances for each session
script_chain_store = {}

# Helper function to get or create a ScriptChain for a session
def get_script_chain(session_id):
    """Get or create a ScriptChain instance for the given session ID."""
    if session_id not in script_chain_store:
        print(f"Creating new ScriptChain for session {session_id}")
        chain = ScriptChain()
        from callbacks import LoggingCallback
        chain.add_callback(LoggingCallback())
        script_chain_store[session_id] = chain
    return script_chain_store[session_id] 