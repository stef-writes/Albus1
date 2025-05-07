import re
from typing import Dict, Any, Tuple, List
from utils import DataAccessor

class TemplateProcessor:
    """
    Unified template processing system that handles all node variable substitutions.
    This replaces both Node._apply_template and the template processing in generate_text_node_api.
    """
    
    def __init__(self, debug_mode=True):
        """Initialize the template processor"""
        self.debug_mode = debug_mode  # Enable detailed logging
        
    def log(self, message):
        """Log messages when debug mode is enabled"""
        if self.debug_mode:
            print(message)
    
    def validate_node_references(self, template_text, available_nodes):
        """
        Validate that all node references in the template exist in available_nodes.
        Returns a tuple: (is_valid, list_of_missing_nodes, list_of_found_nodes)
        """
        if not template_text or not isinstance(template_text, str):
            return True, [], []
            
        # Find all {NodeName} references in the template
        # This pattern matches both simple {NodeName} and indexed {NodeName[n]} references
        reference_pattern = r'\{([^:\}\[]+)(?:\[(\d+)\]|\:item\((\d+)\))?\}'
        matches = re.findall(reference_pattern, template_text)
        
        # Create normalized versions of available nodes for case-insensitive matching
        normalized_available_nodes = {node.lower().strip(): node for node in available_nodes}
        
        missing_nodes = []
        found_nodes = []
        
        for match in matches:
            node_name = match[0]
            normalized_node_name = node_name.lower().strip()
            
            # First try exact match
            if node_name in available_nodes:
                found_nodes.append(node_name)
            # Then try case-insensitive match
            elif normalized_node_name in normalized_available_nodes:
                original_node = normalized_available_nodes[normalized_node_name]
                found_nodes.append(node_name)  # Add the referenced name
                self.log(f"Found node '{node_name}' using case-insensitive matching (original: '{original_node}')")
            else:
                missing_nodes.append(node_name)
        
        is_valid = len(missing_nodes) == 0
        return is_valid, missing_nodes, found_nodes
    
    def process_node_template(self, template, inputs, node_id=None):
        """
        Process a node's template configuration (used by Node._apply_template).
        Returns processed inputs dictionary.
        """
        if not template:
            return inputs
            
        # Create a copy to avoid modifying the original
        processed_inputs = inputs.copy()
        
        try:
            self.log(f"Processing node template for node: {node_id or 'unknown'}")
            
            # First validate we have all required inputs
            missing_inputs = []
            for field_name, template_string in template.items():
                if not isinstance(template_string, str):
                    continue
                    
                # Find all input references in this template
                # Standard Python format variables like {variable_name}
                var_pattern = r'\{([^{}]+)\}'
                variables = re.findall(var_pattern, template_string)
                
                for var in variables:
                    # Skip function references like get_output
                    if '(' in var:
                        continue
                        
                    if var not in inputs and ":" not in var:
                        missing_inputs.append(var)
            
            if missing_inputs:
                self.log(f"Warning: Template missing inputs: {missing_inputs}")
            
            # Process each template field
            for field_name, template_string in template.items():
                # Skip if the field is not a string template
                if not isinstance(template_string, str):
                    continue
                    
                try:
                    # Replace namespaced keys with their values for template formatting
                    template_context = {}
                    
                    # Add non-namespaced values first
                    for key, value in inputs.items():
                        if ":" not in key and not callable(value):
                            template_context[key] = value
                    
                    # Add access to namespaced values through helper functions
                    if "get_node_output" in inputs and callable(inputs["get_node_output"]):
                        get_node_output = inputs["get_node_output"]
                        
                        # Add a function to access node outputs in templates
                        def get_output(node_id, output_key=None):
                            return get_node_output(node_id, output_key)
                        
                        template_context["get_output"] = get_output
                    
                    formatted_value = template_string.format(**template_context)
                    processed_inputs[field_name] = formatted_value
                    
                    self.log(f"Processed template field '{field_name}': {formatted_value[:50]}...")
                    
                except KeyError as e:
                    self.log(f"Warning: Missing key {e} in template for node {node_id}")
                except Exception as e:
                    self.log(f"Error formatting template for node {node_id}: {e}")
        
        except Exception as e:
            self.log(f"Error applying template for node {node_id}: {e}")
            # Fall back to original inputs on error
            return inputs
            
        return processed_inputs
    
    def process_node_references(self, prompt_text, context_data, data_accessor=None):
        """
        Process a prompt text with node references (used by generate_text_node_api).
        Handles direct name matching, normalized matching, and name-to-ID mapping.
        Returns processed prompt and a dictionary of processed node values.
        """
        if not prompt_text or not context_data:
            return prompt_text, {}
            
        processed_prompt = prompt_text
        processed_node_values = {}  # Track which nodes were processed and their values
        
        try:
            self.log(f"Processing node references in prompt")
            
            # Extract node mapping if available
            node_mapping = context_data.get('__node_mapping', {})
            if node_mapping:
                self.log(f"Found node name-to-ID mapping: {node_mapping}")
                
            # Create a normalized version of context_data keys for case-insensitive matching
            # Exclude the special mapping key and ID-prefixed keys from this direct lookup map
            normalized_context_data = {}
            for key, value in context_data.items():
                if key == '__node_mapping' or key == '__current_node':
                    continue
                normalized_key = key.lower().strip()
                normalized_context_data[normalized_key] = (key, value)
            
            self.log(f"Normalized context data keys for name matching: {list(normalized_context_data.keys())}")
            
            # Validate node references against all available keys (names and IDs)
            available_keys_for_validation = [k for k in context_data.keys() 
                if k != '__node_mapping' and k != '__current_node']
            is_valid, missing_nodes, found_nodes = self.validate_node_references(
                prompt_text, set(available_keys_for_validation)
            )
            
            # We don't necessarily fail on missing nodes here, as mapping might resolve them later
            if not is_valid:
                self.log(f"Warning: Initial validation shows potentially missing nodes: {missing_nodes}")
            
            if found_nodes:
                self.log(f"Found initial references to nodes/IDs: {found_nodes}")
            
            # Step 1: Create data accessor if not provided
            # Important: Pass the *full* context_data (including id: keys) to DataAccessor
            # but filter out the mapping key itself
            if not data_accessor and context_data:
                filtered_context_data = {k: v for k, v in context_data.items() 
                    if k != '__node_mapping' and k != '__current_node'}
                data_accessor = DataAccessor(filtered_context_data)
            
            # Step 2: Process advanced references like {NodeName[n]} first
            if data_accessor:
                item_ref_pattern = r'\{([^:\}\[]+)(?:\[(\d+)\]|\:item\((\d+)\))\}'
                
                def replace_item_reference(match):
                    full_match = match.group(0)
                    node_name_ref = match.group(1) # The name used in the reference
                    item_num_str = match.group(2) or match.group(3)
                    
                    self.log(f"Processing item reference: {full_match}")
                    self.log(f"  Node Ref: {node_name_ref}")
                    self.log(f"  Item: {item_num_str}")
                    
                    actual_node_key_for_data = None
                    node_output_for_data = None
                    
                    # Lookup Priority (IMPROVED - id lookup first):
                    # 1. Try to get the node ID from the mapping if available
                    node_id = node_mapping.get(node_name_ref)
                    if node_id:
                        id_key = f"id:{node_id}"
                        if id_key in context_data:
                            actual_node_key_for_data = id_key  # Use the ID key for data lookup
                            node_output_for_data = context_data[id_key]
                            self.log(f"  Found node '{node_name_ref}' via mapping to ID '{node_id}' (using key '{id_key}')")
                    
                    # 2. If not found by ID, try exact name match
                    if actual_node_key_for_data is None and node_name_ref in context_data:
                        actual_node_key_for_data = node_name_ref
                        node_output_for_data = context_data[node_name_ref]
                        self.log(f"  Found node '{node_name_ref}' by exact name match.")
                    
                    # 3. If still not found, try normalized name match
                    if actual_node_key_for_data is None:
                        normalized_node_name_ref = node_name_ref.lower().strip()
                        if normalized_node_name_ref in normalized_context_data:
                            actual_node_key_for_data, node_output_for_data = normalized_context_data[normalized_node_name_ref]
                            self.log(f"  Found node '{node_name_ref}' using normalized matching (actual key: '{actual_node_key_for_data}')")
                        
                    if actual_node_key_for_data is None:
                        self.log(f"  Node reference '{node_name_ref}' not found by ID, name, or normalized name.")
                        return full_match  # Cannot resolve this reference

                    # Ensure DataAccessor has the data under the *actual* key we found
                    if not data_accessor.has_node(actual_node_key_for_data):
                         data_accessor.node_data[actual_node_key_for_data] = node_output_for_data
                         
                    # Convert item number to int
                    try:
                        item_num = int(item_num_str)
                    except ValueError:
                        self.log(f"  Invalid item number: {item_num_str}")
                        return full_match
                    
                    # Get the specific item using the *actual* node key
                    item_content = data_accessor.get_item(actual_node_key_for_data, item_num)
                    if item_content:
                        self.log(f"  Found item {item_num}: {item_content}")
                        processed_node_values[f"{node_name_ref}[{item_num}]"] = item_content 
                        return item_content
                    
                    self.log(f"  Item {item_num} not found in node {actual_node_key_for_data}")
                    return full_match
                
                processed_prompt = re.sub(item_ref_pattern, replace_item_reference, processed_prompt)
                self.log(f"After item reference processing: {processed_prompt[:100]}...")
            
            # Step 3: Process normal node references {NodeName}
            normal_ref_pattern = r'\{([^:\}\[]+?)\}' # Matches only {Name}, not {Name[...]} etc.
            
            # Use finditer to replace iteratively, avoiding multiple replacements if a value itself contains a template
            new_processed_prompt = ""
            last_end = 0
            for match in re.finditer(normal_ref_pattern, processed_prompt):
                node_name_ref = match.group(1)
                template_marker = match.group(0)
                start, end = match.span()
                
                # Append text before the match
                new_processed_prompt += processed_prompt[last_end:start]
                
                self.log(f"Processing normal reference: {template_marker}")
                
                final_value_str = template_marker # Default to keep original if not found
                node_output = None
                actual_node_key = None

                # Lookup Priority (IMPROVED - ID lookup first):
                # 1. Try to get the node ID from the mapping if available
                node_id = node_mapping.get(node_name_ref)
                if node_id:
                    # Direct ID reference
                    id_key = f"id:{node_id}"
                    if id_key in context_data:
                        actual_node_key = id_key
                        node_output = context_data[id_key]
                        self.log(f"  Found via mapping to ID '{node_id}' (key: '{id_key}')")
                
                # 2. If not found by ID, try direct name match
                if node_output is None and node_name_ref in context_data:
                    actual_node_key = node_name_ref
                    node_output = context_data[node_name_ref]
                    self.log(f"  Found by exact name match: '{actual_node_key}'")
                
                # 3. If still not found, try normalized name match
                if node_output is None:
                    normalized_node_name_ref = node_name_ref.lower().strip()
                    if normalized_node_name_ref in normalized_context_data:
                        actual_node_key, node_output = normalized_context_data[normalized_node_name_ref]
                        self.log(f"  Found by normalized name match: '{actual_node_key}'")

                if node_output is not None:
                    self.log(f"  Original node output: '{node_output}'")
                    processed_value = node_output # Start with the raw output
                    # --- Apply Numeric Processing (only if found) ---
                    try:
                        if isinstance(node_output, (int, float)):
                            processed_value = str(node_output)
                            self.log(f"  Numeric value detected (direct): {processed_value}")
                        elif isinstance(node_output, str):
                           if node_output.strip().replace('.', '', 1).isdigit():
                               processed_value = node_output.strip()
                               self.log(f"  Numeric value detected (string): {processed_value}")
                           else:
                               number_pattern = r'^\s*(\d+(\.\d+)?)\s*$'
                               num_match = re.search(number_pattern, node_output)
                               if num_match:
                                   processed_value = num_match.group(1)
                                   self.log(f"  Numeric value detected (pattern): {processed_value}")
                    except Exception as e:
                        self.log(f"  Error processing numeric node output: {e}")
                        processed_value = node_output # Fallback on error
                    # --- End Numeric Processing ---
                    
                    final_value_str = str(processed_value) # Ensure it's a string for substitution
                    new_processed_prompt += final_value_str
                    processed_node_values[node_name_ref] = final_value_str # Log with the reference name
                    self.log(f"  Replaced {template_marker} with: '{final_value_str[:100]}...'")
                else:
                    # Keep the original template marker if not resolved
                    self.log(f"  Could not resolve reference: {template_marker}")
                    error_message = f"[CONTEXT_ERROR: Content for '{node_name_ref}' not found in provided context_data]"
                    new_processed_prompt += error_message
                    
                last_end = end
            
            # Append any remaining text after the last match
            new_processed_prompt += processed_prompt[last_end:]
            processed_prompt = new_processed_prompt
            
            # Print summary 
            self.log(f"\n--- Template Processing Summary ---")
            self.log(f"Original prompt: {prompt_text}")
            self.log(f"Processed nodes (references used):")
            for node_ref, value in processed_node_values.items():
                value_str = str(value) if not isinstance(value, str) else value
                self.log(f"  - {{{node_ref}}}: '{value_str[:100]}...'")
            self.log(f"Final processed prompt: {processed_prompt}")
            self.log(f"--- End Template Processing Summary ---\n")
            
        except Exception as e:
            self.log(f"Error processing template variables: {e}")
            import traceback
            traceback.print_exc()
            # Return original on severe error
            return prompt_text, {}
            
        return processed_prompt, processed_node_values

# Create a global instance of the template processor
template_processor = TemplateProcessor(debug_mode=True) 