#!/usr/bin/env python3
"""
Test script for template processing in the ScriptChain backend.
This simulates the browser console test to debug node reference issues.
"""

import json
import requests
import sys

def test_template_processing(prompt, context_data):
    """
    Test template processing by sending a request to the debug endpoint.
    """
    url = "http://localhost:8000/debug/process_template"
    
    payload = {
        "prompt": prompt,
        "context_data": context_data
    }
    
    print(f"Sending test request to {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("\n=== TEST RESULTS ===")
            print(f"Original prompt: {result['original_prompt']}")
            print(f"Context data keys: {list(result['context_data'].keys())}")
            print(f"Processed prompt: {result['processed_prompt']}")
            print(f"Processed node values: {result['processed_node_values']}")
            print(f"Validation:")
            print(f"  Is valid: {result['validation']['is_valid']}")
            print(f"  Missing nodes: {result['validation']['missing_nodes']}")
            print(f"  Found nodes: {result['validation']['found_nodes']}")
            return result
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Exception: {e}")
        return None

def main():
    # Default test case from the screenshot example
    test_case = {
        "prompt": "Isolate and define the key terms in this question: {Question}",
        "context_data": {
            "Question": "How does algorithmic curation reshape our intellectual development and cultural perspectives?",
            # Add node mapping to simulate what the frontend sends
            "__node_mapping": {
                "Question": "abc123-fake-uuid",
                # Also add lowercase variant for testing
                "question": "abc123-fake-uuid"
            },
            # Add ID-based entry
            "id:abc123-fake-uuid": "How does algorithmic curation reshape our intellectual development and cultural perspectives?"
        }
    }
    
    # Allow passing a custom prompt and node name via command line
    if len(sys.argv) > 2:
        node_name = sys.argv[1]
        node_content = sys.argv[2]
        fake_uuid = "uuid-" + node_name.replace(" ", "-").lower()
        
        # Create mapping entries for variants of the node name
        mapping = {
            node_name: fake_uuid,  # Original name
            node_name.lower(): fake_uuid,  # Lowercase
            node_name.replace(" ", ""): fake_uuid  # No spaces
        }
        
        test_case["context_data"] = {
            node_name: node_content,
            # Add node mapping with variants
            "__node_mapping": mapping,
            # Add ID-based entry
            f"id:{fake_uuid}": node_content
        }
        
        if len(sys.argv) > 3:
            test_case["prompt"] = sys.argv[3]
        else:
            test_case["prompt"] = f"Process this content: {{{node_name}}}"
    
    # Run the test
    test_template_processing(test_case["prompt"], test_case["context_data"])

if __name__ == "__main__":
    main() 