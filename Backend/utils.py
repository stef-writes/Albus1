import re
import json
from typing import Dict, Any, Tuple, List, Optional

class ContentParser:
    """Parser for extracting structured data from node outputs"""
    
    @staticmethod
    def parse_numbered_list(content):
        """Extracts items from a numbered list into a dictionary"""
        if not content or not isinstance(content, str):
            return {}
        
        items = {}
        # Match patterns like "1. Item" or "1) Item" or "1: Item"
        pattern = r'(\d+)[.):]\s+(.*?)(?=\n\d+[.):]\s+|\Z)'
        matches = re.finditer(pattern, content, re.DOTALL)
        
        for match in matches:
            num, text = match.groups()
            items[int(num)] = text.strip()
        
        return items
    
    @staticmethod
    def extract_item(content, item_num):
        """Extract a specific numbered item from content"""
        if not isinstance(item_num, int):
            try:
                item_num = int(item_num)
            except (ValueError, TypeError):
                return None
        
        items = ContentParser.parse_numbered_list(content)
        return items.get(item_num)
    
    @staticmethod
    def try_parse_json(content):
        """Attempt to parse content as JSON"""
        if not content or not isinstance(content, str):
            return None
        
        try:
            # Find JSON-like structures (between { } or [ ])
            json_pattern = r'(\{.*\}|\[.*\])'
            match = re.search(json_pattern, content, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
            return None
        except Exception:
            return None
    
    @staticmethod
    def extract_table(content):
        """Extract tabular data if present"""
        if not content or not isinstance(content, str):
            return None
        
        # Simple markdown table detection
        lines = content.split('\n')
        table_start = None
        table_end = None
        
        # Look for table markers (| --- |)
        for i, line in enumerate(lines):
            if '|' in line and '---' in line and table_start is None:
                table_start = i - 1  # Header row is usually above
                continue
            if table_start is not None and ('|' not in line or line.strip() == ''):
                table_end = i
                break
        
        if table_start is not None:
            table_end = table_end or len(lines)
            if table_start >= 0:
                table_rows = lines[table_start:table_end]
                # Convert to list of dictionaries if it has a header
                if len(table_rows) >= 2:  # Need at least header + separator
                    return table_rows
        return None

class DataAccessor:
    """
    Helper class to provide structured access to node data.
    Used for advanced data extraction from node outputs.
    """
    
    def __init__(self, node_data):
        """Initialize with a dictionary of node outputs"""
        self.node_data = node_data
        self.parser = ContentParser()
        
    def get_node_content(self, node_name):
        """Get raw content from a node"""
        return self.node_data.get(node_name)
        
    def get_item(self, node_name, item_num):
        """Get a specific numbered item from a node's output"""
        if node_name not in self.node_data:
            return None
            
        content = self.node_data[node_name]
        return self.parser.extract_item(content, item_num)
        
    def get_json(self, node_name):
        """Try to parse node output as JSON"""
        if node_name not in self.node_data:
            return None
            
        content = self.node_data[node_name]
        return self.parser.try_parse_json(content)
        
    def get_table(self, node_name):
        """Extract table data from a node if present"""
        if node_name not in self.node_data:
            return None
            
        content = self.node_data[node_name]
        return self.parser.extract_table(content)
        
    def get_all_nodes(self):
        """Get list of all available node names"""
        return list(self.node_data.keys())
        
    def has_node(self, node_name):
        """Check if a node exists"""
        return node_name in self.node_data
        
    def analyze_content(self, node_name):
        """Perform comprehensive analysis of a node's content"""
        if not self.has_node(node_name):
            return None
            
        content = self.node_data[node_name]
        result = {
            "has_numbered_list": False,
            "numbered_items_count": 0,
            "has_json": False,
            "has_table": False,
            "content_length": len(content) if content else 0
        }
        
        # Check for numbered list
        numbered_items = self.parser.parse_numbered_list(content)
        if numbered_items:
            result["has_numbered_list"] = True
            result["numbered_items_count"] = len(numbered_items)
            
        # Check for JSON
        json_data = self.parser.try_parse_json(content)
        if json_data:
            result["has_json"] = True
            
        # Check for table
        table_data = self.parser.extract_table(content)
        if table_data:
            result["has_table"] = True
            
        return result

class InputValidator:
    """Validates inputs for a node before processing."""
    @staticmethod
    def validate(node, available_inputs):
        """Validate that all required inputs for a node are available."""
        missing = []
        for key in node.input_keys:
            if key not in available_inputs or available_inputs[key] is None:
                missing.append(key)
        
        if missing:
            raise ValueError(f"Missing required inputs for node '{node.node_id}': {missing}")
        
        return True 