from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# --- Prompt Templating System ---
@dataclass
class MessageTemplate:
    role: str
    template: str # String with placeholders like {user_input}

    def format(self, **kwargs):
        """Format the template string with provided key-value pairs."""
        # Returns a dictionary like {"role": "user", "content": "formatted text"}
        return {"role": self.role, "content": self.template.format(**kwargs)}

class PromptTemplate:
    def __init__(self, messages: List[MessageTemplate]):
        self.messages = messages

    def format_messages(self, **kwargs):
        """Format all MessageTemplates in the list."""
        return [message.format(**kwargs) for message in self.messages]

# --- API Models ---
class Message(BaseModel):
    role: str
    content: str

class ModelConfigInput(BaseModel):
    # Input model for specifying LLM config in API requests
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: Optional[int] = 1250

class NodeInput(BaseModel):
    # Input model for adding/defining a node via API
    node_id: str
    node_type: str
    input_keys: List[str] = []
    output_keys: List[str] = []
    # Renamed field from model_config to llm_config due to Pydantic V2 conflict
    llm_config: Optional[ModelConfigInput] = None

class EdgeInput(BaseModel):
    # Input model for adding an edge via API
    from_node: str
    to_node: str

class GenerateTextNodeRequest(BaseModel):
    # Input model for the NEW single-node text generation endpoint
    prompt_text: str # The final, already formatted prompt text
    # Renamed from model_config to avoid Pydantic v2 conflict
    llm_config: Optional[ModelConfigInput] = None # Optional config override
    # Change context_data to allow Any value type to accommodate the mapping object
    context_data: Optional[Dict[str, Any]] = None # Map of node names/ids to their outputs/mapping

class GenerateTextNodeResponse(BaseModel):
    # Output model for the NEW single-node text generation endpoint
    generated_text: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost: Optional[float] = None
    duration: Optional[float] = None

class NodeNameUpdate(BaseModel):
    name: str

class TemplateValidationRequest(BaseModel):
    prompt_text: str
    available_nodes: List[str]

class TemplateValidationResponse(BaseModel):
    is_valid: bool
    missing_nodes: List[str]
    found_nodes: List[str]
    warnings: Optional[List[str]] = None 