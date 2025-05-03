from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from typing import Optional, Dict, Any, List
import os
from dotenv import load_dotenv
import re # Import re for regex

load_dotenv()

# MongoDB connection string - defaults to localhost
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")

# Create MongoDB client
client = AsyncIOMotorClient(MONGODB_URL)
db = client.albus_db  # Database name

# Collections
nodes_collection = db.nodes
chains_collection = db.chains

async def save_node(node_data: Dict[str, Any]) -> str:
    """Save a node to the database"""
    # Ensure node_id is included in the data
    if "node_id" not in node_data:
        raise ValueError("node_id is required in node_data")
    result = await nodes_collection.insert_one(node_data)
    return str(result.inserted_id)

async def get_node(node_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a node by its ID"""
    return await nodes_collection.find_one({"node_id": node_id})

async def save_chain(chain_data: Dict[str, Any]) -> str:
    """Save a chain to the database"""
    result = await chains_collection.insert_one(chain_data)
    return str(result.inserted_id)

async def get_chain(chain_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a chain by its ID"""
    return await chains_collection.find_one({"_id": chain_id})

async def get_all_chains() -> List[Dict[str, Any]]:
    """Retrieve all chains"""
    chains = []
    async for chain in chains_collection.find():
        chains.append(chain)
    return chains

# Modify get_all_nodes to support searching by name
async def get_all_nodes(name_query: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve all nodes, optionally filtering by name using regex."""
    query = {}
    if name_query:
        # Use case-insensitive regex search on the 'name' field
        escaped_query = re.escape(name_query) 
        query["name"] = {"$regex": escaped_query, "$options": "i"} # Search 'name' field
        
    nodes = []
    # Limit results for search queries to avoid overwhelming output, e.g., 20
    cursor = nodes_collection.find(query).limit(20 if name_query else 0) # Limit only if searching
    async for node in cursor:
        # Ensure _id is converted to string if needed by frontend/Pydantic
        if "_id" in node:
            node["_id"] = str(node["_id"])
        nodes.append(node)
    return nodes

async def update_node(node_id: str, output: Any) -> bool:
    """Update a node's output"""
    result = await nodes_collection.update_one(
        {"node_id": node_id},
        {"$set": {"output": output}}
    )
    return result.modified_count > 0

async def update_node_name(node_id: str, name: str) -> bool:
    """Update the name of a node by its ID."""
    result = await nodes_collection.update_one(
        {"node_id": node_id},
        {"$set": {"name": name}}
    )
    return result.modified_count > 0

async def delete_node(node_id: str) -> bool:
    """Delete a node by its ID"""
    result = await nodes_collection.delete_one({"node_id": node_id})
    return result.deleted_count > 0 