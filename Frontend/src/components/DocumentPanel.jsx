import React, { useState, useEffect, useRef } from 'react';
import './DocumentPanel.css';
// Import the API service
import { searchNodes, getNodeDetails } from '../services/api'; 

const DocumentPanel = ({ 
  isOpen, 
  onClose, 
  documents = [], 
  onDocumentSave, 
  onAddDocument,
  // onNodeSelect // Remove unused prop
}) => {
  const [activeDocIndex, setActiveDocIndex] = useState(0);
  const [content, setContent] = useState('');
  const [name, setName] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  // const [searchResults, setSearchResults] = useState([]); // Remove old search results state
  const [nodeSearchResults, setNodeSearchResults] = useState([]); // State for backend node search results
  const [isSearching, setIsSearching] = useState(false); // State to show loading indicator
  const [panelWidth, setPanelWidth] = useState(500); // Default width
  const panelRef = useRef(null);
  const resizeStartX = useRef(null);
  const initialWidth = useRef(null);
  const textareaRef = useRef(null); // Add ref for the textarea

  // Initialize with first document or empty state
  useEffect(() => {
    if (documents && documents.length > 0) {
      setActiveDocIndex(0);
      setContent(documents[0].content || '');
      setName(documents[0].name || '');
    }
  }, [documents]);

  // Update content when active document changes
  useEffect(() => {
    if (documents && documents[activeDocIndex]) {
      setContent(documents[activeDocIndex].content || '');
      setName(documents[activeDocIndex].name || '');
    }
  }, [activeDocIndex, documents]);

  const handleSave = () => {
    if (documents && documents[activeDocIndex]) {
      onDocumentSave(documents[activeDocIndex].id, { name, content });
    }
  };

  const handleAddDocument = () => {
    onAddDocument();
  };

  const handleTabClick = (index) => {
    setActiveDocIndex(index);
  };

  // --- Modified Search Functionality (Placeholder for API call) ---
  const handleSearch = async (e) => {
    const query = e.target.value;
    setSearchQuery(query);
    setNodeSearchResults([]); // Clear previous results

    if (!query.trim()) {
      return; // Don't search if query is empty
    }

    setIsSearching(true);
    try {
      // --- Use the actual API call --- 
      console.log(`Searching for nodes with query: ${query}`);
      const results = await searchNodes(query); 
      setNodeSearchResults(results); 
      console.log('Actual search results:', results);
      // --- End of API call section ---
      
    } catch (error) {
      console.error("Error searching nodes:", error);
      setNodeSearchResults([]); // Clear results on error
    } finally {
      setIsSearching(false);
    }
  };
  
  // Function to handle clicking the 'Insert Output' button
  const handleInsertOutput = async (nodeId) => {
    console.log("Insert output requested for node:", nodeId);
    try {
      const nodeDetails = await getNodeDetails(nodeId);
      if (nodeDetails && typeof nodeDetails.output !== 'undefined') {
        const outputToInsert = String(nodeDetails.output); // Ensure it's a string
        
        if (textareaRef.current) {
          const textarea = textareaRef.current;
          const start = textarea.selectionStart;
          const end = textarea.selectionEnd;
          
          // Insert the text at the cursor position
          const currentContent = content;
          const newContent = currentContent.substring(0, start) + outputToInsert + currentContent.substring(end);
          setContent(newContent);
          
          // Optional: Move cursor to the end of the inserted text
          // Needs to be done slightly deferred after state update
          setTimeout(() => {
            textarea.focus();
            textarea.setSelectionRange(start + outputToInsert.length, start + outputToInsert.length);
          }, 0);
          
          console.log(`Inserted output from ${nodeId}`);
        }
      } else {
        console.warn(`Node ${nodeId} not found or has no output.`);
        // Maybe show a small notification to the user
      }
    } catch (error) {
      console.error("Error fetching node details for insertion:", error);
    }
    // Clear search after attempting insertion
    setSearchQuery(''); 
    setNodeSearchResults([]);
  };

  // Start resizing the panel
  const startResize = (e) => {
    // Record the initial position and panel width
    resizeStartX.current = e.clientX;
    initialWidth.current = panelWidth;
    
    // Add event listeners for mouse movement and release
    document.addEventListener('mousemove', handleResize);
    document.addEventListener('mouseup', stopResize);
    
    // Prevent text selection during resize
    e.preventDefault();
  };

  // Update width during resizing
  const handleResize = (e) => {
    if (resizeStartX.current !== null && initialWidth.current !== null) {
      // Calculate how much the mouse has moved horizontally
      const diff = resizeStartX.current - e.clientX;
      
      // Set new width (moving left increases width, moving right decreases it)
      const newWidth = Math.max(300, Math.min(800, initialWidth.current + diff));
      setPanelWidth(newWidth);
      
      // Update the panel's style directly for smoother resize
      if (panelRef.current) {
        panelRef.current.style.width = `${newWidth}px`;
        panelRef.current.style.right = isOpen ? '0' : `-${newWidth}px`;
      }
    }
  };

  // Stop resizing
  const stopResize = () => {
    resizeStartX.current = null;
    initialWidth.current = null;
    document.removeEventListener('mousemove', handleResize);
    document.removeEventListener('mouseup', stopResize);
  };

  // Update panel styles when width or open state changes
  useEffect(() => {
    if (panelRef.current) {
      panelRef.current.style.width = `${panelWidth}px`;
      panelRef.current.style.right = isOpen ? '0' : `-${panelWidth}px`;
    }
  }, [panelWidth, isOpen]);

  return (
    <div 
      className={`document-panel ${isOpen ? 'open' : ''}`} 
      ref={panelRef}
      style={{ width: `${panelWidth}px`, right: isOpen ? '0' : `-${panelWidth}px` }}
    >
      <div className="resize-handle" onMouseDown={startResize}></div>
      
      <div className="document-panel-header">
        <h3>Document Editor</h3>
        <button className="close-button" onClick={onClose}>×</button>
      </div>

      <div className="document-tabs">
        {documents.map((doc, index) => (
          <button
            key={doc.id}
            className={`document-tab ${index === activeDocIndex ? 'active' : ''}`}
            onClick={() => handleTabClick(index)}
          >
            {doc.name || `Document ${index + 1}`}
          </button>
        ))}
        <button className="add-document-button" onClick={handleAddDocument}>+</button>
      </div>

      <div className="document-content">
        <div className="document-toolbox">
          <input
            type="text"
            className="document-name-input"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Document Name"
          />
          
          <div className="search-container">
            <input
              type="text"
              className="search-input"
              value={searchQuery}
              onChange={handleSearch}
              placeholder="Search Nodes..." // Update placeholder
            />
            {searchQuery && (
              <span className="search-results-count">
                {isSearching ? 'Searching...' : 
                 `${nodeSearchResults.length} ${nodeSearchResults.length === 1 ? 'result' : 'results'}`}
              </span>
            )}
          </div>
        </div>
        
        <textarea
          ref={textareaRef} // Add ref to textarea
          className="document-editor"
          value={content}
          onChange={(e) => setContent(e.target.value)}
          placeholder="Start typing your document here..."
        />
        
        {/* Display Node Search Results */} 
        {nodeSearchResults.length > 0 && (
          <div className="search-results node-search-results"> {/* Add class for specific styling */}
            {nodeSearchResults.map((node) => (
              <div 
                key={node._id || node.node_id} // Use _id if available, fallback to node_id
                className="search-result-item node-result-item" 
              >
                {/* Display node_id as the result */} 
                {node.node_id} 
                <button 
                  className="insert-output-button"
                  onClick={() => handleInsertOutput(node.node_id)} // Call insert handler
                  title={`Insert output of ${node.node_id}`}
                >
                  Insert
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="document-footer">
        <button className="save-button" onClick={handleSave}>Save</button>
      </div>
    </div>
  );
};

export default DocumentPanel; 