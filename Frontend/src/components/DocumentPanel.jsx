import React, { useState, useEffect, useRef } from 'react';
import './DocumentPanel.css';

const DocumentPanel = ({ isOpen, onClose, documents = [], onDocumentSave, onAddDocument }) => {
  const [activeDocIndex, setActiveDocIndex] = useState(0);
  const [content, setContent] = useState('');
  const [name, setName] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [panelWidth, setPanelWidth] = useState(500); // Default width
  const panelRef = useRef(null);
  const resizeStartX = useRef(null);
  const initialWidth = useRef(null);

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

  // Search functionality
  const handleSearch = (e) => {
    const query = e.target.value;
    setSearchQuery(query);
    
    if (!query.trim()) {
      setSearchResults([]);
      return;
    }
    
    // Basic search through the current document
    const searchTermLower = query.toLowerCase();
    if (content) {
      const contentLower = content.toLowerCase();
      const results = [];
      let startIndex = 0;
      
      // Find all occurrences
      while (startIndex < contentLower.length) {
        const foundIndex = contentLower.indexOf(searchTermLower, startIndex);
        if (foundIndex === -1) break;
        
        results.push({
          index: foundIndex,
          text: content.substring(foundIndex, foundIndex + query.length),
          beforeContext: content.substring(Math.max(0, foundIndex - 20), foundIndex),
          afterContext: content.substring(foundIndex + query.length, Math.min(content.length, foundIndex + query.length + 20))
        });
        
        startIndex = foundIndex + 1;
      }
      
      setSearchResults(results);
    }
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
              placeholder="Search..."
            />
            {searchQuery && (
              <span className="search-results-count">
                {searchResults.length} {searchResults.length === 1 ? 'result' : 'results'}
              </span>
            )}
          </div>
        </div>
        
        <textarea
          className="document-editor"
          value={content}
          onChange={(e) => setContent(e.target.value)}
          placeholder="Start typing your document here..."
        />
        
        {searchResults.length > 0 && (
          <div className="search-results">
            {searchResults.slice(0, 5).map((result, index) => (
              <div key={index} className="search-result-item">
                <span className="search-context">{result.beforeContext}</span>
                <span className="search-match">{result.text}</span>
                <span className="search-context">{result.afterContext}</span>
              </div>
            ))}
            {searchResults.length > 5 && (
              <div className="search-more">+ {searchResults.length - 5} more matches</div>
            )}
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