import React from "react";
import UploadZone from "./UploadZone";

export default function Sidebar({
  documents,
  activeDocId,
  setActiveDocId,
  onDeleteDoc,
  onDeleteChat,
  onUploadStart,
  onUploadSuccess,
  onUploadError,
  activeChats = [],
}) {
  const formatBytes = (bytes, decimals = 2) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i];
  };

  const chatHistoryDocuments = documents.filter((doc) => activeChats.includes(doc.id));

  return (
    <aside className="sidebar">
      <div className="logo-section">
        <div className="logo-icon">🧠</div>
        <span className="logo-text">DocMind AI</span>
      </div>

      {/* Scrollable Container for lists */}
      <div className="sidebar-scroll-container">
        {/* Segment 1: Uploaded Documents */}
        <div className="doc-list-section">
          <h3 className="doc-list-title">Uploaded Documents</h3>
          {documents.length === 0 ? (
            <div style={{ padding: "12px", textAlign: "center", color: "var(--text-muted)", fontSize: "12px" }}>
              No documents uploaded yet.
            </div>
          ) : (
            documents.map((doc) => (
              <div
                key={doc.id}
                className={`doc-item ${activeDocId === doc.id ? "active" : ""}`}
                onClick={() => setActiveDocId(doc.id)}
              >
                <div className="doc-info">
                  <span style={{ fontSize: "16px" }}>📄</span>
                  <div style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
                    <span className="doc-name" title={doc.name}>
                      {doc.name}
                    </span>
                    <span className="doc-size">{formatBytes(doc.size)}</span>
                  </div>
                </div>
                <button
                  className="delete-btn"
                  title="Delete document"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteDoc(doc.id);
                  }}
                >
                  🗑️
                </button>
              </div>
            ))
          )}
        </div>

        {/* Segment 2: Chat History */}
        <div className="chat-history-section">
          <h3 className="doc-list-title">Chat History</h3>
          {chatHistoryDocuments.length === 0 ? (
            <div style={{ padding: "12px", textAlign: "center", color: "var(--text-muted)", fontSize: "12px" }}>
              No active conversations.
            </div>
          ) : (
            chatHistoryDocuments.map((doc) => (
              <div
                key={`chat-${doc.id}`}
                className={`doc-item ${activeDocId === doc.id ? "active" : ""}`}
                onClick={() => setActiveDocId(doc.id)}
              >
                <div className="doc-info">
                  <span style={{ fontSize: "16px" }}>💬</span>
                  <div style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
                    <span className="doc-name" title={doc.name}>
                      {doc.name} Chat
                    </span>
                    <span style={{ fontSize: "10px", color: "var(--text-muted)" }}>Active Thread</span>
                  </div>
                </div>
                <button
                  className="delete-btn"
                  title="Delete chat history"
                  onClick={(e) => {
                    e.stopPropagation();
                    if (onDeleteChat) {
                      onDeleteChat(doc.id);
                    }
                  }}
                >
                  🗑️
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      <UploadZone
        onUploadStart={onUploadStart}
        onUploadSuccess={onUploadSuccess}
        onUploadError={onUploadError}
      />
    </aside>
  );
}
