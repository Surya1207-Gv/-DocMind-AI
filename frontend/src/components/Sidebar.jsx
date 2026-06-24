import React from "react";
import UploadZone from "./UploadZone";

export default function Sidebar({
  documents,
  activeDocId,
  setActiveDocId,
  onDeleteDoc,
  onUploadStart,
  onUploadSuccess,
  onUploadError,
  username,
  fullName,
  email,
  onLogout,
}) {
  const formatBytes = (bytes, decimals = 2) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i];
  };

  return (
    <aside className="sidebar">
      <div className="logo-section">
        <div className="logo-icon">🧠</div>
        <span className="logo-text">DocMind AI</span>
      </div>

      {username && (
        <div className="user-profile-badge" style={{ margin: "16px 16px 0 16px", display: "flex", flexDirection: "column", alignItems: "flex-start", gap: "6px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", width: "100%", alignItems: "center" }}>
            <span className="user-username" title={username} style={{ fontWeight: "700" }}>👤 {username}</span>
            <button className="logout-btn" onClick={onLogout}>Logout</button>
          </div>
          {fullName && (
            <div style={{ fontSize: "11px", color: "var(--text-primary)", opacity: 0.9 }}>
              {fullName}
            </div>
          )}
          {email && (
            <div style={{ fontSize: "10px", color: "var(--text-muted)", wordBreak: "break-all" }}>
              {email}
            </div>
          )}
        </div>
      )}

      <div className="doc-list-section">
        <h3 className="doc-list-title">My Documents</h3>
        {documents.length === 0 ? (
          <div style={{ padding: "12px", textAlign: "center", color: "var(--text-muted)", fontSize: "12px" }}>
            No documents uploaded yet. Drop a PDF below to begin analyzing!
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

      <UploadZone
        onUploadStart={onUploadStart}
        onUploadSuccess={onUploadSuccess}
        onUploadError={onUploadError}
      />
    </aside>
  );
}
