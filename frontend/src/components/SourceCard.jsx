import React, { useState } from "react";

export default function SourceCard({ source }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div style={{ display: "flex", flexDirection: "column", width: "100%" }}>
      <div className="source-tag" onClick={() => setExpanded(!expanded)}>
        <span>📄 Page {source.page}</span>
        <span className="source-relevance">{source.relevance}% Match</span>
        <span>{expanded ? "▲" : "▼"}</span>
      </div>
      {expanded && (
        <div className="source-expanded-card">
          <div className="source-card-header">
            <span>FROM: {source.doc_name}</span>
            <span>RELEVANCE: {source.relevance}%</span>
          </div>
          <div style={{ fontStyle: "italic", whiteSpace: "pre-wrap" }}>
            "{source.text}"
          </div>
        </div>
      )}
    </div>
  );
}
