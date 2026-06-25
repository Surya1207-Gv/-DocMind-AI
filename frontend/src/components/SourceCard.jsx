import React, { useState } from "react";

export default function SourceCard({ source }) {
  const [expanded, setExpanded] = useState(false);

  const getRelevanceLabel = (rel) => {
    if (rel >= 80) return "High Match";
    if (rel >= 65) return "Medium Match";
    return "Low Match";
  };

  const getRelevanceColor = (rel) => {
    if (rel >= 80) return "var(--color-success)";
    if (rel >= 65) return "var(--color-warning)";
    return "var(--color-error)";
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", width: "100%" }}>
      <div className="source-tag" onClick={() => setExpanded(!expanded)}>
        <span>📄 Page {source.page}</span>
        <span 
          className="source-relevance" 
          style={{ color: getRelevanceColor(source.relevance), fontWeight: 600 }}
        >
          {getRelevanceLabel(source.relevance)}
        </span>
        <span>{expanded ? "▲" : "▼"}</span>
      </div>
      {expanded && (
        <div className="source-expanded-card">
          <div className="source-card-header">
            <span>FROM: {source.doc_name}</span>
            <span style={{ color: getRelevanceColor(source.relevance), fontWeight: 600 }}>
              RELEVANCE: {getRelevanceLabel(source.relevance).toUpperCase()}
            </span>
          </div>
          <div style={{ fontStyle: "italic", whiteSpace: "pre-wrap" }}>
            "{source.text}"
          </div>
        </div>
      )}
    </div>
  );
}
