import React, { useState } from "react";

export default function SourceCard({ source, showDocName = false }) {
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

  // A passage stitched together with its following chunk can straddle a page
  // break. Showing only the first page number would send the reader to a page
  // that does not contain the quoted text.
  const pageLabel =
    source.page_end && source.page_end !== source.page
      ? `Pages ${source.page}–${source.page_end}`
      : `Page ${source.page}`;

  // supports_answer is set by claim verification: it distinguishes a passage
  // that actually carries a claim in the answer from one that was merely
  // retrieved alongside it. Only shown when the backend reported it.
  const verified = source.supports_answer === true;

  return (
    <div style={{ display: "flex", flexDirection: "column", width: "100%" }}>
      <div className="source-tag" onClick={() => setExpanded(!expanded)}>
        <span title={showDocName ? source.doc_name : undefined}>
          📄 {showDocName ? `${source.doc_name} · ${pageLabel}` : pageLabel}
        </span>
        {verified && (
          <span
            title="Claim verification found this passage supporting the answer"
            style={{ color: "var(--color-success)", fontWeight: 600 }}
          >
            ✓ Supports answer
          </span>
        )}
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
          {source.section && (
            <div style={{ fontSize: "11px", color: "var(--text-muted)", marginBottom: "6px" }}>
              SECTION: {source.section}
            </div>
          )}
          {source.source_url && (
            <div style={{ fontSize: "11px", marginBottom: "6px", wordBreak: "break-all" }}>
              <a href={source.source_url} target="_blank" rel="noopener noreferrer">
                {source.source_url}
              </a>
            </div>
          )}
          <div style={{ fontStyle: "italic", whiteSpace: "pre-wrap" }}>
            "{source.text}"
          </div>
        </div>
      )}
    </div>
  );
}
