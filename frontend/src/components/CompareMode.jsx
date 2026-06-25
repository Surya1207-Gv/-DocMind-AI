import React, { useState } from "react";
import { chatApi } from "../api";

function renderMarkdown(text) {
  if (!text) return null;
  const lines = text.split("\n");
  const elements = [];
  let currentList = [];

  const parseInline = (str) => {
    if (!str) return "";
    const parts = str.split("**");
    return parts.map((part, idx) => {
      if (idx % 2 === 1) {
        return <strong key={idx}>{part}</strong>;
      }
      return part;
    });
  };

  const flushList = (key) => {
    if (currentList.length > 0) {
      elements.push(<ul key={`list-${key}`} style={{ paddingLeft: "20px", marginBottom: "12px", listStyleType: "disc" }}>{currentList}</ul>);
      currentList = [];
    }
  };

  lines.forEach((line, index) => {
    const trimmed = line.trim();
    if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
      const content = trimmed.substring(2);
      currentList.push(<li key={index} style={{ marginBottom: "4px" }}>{parseInline(content)}</li>);
    } else {
      flushList(index);
      if (line === "") {
        elements.push(<div key={index} style={{ height: "8px" }} />);
      } else if (trimmed.startsWith("### ")) {
        elements.push(<h3 key={index} style={{ margin: "12px 0 6px 0", color: "var(--text-primary)", fontSize: "15px", fontWeight: "700" }}>{parseInline(trimmed.substring(4))}</h3>);
      } else if (trimmed.startsWith("## ")) {
        elements.push(<h2 key={index} style={{ margin: "16px 0 8px 0", color: "var(--text-primary)", fontSize: "17px", fontWeight: "700" }}>{parseInline(trimmed.substring(3))}</h2>);
      } else if (trimmed.startsWith("# ")) {
        elements.push(<h1 key={index} style={{ margin: "20px 0 10px 0", color: "var(--text-primary)", fontSize: "20px", fontWeight: "800" }}>{parseInline(trimmed.substring(2))}</h1>);
      } else {
        elements.push(<p key={index} style={{ marginBottom: "8px", lineHeight: "1.5" }}>{parseInline(line)}</p>);
      }
    }
  });

  flushList(lines.length);
  return elements;
}

export default function CompareMode({ documents }) {
  const [selectedIds, setSelectedIds] = useState([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [comparison, setComparison] = useState(null);
  const [error, setError] = useState("");

  const toggleSelect = (id) => {
    setSelectedIds((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
    );
  };

  const handleCompare = async (e) => {
    e.preventDefault();
    if (selectedIds.length < 2) {
      setError("Please select at least 2 documents to compare.");
      return;
    }
    if (!query.trim()) {
      setError("Please enter a comparison question.");
      return;
    }

    setLoading(true);
    setError("");
    setComparison(null);

    try {
      const data = await chatApi.compare(selectedIds, query);
      setComparison(data);
    } catch (err) {
      setError("Comparison failed. Verify your server is running and Gemini API is valid.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="compare-container">
      <h3>🔍 Multi-Document Comparison</h3>
      <p style={{ fontSize: "12px", color: "var(--text-muted)", lineHeight: "1.4" }}>
        Select 2 or more files and enter a query to draw comparisons between their contents.
      </p>

      <form onSubmit={handleCompare} style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
        <div className="compare-multiselect-list">
          {documents.map((doc) => {
            const isChecked = selectedIds.includes(doc.id);
            return (
              <div
                key={doc.id}
                className={`compare-checkbox-item ${isChecked ? "checked" : ""}`}
                onClick={() => toggleSelect(doc.id)}
              >
                <input
                  type="checkbox"
                  checked={isChecked}
                  onChange={() => {}} // Handled by outer click
                  style={{ marginRight: "8px", cursor: "pointer" }}
                />
                <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {doc.name}
                </span>
              </div>
            );
          })}
        </div>

        <div className="compare-search-wrapper">
          <input
            type="text"
            className="compare-input"
            placeholder="e.g., What are the differences in payment terms?"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={loading}
          />
          <button
            type="submit"
            className="compare-submit-btn"
            disabled={loading || selectedIds.length < 2 || !query.trim()}
          >
            {loading ? "Analyzing..." : "Run Comparison"}
          </button>
        </div>
      </form>

      {error && (
        <div style={{ color: "var(--color-error)", fontSize: "12px", padding: "10px", background: "rgba(239, 68, 68, 0.1)", borderRadius: "6px" }}>
          {error}
        </div>
      )}

      {comparison && (
        <div style={{ display: "flex", flexDirection: "column", gap: "16px", marginTop: "16px" }}>
          <div className="compare-answer-card markdown-body">
            {renderMarkdown(comparison.comparison_answer)}
          </div>

          <h4 className="analytics-section-title">Individual Summaries</h4>
          {comparison.documents.map((doc, idx) => (
            <div key={idx} className="stat-card">
              <span className="doc-name" style={{ color: "var(--text-link)", fontWeight: 600 }}>{doc.doc_name}</span>
              <p style={{ fontSize: "12px", color: "var(--text-secondary)", marginTop: "4px", lineHeight: "1.4" }}>
                {doc.summary}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
