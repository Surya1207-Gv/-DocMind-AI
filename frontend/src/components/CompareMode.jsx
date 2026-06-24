import React, { useState } from "react";
import { chatApi } from "../api";
import ReactMarkdown from "react-markdown";

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
            <ReactMarkdown>{comparison.comparison_answer}</ReactMarkdown>
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
