import React from "react";
import SmartAlerts from "./SmartAlerts";

export default function AnalyticsDash({ analytics, onAskQuestion }) {
  if (!analytics) {
    return (
      <div style={{ padding: "20px", textAlign: "center", color: "var(--text-muted)" }}>
        Select a document from the sidebar to view its intelligence dashboard.
      </div>
    );
  }

  return (
    <div className="analytics-dash">
      <div className="analytics-grid">
        <div className="stat-card">
          <span className="stat-value">{analytics.page_count}</span>
          <span className="stat-label">Pages</span>
        </div>
        <div className="stat-card">
          <span className="stat-value">{analytics.read_time_mins}m</span>
          <span className="stat-label">Read Time</span>
        </div>
        <div className="stat-card" style={{ gridColumn: "span 2" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span className="stat-value" style={{ fontSize: "16px" }}>
              {analytics.complexity_score}
            </span>
            <span
              className="entity-type"
              style={{
                background:
                  analytics.complexity_score === "Easy"
                    ? "rgba(16, 185, 129, 0.2)"
                    : analytics.complexity_score === "Medium"
                    ? "rgba(245, 158, 11, 0.2)"
                    : "rgba(239, 68, 68, 0.2)",
                color:
                  analytics.complexity_score === "Easy"
                    ? "#34d399"
                    : analytics.complexity_score === "Medium"
                    ? "#fbbf24"
                    : "#f87171",
              }}
            >
              Complexity
            </span>
          </div>
          <span className="stat-label" style={{ marginTop: "4px" }}>
            Words: {analytics.word_count.toLocaleString()}
          </span>
        </div>
      </div>

      <h4 className="analytics-section-title">Executive Summary</h4>
      <div style={{ marginBottom: "16px" }}>
        {analytics.summary.map((point, idx) => (
          <div key={idx} className="summary-bullet">
            {point}
          </div>
        ))}
      </div>

      <h4 className="analytics-section-title">Key Entities Extracted</h4>
      <div className="entities-container">
        {analytics.entities.map((ent, idx) => (
          <div key={idx} className="entity-badge" title={ent.description}>
            <span className={`entity-type ${ent.type.toLowerCase()}`}>{ent.type}</span>
            <span style={{ fontWeight: 500 }}>{ent.name}</span>
          </div>
        ))}
      </div>

      <SmartAlerts alerts={analytics.alerts} />

      <h4 className="analytics-section-title">Suggested Questions</h4>
      <div className="suggested-qs">
        {analytics.suggested_questions.map((q, idx) => (
          <button key={idx} className="suggested-q-btn" onClick={() => onAskQuestion(q)}>
            💡 "{q}"
          </button>
        ))}
      </div>
    </div>
  );
}
