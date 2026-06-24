import React from "react";

export default function ConfidenceMeter({ confidence }) {
  // Determine color based on confidence level
  const getColor = () => {
    if (confidence >= 80) return "var(--color-success)";
    if (confidence >= 50) return "var(--color-warning)";
    return "var(--color-error)";
  };

  return (
    <div className="confidence-display" title={`Average retrieval similarity score: ${confidence}%`}>
      <span>AI Confidence:</span>
      <div className="confidence-bar-bg">
        <div
          className="confidence-bar-fill"
          style={{
            width: `${confidence}%`,
            backgroundColor: getColor(),
            transition: "width 0.8s ease-in-out",
          }}
        />
      </div>
      <span style={{ fontWeight: 600, color: getColor() }}>{confidence}%</span>
    </div>
  );
}
