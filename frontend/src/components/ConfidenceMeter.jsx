import React from "react";

export default function ConfidenceMeter({ confidence, confidenceLabel }) {
  // Use confidenceLabel if provided; otherwise derive it from raw score
  const label = confidenceLabel || (confidence >= 80 ? "High" : (confidence >= 65 ? "Medium" : "Low"));

  // Determine color based on confidence level
  const getColor = () => {
    if (label === "High") return "var(--color-success)";
    if (label === "Medium") return "var(--color-warning)";
    return "var(--color-error)";
  };

  const widthVal = confidence || (label === "High" ? 90 : (label === "Medium" ? 65 : 35));

  return (
    <div className="confidence-display" title={`AI Confidence: ${label} (Score: ${confidence || 0}%)`}>
      <span>AI Confidence:</span>
      <div className="confidence-bar-bg">
        <div
          className="confidence-bar-fill"
          style={{
            width: `${widthVal}%`,
            backgroundColor: getColor(),
            transition: "width 0.8s ease-in-out",
          }}
        />
      </div>
      <span 
        style={{ 
          fontSize: "11px",
          fontWeight: 600, 
          color: getColor(),
          backgroundColor: `${getColor()}1a`, // very light background
          padding: "2px 8px",
          borderRadius: "4px",
          border: `1px solid ${getColor()}33`,
          textTransform: "uppercase",
          letterSpacing: "0.5px"
        }}
      >
        {label}
      </span>
    </div>
  );
}
