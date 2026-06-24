import React from "react";

export default function SmartAlerts({ alerts }) {
  if (!alerts || alerts.length === 0) return null;

  const getAlertIcon = (type) => {
    switch (type) {
      case "warning":
        return "⚠️";
      case "date":
        return "📅";
      case "stat":
        return "📈";
      case "insight":
      default:
        return "💡";
    }
  };

  return (
    <div style={{ marginTop: "16px" }}>
      <h4 className="analytics-section-title">Proactive Alerts</h4>
      <div className="alerts-list">
        {alerts.map((alert, idx) => (
          <div key={idx} className={`alert-item ${alert.type}`}>
            <span className="alert-icon">{getAlertIcon(alert.type)}</span>
            <div className="alert-text">
              <span>{alert.content}</span>
              {alert.page && (
                <span
                  style={{
                    display: "block",
                    fontSize: "10px",
                    color: "var(--text-muted)",
                    marginTop: "2px",
                  }}
                >
                  Page Reference: {alert.page}
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
