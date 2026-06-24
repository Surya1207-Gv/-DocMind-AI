import React from "react";

export default function ChatModeSelector({ activeMode, onChangeMode }) {
  const modes = [
    { id: "qa", label: "Q&A" },
    { id: "summary", label: "Summary" },
    { id: "deep", label: "Deep" },
    { id: "eli5", label: "ELI5" },
  ];

  return (
    <div className="mode-selector">
      {modes.map((mode) => (
        <button
          key={mode.id}
          className={`mode-btn ${activeMode === mode.id ? "active" : ""}`}
          onClick={() => onChangeMode(mode.id)}
          title={`Switch to ${mode.label} Mode`}
        >
          {mode.label}
        </button>
      ))}
    </div>
  );
}
