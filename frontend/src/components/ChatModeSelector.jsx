import React from "react";

export default function ChatModeSelector({ activeMode, onChangeMode }) {
  const modes = [
    { id: "qa", label: "Q&A", tooltip: "Direct Q&A: Focuses on quick, factual answers retrieved directly from references." },
    { id: "summary", label: "Summary", tooltip: "Summary Mode: Condenses context into structural outlines, highlights, and bullet points." },
    { id: "deep", label: "Deep", tooltip: "Deep Analysis: Broad, exhaustive analysis scanning up to 9 pages/sources." },
    { id: "eli5", label: "ELI5", tooltip: "ELI5 (Explain Like I'm 5): Explains complex concepts using simple analogies and child-friendly wording." },
  ];

  return (
    <div className="mode-selector">
      {modes.map((mode) => (
        <button
          key={mode.id}
          className={`mode-btn ${activeMode === mode.id ? "active" : ""}`}
          onClick={() => onChangeMode(mode.id)}
          title={mode.tooltip}
        >
          {mode.label}
        </button>
      ))}
    </div>
  );
}
