import React from "react";
import SourceCard from "./SourceCard";
import ConfidenceMeter from "./ConfidenceMeter";
import TypingIndicator from "./TypingIndicator";

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

export default function MessageBubble({ message }) {
  const isUser = message.role === "user";

  return (
    <div className={`message-wrapper ${message.role}`}>
      <div className="avatar">{isUser ? "U" : "AI"}</div>
      <div className="message-content-container">
        <div className="message-bubble markdown-body">
          {isUser ? (
            <p>{message.content}</p>
          ) : message.content ? (
            renderMarkdown(message.content)
          ) : (
            <TypingIndicator />
          )}
        </div>

        {/* If AI Response has sources and confidence, display them */}
        {!isUser && message.confidence !== undefined && message.confidence > 0 && (
          <ConfidenceMeter confidence={message.confidence} confidenceLabel={message.confidence_label} />
        )}

        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="message-sources">
            {message.sources.map((source, idx) => (
              <SourceCard key={idx} source={source} />
            ))}
          </div>
        )}

        <span className="message-time">
          {message.timestamp || new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </span>
      </div>
    </div>
  );
}
