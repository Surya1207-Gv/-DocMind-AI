import React from "react";
import ReactMarkdown from "react-markdown";
import SourceCard from "./SourceCard";
import ConfidenceMeter from "./ConfidenceMeter";

export default function MessageBubble({ message }) {
  const isUser = message.role === "user";

  return (
    <div className={`message-wrapper ${message.role}`}>
      <div className="avatar">{isUser ? "U" : "AI"}</div>
      <div className="message-content-container">
        <div className="message-bubble markdown-body">
          {isUser ? (
            <p>{message.content}</p>
          ) : (
            <ReactMarkdown>{message.content}</ReactMarkdown>
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
