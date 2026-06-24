import React, { useState, useRef, useEffect } from "react";
import MessageBubble from "./MessageBubble";
import TypingIndicator from "./TypingIndicator";
import ChatModeSelector from "./ChatModeSelector";
import ExportButton from "./ExportButton";

export default function ChatWindow({
  messages,
  activeDoc,
  activeMode,
  onChangeMode,
  onSendMessage,
  loading,
  onClearHistory,
}) {
  const [input, setInput] = useState("");
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;
    onSendMessage(input.trim());
    setInput("");
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="chat-container">
      <header className="chat-header">
        <div className="header-doc-info">
          <span className="header-title">
            {activeDoc ? activeDoc.name : "Global Chat Mode"}
          </span>
          <span className="header-subtitle">
            {activeDoc
              ? "Querying selected document"
              : "Upload documents and start asking questions"}
          </span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          {activeDoc && (
            <>
              {messages.length > 0 && (
                <button
                  className="btn-action-outline"
                  style={{ fontSize: "11px", padding: "6px 10px", margin: 0 }}
                  onClick={onClearHistory}
                >
                  🧹 Clear Chat
                </button>
              )}
              <ExportButton
                messages={messages}
                docName={activeDoc.name}
                disabled={messages.length === 0}
              />
            </>
          )}
          <ChatModeSelector activeMode={activeMode} onChangeMode={onChangeMode} />
        </div>
      </header>

      <div className="message-list">
        {messages.length === 0 ? (
          <div className="welcome-screen">
            <h1 className="welcome-title">DocMind AI</h1>
            <p className="welcome-desc">
              An advanced Document Intelligence Platform. Upload complex PDF documents
              to extract deep insights, generate MCQs, compare cross-references, and
              chat semantically.
            </p>
            <div className="welcome-features">
              <div className="welcome-feat-card">
                <div className="welcome-feat-icon">📊</div>
                <div className="welcome-feat-title">Intelligence Dashboard</div>
                <div className="welcome-feat-desc">
                  Auto summary, entity extraction, stats and read times.
                </div>
              </div>
              <div className="welcome-feat-card">
                <div className="welcome-feat-icon">📝</div>
                <div className="welcome-feat-title">Auto MCQ Quiz</div>
                <div className="welcome-feat-desc">
                  Generate instant 10-question assessments with citations.
                </div>
              </div>
              <div className="welcome-feat-card">
                <div className="welcome-feat-icon">🔍</div>
                <div className="welcome-feat-title">Multi-Doc Comparison</div>
                <div className="welcome-feat-desc">
                  Highlight differences and match correlations across multiple documents.
                </div>
              </div>
              <div className="welcome-feat-card">
                <div className="welcome-feat-icon">💬</div>
                <div className="welcome-feat-title">4 Chat Modes</div>
                <div className="welcome-feat-desc">
                  Toggle Q&A, detailed analytics, summaries, or ELI5 explanations.
                </div>
              </div>
            </div>
          </div>
        ) : (
          messages.map((msg, idx) => <MessageBubble key={msg.id || idx} message={msg} />)
        )}

        {loading && (
          <div className="message-wrapper assistant">
            <div className="avatar">AI</div>
            <div className="message-content-container">
              <div className="message-bubble" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-color)" }}>
                <TypingIndicator />
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-container">
        <form onSubmit={handleSubmit} className="chat-input-wrapper">
          <textarea
            className="chat-input"
            rows="1"
            placeholder={
              activeDoc
                ? `Ask anything about "${activeDoc.name}"...`
                : "Upload and select a PDF to begin..."
            }
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading || !activeDoc}
          />
          <button
            type="submit"
            className="send-btn"
            disabled={!input.trim() || loading || !activeDoc}
          >
            ➔
          </button>
        </form>
      </div>
    </div>
  );
}
