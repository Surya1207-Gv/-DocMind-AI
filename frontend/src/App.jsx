import React, { useState, useEffect } from "react";
import Sidebar from "./components/Sidebar";
import ChatWindow from "./components/ChatWindow";
import AnalyticsDash from "./components/AnalyticsDash";
import QuizPanel from "./components/QuizPanel";
import CompareMode from "./components/CompareMode";
import ParticleBackground from "./components/ParticleBackground";
import { documentApi, chatApi, authApi } from "./api";
import "./App.css"; // Auth-specific styling overrides

export default function App() {
  const [token, setToken] = useState(localStorage.getItem("token"));
  const [username, setUsername] = useState(localStorage.getItem("username"));
  const [email, setEmail] = useState(localStorage.getItem("email") || "");
  const [fullName, setFullName] = useState(localStorage.getItem("fullName") || "");
  const [documents, setDocuments] = useState([]);
  const [activeDocId, setActiveDocId] = useState(null);
  const [analytics, setAnalytics] = useState(null);
  
  // Tab control for context panel: "analytics" | "quiz" | "compare"
  const [activeTab, setActiveTab] = useState("analytics");
  
  // Mode control for chat: "qa" | "summary" | "deep" | "eli5"
  const [activeMode, setActiveMode] = useState("qa");
  
  // Track chat histories separately per document
  const [chats, setChats] = useState({}); // { [docId]: [messages] }
  const [chatLoading, setChatLoading] = useState(false);
  const [toasts, setToasts] = useState([]);

  // Auth Form State (Switch between Login and Signup)
  const [isLoginView, setIsLoginView] = useState(true);
  const [authUsername, setAuthUsername] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [authEmail, setAuthEmail] = useState("");
  const [authFullName, setAuthFullName] = useState("");
  const [authConfirmPassword, setAuthConfirmPassword] = useState("");
  const [authError, setAuthError] = useState("");
  const [authLoading, setAuthLoading] = useState(false);

  // Toast Helper
  const addToast = (message, type = "info") => {
    const id = Date.now();
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4000);
  };

  // Force logout handler when intercepting 401
  useEffect(() => {
    const handleForceLogout = () => {
      setToken(null);
      setUsername(null);
      setEmail("");
      setFullName("");
      setDocuments([]);
      setActiveDocId(null);
      setAnalytics(null);
      setChats({});
    };
    window.addEventListener("auth-logout", handleForceLogout);
    return () => window.removeEventListener("auth-logout", handleForceLogout);
  }, []);

  // Fetch documents on load (if token exists)
  const loadDocuments = async () => {
    if (!token) return;
    try {
      const list = await documentApi.list();
      setDocuments(list);
      // Auto-select first doc if list is not empty and none is selected
      if (list.length > 0 && !activeDocId) {
        setActiveDocId(list[0].id);
      }
    } catch (err) {
      if (err.response?.status !== 401) {
        addToast("Backend server offline. Start uvicorn to connect.", "error");
      }
    }
  };

  useEffect(() => {
    loadDocuments();
  }, [token]);

  // Fetch document analytics + persistent chat history when active document changes
  useEffect(() => {
    if (!activeDocId || !token) {
      setAnalytics(null);
      return;
    }
    
    const loadAnalytics = async () => {
      try {
        const data = await documentApi.getAnalytics(activeDocId);
        setAnalytics(data);
      } catch (err) {
        setAnalytics(null);
      }
    };

    const loadChatHistory = async () => {
      try {
        const history = await chatApi.getHistory(activeDocId);
        setChats((prev) => ({
          ...prev,
          [activeDocId]: history,
        }));
      } catch (err) {
        console.error("Failed to load chat history:", err);
      }
    };

    loadAnalytics();
    loadChatHistory();
  }, [activeDocId, token]);

  // Auth Action Handlers
  const handleAuthSubmit = async (e) => {
    e.preventDefault();
    
    if (isLoginView) {
      if (!authUsername.trim() || !authPassword.trim()) {
        setAuthError("Please fill out all fields.");
        return;
      }
    } else {
      if (
        !authUsername.trim() ||
        !authPassword.trim() ||
        !authEmail.trim() ||
        !authFullName.trim() ||
        !authConfirmPassword.trim()
      ) {
        setAuthError("Please fill out all fields.");
        return;
      }
      if (authUsername.trim().length < 3) {
        setAuthError("Username must be at least 3 characters.");
        return;
      }
      if (authFullName.trim().length < 2) {
        setAuthError("Full name must be at least 2 characters.");
        return;
      }
      if (authPassword.trim().length < 4) {
        setAuthError("Password must be at least 4 characters.");
        return;
      }
      if (authPassword !== authConfirmPassword) {
        setAuthError("Passwords do not match.");
        return;
      }
      if (!authEmail.includes("@") || !authEmail.includes(".")) {
        setAuthError("Invalid email format.");
        return;
      }
    }

    setAuthError("");
    setAuthLoading(true);

    try {
      let data;
      if (isLoginView) {
        data = await authApi.login(authUsername, authPassword);
        addToast(`Welcome back, ${data.username}!`, "success");
      } else {
        data = await authApi.register(authUsername, authPassword, authEmail, authFullName);
        addToast("Account created successfully!", "success");
      }

      localStorage.setItem("token", data.access_token);
      localStorage.setItem("username", data.username);
      localStorage.setItem("email", data.email || "");
      localStorage.setItem("fullName", data.full_name || "");
      setToken(data.access_token);
      setUsername(data.username);
      setEmail(data.email || "");
      setFullName(data.full_name || "");
      
      // Reset auth forms
      setAuthUsername("");
      setAuthPassword("");
      setAuthEmail("");
      setAuthFullName("");
      setAuthConfirmPassword("");
    } catch (err) {
      const msg = err.response?.data?.detail || "Authentication failed. Try again.";
      setAuthError(msg);
    } finally {
      setAuthLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("username");
    localStorage.removeItem("email");
    localStorage.removeItem("fullName");
    setToken(null);
    setUsername(null);
    setEmail("");
    setFullName("");
    setDocuments([]);
    setActiveDocId(null);
    setAnalytics(null);
    setChats({});
    addToast("Logged out successfully.", "info");
  };

  // Sidebar Upload Callback Handlers
  const handleUploadStart = () => {
    addToast("Uploading and indexing PDF. Please wait...", "info");
  };

  const handleUploadSuccess = (data) => {
    addToast(`Successfully indexed "${data.document.name}"!`, "success");
    setDocuments((prev) => [...prev, data.document]);
    setActiveDocId(data.document.id);
    setAnalytics(data.analytics);
    setActiveTab("analytics"); // Open dashboard automatically
  };

  const handleUploadError = (message) => {
    addToast(message, "error");
  };

  const handleDeleteDoc = async (docId) => {
    try {
      await documentApi.delete(docId);
      addToast("Document deleted successfully.", "success");
      setDocuments((prev) => prev.filter((d) => d.id !== docId));
      
      // Clean up chats for this document
      setChats((prev) => {
        const copy = { ...prev };
        delete copy[docId];
        return copy;
      });

      if (activeDocId === docId) {
        setActiveDocId(null);
        setAnalytics(null);
      }
    } catch (err) {
      addToast("Failed to delete document.", "error");
    }
  };

  // Clear Persistent History Log in SQLite
  const handleClearHistory = async () => {
    if (!activeDocId) return;
    try {
      await chatApi.clearHistory(activeDocId);
      setChats((prev) => ({
        ...prev,
        [activeDocId]: [],
      }));
      addToast("Chat history cleared.", "success");
    } catch (err) {
      addToast("Failed to clear history.", "error");
    }
  };

  // Chat Actions (Streams via SSE fetch)
  const handleSendMessage = async (text) => {
    if (!activeDocId) return;

    // Create User message object
    const userMsg = {
      id: `user-${Date.now()}-${Math.random()}`,
      role: "user",
      content: text,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    // Initialize AI response bubble with empty content
    const assistantMsg = {
      id: `asst-${Date.now()}-${Math.random()}`,
      role: "assistant",
      content: "",
      confidence: 0,
      sources: [],
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    // Add both user and assistant messages in a single update to prevent batching race conditions
    setChats((prev) => {
      const current = prev[activeDocId] || [];
      return {
        ...prev,
        [activeDocId]: [...current, userMsg, assistantMsg],
      };
    });
    
    setChatLoading(true);

    try {
      // Map React messages format to ChatMessage expected format
      const currentChat = chats[activeDocId] || [];
      const formattedHistory = currentChat.map((m) => ({
        role: m.role,
        content: m.content,
      }));

      // Request stream directly from backend main.py endpoint
      const response = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`,
        },
        body: JSON.stringify({
          question: text,
          doc_ids: [activeDocId],
          history: formattedHistory,
          mode: activeMode,
        }),
      });

      if (!response.ok) {
        if (response.status === 401) {
          localStorage.removeItem("token");
          setToken(null);
          addToast("Session expired. Please log in again.", "error");
          return;
        }
        throw new Error("Streaming call failed.");
      }

      // Read SSE stream chunks
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let partialLine = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const decodedChunk = decoder.decode(value, { stream: true });
        const lines = (partialLine + decodedChunk).split("\n");
        partialLine = lines.pop(); // Keep partial line for next iteration

        for (const line of lines) {
          const cleanLine = line.trim();
          if (cleanLine.startsWith("data: ")) {
            const jsonStr = cleanLine.substring(6).trim();
            if (jsonStr === "[DONE]") break;
            
            try {
              const parsed = JSON.parse(jsonStr);
              if (parsed.type === "metadata") {
                // Update confidence and sources
                setChats((prev) => {
                  const historyCopy = [...prev[activeDocId]];
                  const idx = historyCopy.length - 1;
                  historyCopy[idx] = {
                    ...historyCopy[idx],
                    confidence: parsed.confidence,
                    sources: parsed.sources,
                    content: parsed.content !== undefined ? parsed.content : historyCopy[idx].content,
                  };
                  return { ...prev, [activeDocId]: historyCopy };
                });
              } else if (parsed.type === "token") {
                // Append text chunk token
                setChats((prev) => {
                  const historyCopy = [...prev[activeDocId]];
                  const idx = historyCopy.length - 1;
                  historyCopy[idx] = {
                    ...historyCopy[idx],
                    content: historyCopy[idx].content + parsed.text,
                  };
                  return { ...prev, [activeDocId]: historyCopy };
                });
              }
            } catch (err) {
              // Ignore parse errors on half-written lines
            }
          }
        }
      }
    } catch (err) {
      addToast("Failed to generate response. Check backend connection.", "error");
    } finally {
      setChatLoading(false);
    }
  };

  // Auto-fill chat question from suggested question buttons
  const handleAskSuggestedQuestion = (qText) => {
    handleSendMessage(qText);
  };

  const activeDoc = documents.find((d) => d.id === activeDocId);
  const currentMessages = activeDocId ? chats[activeDocId] || [] : [];

  // Authenticated Screen Rendering Guard
  if (!token) {
    return (
      <div className="login-page-container">
        <ParticleBackground />
        <div className="login-card">
          <div className="login-header">
            <div className="login-logo">🧠</div>
            <h2 className="login-title">{isLoginView ? "Sign In" : "Create Account"}</h2>
            <p className="login-subtitle">
              {isLoginView
                ? "Enter your credentials to enter the platform."
                : "Create a local secure account to begin analyzing PDFs."}
            </p>
          </div>

          <form onSubmit={handleAuthSubmit} className="login-form">
            {!isLoginView && (
              <>
                <div className="login-input-group">
                  <label className="login-label">Full Name</label>
                  <input
                    type="text"
                    className="login-input"
                    placeholder="e.g. John Doe"
                    value={authFullName}
                    onChange={(e) => setAuthFullName(e.target.value)}
                    disabled={authLoading}
                  />
                </div>

                <div className="login-input-group">
                  <label className="login-label">Email Address</label>
                  <input
                    type="email"
                    className="login-input"
                    placeholder="e.g. john@example.com"
                    value={authEmail}
                    onChange={(e) => setAuthEmail(e.target.value)}
                    disabled={authLoading}
                  />
                </div>
              </>
            )}

            <div className="login-input-group">
              <label className="login-label">Username</label>
              <input
                type="text"
                className="login-input"
                placeholder="e.g. admin"
                value={authUsername}
                onChange={(e) => setAuthUsername(e.target.value)}
                disabled={authLoading}
              />
            </div>

            <div className="login-input-group">
              <label className="login-label">Password</label>
              <input
                type="password"
                className="login-input"
                placeholder="••••••••"
                value={authPassword}
                onChange={(e) => setAuthPassword(e.target.value)}
                disabled={authLoading}
              />
            </div>

            {!isLoginView && (
              <div className="login-input-group">
                <label className="login-label">Confirm Password</label>
                <input
                  type="password"
                  className="login-input"
                  placeholder="••••••••"
                  value={authConfirmPassword}
                  onChange={(e) => setAuthConfirmPassword(e.target.value)}
                  disabled={authLoading}
                />
              </div>
            )}

            {authError && (
              <div style={{ color: "var(--color-error)", fontSize: "12px", background: "rgba(239, 68, 68, 0.1)", padding: "10px", borderRadius: "6px", textAlign: "center" }}>
                ⚠️ {authError}
              </div>
            )}

            <button type="submit" className="login-btn" disabled={authLoading}>
              {authLoading ? "Authenticating..." : isLoginView ? "Sign In" : "Create Account"}
            </button>
          </form>

          <div className="login-toggle-text">
            {isLoginView ? "Don't have an account?" : "Already have an account?"}{" "}
            <button
              type="button"
              className="login-toggle-link"
              onClick={() => {
                setIsLoginView(!isLoginView);
                setAuthError("");
              }}
              disabled={authLoading}
            >
              {isLoginView ? "Create an account" : "Sign In instead"}
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="app-container">
      <ParticleBackground />

      <Sidebar
        documents={documents}
        activeDocId={activeDocId}
        setActiveDocId={setActiveDocId}
        onDeleteDoc={handleDeleteDoc}
        onUploadStart={handleUploadStart}
        onUploadSuccess={handleUploadSuccess}
        onUploadError={handleUploadError}
        username={username}
        fullName={fullName}
        email={email}
        onLogout={handleLogout}
      />

      <ChatWindow
        messages={currentMessages}
        activeDoc={activeDoc}
        activeMode={activeMode}
        onChangeMode={setActiveMode}
        onSendMessage={handleSendMessage}
        loading={chatLoading}
        onClearHistory={handleClearHistory}
      />

      <section className="context-panel">
        <header className="panel-header">
          <span className="panel-title">Document Intelligence</span>
        </header>

        <div className="panel-tabs">
          <button
            className={`panel-tab ${activeTab === "analytics" ? "active" : ""}`}
            onClick={() => setActiveTab("analytics")}
          >
            Dashboard
          </button>
          <button
            className={`panel-tab ${activeTab === "quiz" ? "active" : ""}`}
            onClick={() => setActiveTab("quiz")}
          >
            Assessments
          </button>
          <button
            className={`panel-tab ${activeTab === "compare" ? "active" : ""}`}
            onClick={() => setActiveTab("compare")}
          >
            Cross-Compare
          </button>
        </div>

        <div className="panel-content">
          {activeTab === "analytics" && (
            <AnalyticsDash
              analytics={analytics}
              onAskQuestion={handleAskSuggestedQuestion}
            />
          )}

          {activeTab === "quiz" && (
            <QuizPanel
              docId={activeDocId}
              docName={activeDoc ? activeDoc.name : ""}
            />
          )}

          {activeTab === "compare" && (
            <CompareMode documents={documents} />
          )}
        </div>
      </section>

      {/* Toast Notifications */}
      <div className="toast-container">
        {toasts.map((toast) => (
          <div key={toast.id} className={`toast ${toast.type}`}>
            <span>
              {toast.type === "success"
                ? "✅"
                : toast.type === "error"
                ? "❌"
                : "ℹ️"}
            </span>
            <span>{toast.message}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
