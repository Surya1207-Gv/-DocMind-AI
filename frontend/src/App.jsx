import React, { useState, useEffect } from "react";
import Sidebar from "./components/Sidebar";
import ChatWindow from "./components/ChatWindow";
import AnalyticsDash from "./components/AnalyticsDash";
import QuizPanel from "./components/QuizPanel";
import CompareMode from "./components/CompareMode";
import ParticleBackground from "./components/ParticleBackground";
import LandingPage from "./components/LandingPage";
import { documentApi, chatApi, authApi } from "./api";
import ProfileSection from "./components/ProfileSection";
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

  // View state: "landing" | "auth" | "app"
  const [view, setView] = useState(token ? "app" : "landing");

  // Document intelligence insights panel toggle (spacious chat window style)
  const [showInsights, setShowInsights] = useState(true);

  // Layout resizing states
  const [sidebarWidth, setSidebarWidth] = useState(280);
  const [insightsWidth, setInsightsWidth] = useState(380);
  const [isResizingSidebar, setIsResizingSidebar] = useState(false);
  const [isResizingInsights, setIsResizingInsights] = useState(false);
  
  // Document IDs with active chats
  const [activeChats, setActiveChats] = useState([]);

  // Profile Edit modal states
  const [isProfileModalOpen, setIsProfileModalOpen] = useState(false);
  const [editUsername, setEditUsername] = useState("");
  const [editFullName, setEditFullName] = useState("");
  const [editEmail, setEditEmail] = useState("");
  const [editPassword, setEditPassword] = useState("");
  const [editError, setEditError] = useState("");
  const [editLoading, setEditLoading] = useState(false);

  // Ref to track active resizing
  const resizeState = React.useRef({ isResizingSidebar: false, isResizingInsights: false });

  const handleSidebarResizeStart = (e) => {
    e.preventDefault();
    resizeState.current.isResizingSidebar = true;
    setIsResizingSidebar(true);
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
  };

  const handleInsightsResizeStart = (e) => {
    e.preventDefault();
    resizeState.current.isResizingInsights = true;
    setIsResizingInsights(true);
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
  };

  const handleMouseMove = (e) => {
    if (resizeState.current.isResizingSidebar) {
      const newWidth = Math.max(220, Math.min(450, e.clientX));
      setSidebarWidth(newWidth);
    } else if (resizeState.current.isResizingInsights) {
      const newWidth = Math.max(280, Math.min(600, window.innerWidth - e.clientX));
      setInsightsWidth(newWidth);
    }
  };

  const handleMouseUp = () => {
    resizeState.current.isResizingSidebar = false;
    resizeState.current.isResizingInsights = false;
    setIsResizingSidebar(false);
    setIsResizingInsights(false);
    document.removeEventListener("mousemove", handleMouseMove);
    document.removeEventListener("mouseup", handleMouseUp);
  };

  // Profile update handler
  const handleUpdateProfile = async (e) => {
    e.preventDefault();
    if (!editUsername.trim() || !editEmail.trim() || !editFullName.trim()) {
      setEditError("Please fill out all required fields.");
      return;
    }
    if (editUsername.trim().length < 3) {
      setEditError("Username must be at least 3 characters.");
      return;
    }
    if (editFullName.trim().length < 2) {
      setEditError("Full name must be at least 2 characters.");
      return;
    }
    if (editPassword && editPassword.length < 4) {
      setEditError("Password must be at least 4 characters.");
      return;
    }
    const lowerEmail = editEmail.toLowerCase();
    if (!lowerEmail.endsWith("@gmail.com") && !lowerEmail.endsWith("@google.com") && !lowerEmail.endsWith("@googlemail.com")) {
      setEditError("Profile requires a Google email account (@gmail.com or @google.com).");
      return;
    }

    setEditError("");
    setEditLoading(true);

    try {
      const data = await authApi.updateProfile(
        editUsername.trim(),
        editEmail.trim(),
        editFullName.trim(),
        editPassword
      );

      localStorage.setItem("username", data.username);
      localStorage.setItem("email", data.email || "");
      localStorage.setItem("fullName", data.full_name || "");
      setUsername(data.username);
      setEmail(data.email || "");
      setFullName(data.full_name || "");

      addToast("Profile settings updated successfully!", "success");
      setIsProfileModalOpen(false);
      setEditPassword("");
    } catch (err) {
      const msg = err.response?.data?.detail || "Failed to update profile settings.";
      setEditError(msg);
    } finally {
      setEditLoading(false);
    }
  };

  const handleOpenEditProfile = () => {
    setEditUsername(username || "");
    setEditFullName(fullName || "");
    setEditEmail(email || "");
    setEditPassword("");
    setEditError("");
    setIsProfileModalOpen(true);
  };

  // Adjust body overflow depending on whether the main dashboard or the landing/auth pages are active
  useEffect(() => {
    if (token && view === "app") {
      document.body.style.overflow = "hidden";
      document.body.style.height = "100vh";
    } else {
      document.body.style.overflow = "auto";
      document.body.style.height = "auto";
    }
    return () => {
      document.body.style.overflow = "";
      document.body.style.height = "";
    };
  }, [token, view]);

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
      setView("landing");
    };
    window.addEventListener("auth-logout", handleForceLogout);
    return () => window.removeEventListener("auth-logout", handleForceLogout);
  }, []);

  const loadActiveChats = async () => {
    if (!token) return;
    try {
      const activeIds = await chatApi.getActiveChats();
      setActiveChats(activeIds);
    } catch (err) {
      console.error("Failed to load active chats:", err);
    }
  };

  // Fetch documents on load (if token exists)
  const loadDocuments = async () => {
    if (!token) return;
    try {
      const list = await documentApi.list();
      setDocuments(list);
      loadActiveChats();
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
    
    let isMounted = true;
    let pollTimer = null;
    
    const loadAnalytics = async () => {
      try {
        const data = await documentApi.getAnalytics(activeDocId);
        if (!isMounted) return;
        setAnalytics(data);
        
        // Check if analytics are still in placeholder (processing) state
        const isPlaceholder = data && data.summary && data.summary.some(
          s => s.includes("Analyzing document content") || s.includes("Please wait a moment")
        );
        
        if (isPlaceholder) {
          pollTimer = setTimeout(loadAnalytics, 2500); // Poll every 2.5 seconds
        }
      } catch (err) {
        if (isMounted) setAnalytics(null);
      }
    };

    const loadChatHistory = async () => {
      try {
        const history = await chatApi.getHistory(activeDocId);
        if (!isMounted) return;
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

    return () => {
      isMounted = false;
      if (pollTimer) clearTimeout(pollTimer);
    };
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
      const lowerEmail = authEmail.toLowerCase();
      if (!lowerEmail.endsWith("@gmail.com") && !lowerEmail.endsWith("@google.com") && !lowerEmail.endsWith("@googlemail.com")) {
        setAuthError("Registration requires a Google email account (@gmail.com or @google.com).");
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
      setView("app");
      
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
    setActiveChats([]);
    setView("landing");
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
    loadActiveChats();
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
      loadActiveChats();
    } catch (err) {
      addToast("Failed to delete document.", "error");
    }
  };

  const handleDeleteChat = async (docId) => {
    try {
      await chatApi.clearHistory(docId);
      setChats((prev) => ({
        ...prev,
        [docId]: [],
      }));
      addToast("Chat history deleted.", "success");
      if (activeDocId === docId) {
        setActiveDocId(null);
        setAnalytics(null);
      }
      loadActiveChats();
    } catch (err) {
      addToast("Failed to delete chat history.", "error");
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
      loadActiveChats();
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
      generating: true,
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
                    confidence_label: parsed.confidence_label,
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
      setChats((prev) => {
        if (!activeDocId || !prev[activeDocId]) return prev;
        const historyCopy = [...prev[activeDocId]];
        const idx = historyCopy.length - 1;
        if (idx >= 0 && historyCopy[idx].role === "assistant") {
          historyCopy[idx] = {
            ...historyCopy[idx],
            generating: false,
          };
        }
        return { ...prev, [activeDocId]: historyCopy };
      });
      loadActiveChats();
    }
  };

  useEffect(() => {
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, []);

  // Auto-fill chat question from suggested question buttons
  const handleAskSuggestedQuestion = (qText) => {
    handleSendMessage(qText);
  };

  const activeDoc = documents.find((d) => d.id === activeDocId);
  const currentMessages = activeDocId ? chats[activeDocId] || [] : [];

  // Landing Page
  if (!token && view === "landing") {
    return (
      <LandingPage
        onLogin={() => { setIsLoginView(true); setView("auth"); }}
        onSignup={() => { setIsLoginView(false); setView("auth"); }}
      />
    );
  }

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
                    placeholder="e.g. john@gmail.com"
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

          <div className="login-toggle-text" style={{ marginTop: "4px" }}>
            <button
              type="button"
              className="login-toggle-link"
              style={{ color: "#475569", textDecoration: "none", fontSize: "12px" }}
              onClick={() => setView("landing")}
            >
              ← Back to Home
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`app-container ${showInsights ? "insights-open" : "insights-closed"}`}
      style={{
        gridTemplateColumns: `${sidebarWidth}px 6px 1fr ${showInsights ? `6px ${insightsWidth}px` : ""}`
      }}
    >
      <ParticleBackground />

      <Sidebar
        documents={documents}
        activeDocId={activeDocId}
        setActiveDocId={setActiveDocId}
        onDeleteDoc={handleDeleteDoc}
        onDeleteChat={handleDeleteChat}
        onUploadStart={handleUploadStart}
        onUploadSuccess={handleUploadSuccess}
        onUploadError={handleUploadError}
        activeChats={activeChats}
      />

      <div
        className={`layout-divider sidebar-divider ${isResizingSidebar ? "active" : ""}`}
        onMouseDown={handleSidebarResizeStart}
      />

      <ChatWindow
        messages={currentMessages}
        activeDoc={activeDoc}
        activeMode={activeMode}
        onChangeMode={setActiveMode}
        onSendMessage={handleSendMessage}
        loading={chatLoading}
        onClearHistory={handleClearHistory}
        showInsights={showInsights}
        onToggleInsights={() => setShowInsights(!showInsights)}
        username={username}
        fullName={fullName}
        email={email}
        onLogout={handleLogout}
        onEditProfile={handleOpenEditProfile}
        onCloseChat={() => setActiveDocId(null)}
      />

      {showInsights && (
        <div
          className={`layout-divider insights-divider ${isResizingInsights ? "active" : ""}`}
          onMouseDown={handleInsightsResizeStart}
        />
      )}

      {showInsights && (
        <section className="context-panel">
          <header className="panel-header">
            <span className="panel-title">Document Intelligence</span>
            {username && (
              <ProfileSection
                username={username}
                fullName={fullName}
                email={email}
                onLogout={handleLogout}
                onEditProfile={handleOpenEditProfile}
              />
            )}
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
      )}

      {/* Edit Profile Modal */}
      {isProfileModalOpen && (
        <div className="modal-overlay" style={{
          position: "fixed",
          top: 0,
          left: 0,
          width: "100vw",
          height: "100vh",
          background: "rgba(0, 0, 0, 0.65)",
          backdropFilter: "blur(4px)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          zIndex: 1000,
        }}>
          <div className="modal-card" style={{
            background: "rgba(22, 23, 30, 0.95)",
            backdropFilter: "blur(16px)",
            border: "1px solid var(--border-color)",
            borderRadius: "var(--radius-lg)",
            padding: "32px",
            width: "100%",
            maxWidth: "400px",
            display: "flex",
            flexDirection: "column",
            gap: "20px",
            boxShadow: "var(--glass-shadow)"
          }}>
            <h3 style={{ margin: 0, fontSize: "20px", fontWeight: "800", color: "var(--text-primary)" }}>Edit Profile Settings</h3>
            <form onSubmit={handleUpdateProfile} style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
              <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                <label style={{ fontSize: "11px", fontWeight: "700", color: "var(--text-muted)", textTransform: "uppercase" }}>Full Name</label>
                <input
                  type="text"
                  style={{
                    padding: "10px 12px",
                    background: "rgba(255, 255, 255, 0.03)",
                    border: "1px solid var(--border-color)",
                    borderRadius: "8px",
                    color: "var(--text-primary)",
                    outline: "none"
                  }}
                  value={editFullName}
                  onChange={(e) => setEditFullName(e.target.value)}
                  disabled={editLoading}
                  required
                />
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                <label style={{ fontSize: "11px", fontWeight: "700", color: "var(--text-muted)", textTransform: "uppercase" }}>Email Address</label>
                <input
                  type="email"
                  style={{
                    padding: "10px 12px",
                    background: "rgba(255, 255, 255, 0.03)",
                    border: "1px solid var(--border-color)",
                    borderRadius: "8px",
                    color: "var(--text-primary)",
                    outline: "none"
                  }}
                  value={editEmail}
                  onChange={(e) => setEditEmail(e.target.value)}
                  disabled={editLoading}
                  required
                />
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                <label style={{ fontSize: "11px", fontWeight: "700", color: "var(--text-muted)", textTransform: "uppercase" }}>Username</label>
                <input
                  type="text"
                  style={{
                    padding: "10px 12px",
                    background: "rgba(255, 255, 255, 0.03)",
                    border: "1px solid var(--border-color)",
                    borderRadius: "8px",
                    color: "var(--text-primary)",
                    outline: "none"
                  }}
                  value={editUsername}
                  onChange={(e) => setEditUsername(e.target.value)}
                  disabled={editLoading}
                  required
                />
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                <label style={{ fontSize: "11px", fontWeight: "700", color: "var(--text-muted)", textTransform: "uppercase" }}>New Password (optional)</label>
                <input
                  type="password"
                  style={{
                    padding: "10px 12px",
                    background: "rgba(255, 255, 255, 0.03)",
                    border: "1px solid var(--border-color)",
                    borderRadius: "8px",
                    color: "var(--text-primary)",
                    outline: "none"
                  }}
                  placeholder="Leave blank to keep current"
                  value={editPassword}
                  onChange={(e) => setEditPassword(e.target.value)}
                  disabled={editLoading}
                />
              </div>

              {editError && (
                <div style={{ color: "var(--color-error)", fontSize: "12px", background: "rgba(239, 68, 68, 0.1)", padding: "10px", borderRadius: "6px", textAlign: "center" }}>
                  ⚠️ {editError}
                </div>
              )}

              <div style={{ display: "flex", gap: "12px", marginTop: "10px", justifyContent: "flex-end" }}>
                <button
                  type="button"
                  onClick={() => setIsProfileModalOpen(false)}
                  style={{
                    background: "transparent",
                    border: "1px solid var(--border-color)",
                    color: "var(--text-secondary)",
                    padding: "10px 20px",
                    borderRadius: "8px",
                    cursor: "pointer",
                    fontSize: "13px"
                  }}
                  disabled={editLoading}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  style={{
                    background: "var(--gradient-accent)",
                    border: "none",
                    color: "white",
                    padding: "10px 20px",
                    borderRadius: "8px",
                    cursor: "pointer",
                    fontSize: "13px",
                    fontWeight: "600"
                  }}
                  disabled={editLoading}
                >
                  {editLoading ? "Saving..." : "Save Changes"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

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
