import React, { useState, useEffect, useRef } from "react";

export default function ProfileSection({ username, fullName, email, onLogout, onEditProfile }) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);

  useEffect(() => {
    const handleOutsideClick = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setIsOpen(false);
      }
    };
    if (isOpen) {
      document.addEventListener("click", handleOutsideClick);
    }
    return () => document.removeEventListener("click", handleOutsideClick);
  }, [isOpen]);

  return (
    <div className="header-profile-container" ref={dropdownRef} style={{ position: "relative" }}>
      <button
        className="profile-trigger-btn"
        onClick={() => setIsOpen(!isOpen)}
        style={{
          background: "rgba(255, 255, 255, 0.04)",
          border: "1px solid var(--border-color)",
          borderRadius: "20px",
          padding: "6px 14px",
          color: "var(--text-primary)",
          fontSize: "12px",
          fontWeight: "600",
          cursor: "pointer",
          display: "flex",
          alignItems: "center",
          gap: "6px",
          userSelect: "none",
          transition: "var(--transition-smooth)"
        }}
      >
        👤 <span style={{ textOverflow: "ellipsis", overflow: "hidden", whiteSpace: "nowrap", maxWidth: "90px" }}>{username}</span> ▼
      </button>

      {isOpen && (
        <div
          className="profile-dropdown-card"
          style={{
            position: "absolute",
            top: "100%",
            right: 0,
            marginTop: "8px",
            background: "rgba(20, 21, 26, 0.98)",
            backdropFilter: "blur(12px)",
            border: "1px solid var(--border-color)",
            borderRadius: "var(--radius-md)",
            padding: "16px",
            width: "220px",
            display: "flex",
            flexDirection: "column",
            gap: "12px",
            boxShadow: "var(--glass-shadow)",
            zIndex: 1000,
          }}
        >
          <div style={{ display: "flex", flexDirection: "column", gap: "4px", borderBottom: "1px solid var(--border-color)", paddingBottom: "10px" }}>
            <span style={{ fontWeight: "700", color: "var(--text-primary)", fontSize: "13px" }}>👤 {username}</span>
            {fullName && <span style={{ color: "var(--text-secondary)", fontSize: "11px" }}>{fullName}</span>}
            {email && <span style={{ color: "var(--text-muted)", fontSize: "10px", wordBreak: "break-all" }}>{email}</span>}
          </div>
          <button
            onClick={() => { setIsOpen(false); onEditProfile(); }}
            style={{
              background: "rgba(255, 255, 255, 0.04)",
              border: "1px solid var(--border-color)",
              color: "var(--text-primary)",
              padding: "8px 12px",
              borderRadius: "6px",
              fontSize: "12px",
              fontWeight: "600",
              cursor: "pointer",
              textAlign: "left",
              width: "100%",
              transition: "var(--transition-smooth)"
            }}
          >
            ⚙️ Edit Profile
          </button>
          <button
            className="dashboard-logout-btn"
            onClick={onLogout}
            style={{
              width: "100%",
              justifyContent: "center"
            }}
          >
            Logout 🚪
          </button>
        </div>
      )}
    </div>
  );
}
