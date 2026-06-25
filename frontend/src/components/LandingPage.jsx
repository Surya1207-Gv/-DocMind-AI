import React, { useEffect, useRef } from "react";
import ParticleBackground from "./ParticleBackground";

const FEATURES = [
  {
    icon: "🔍",
    title: "Precision Smart Search",
    desc: "Uses an intelligent double-engine search to find the exact pages and sentences you need, ensuring every answer is precise and relevant.",
  },
  {
    icon: "🤖",
    title: "Automated Document Analysis",
    desc: "Every time you upload a document, the AI automatically extracts key concepts, generates a summary, highlights alerts, and suggests questions.",
  },
  {
    icon: "💬",
    title: "4-Mode Interactive Chat",
    desc: "Switch between standard Q&A, Summaries, Deep Analysis, or ELI5 (Explain Like I'm 5) mode depending on your depth of study.",
  },
  {
    icon: "📊",
    title: "Confidence Scoring",
    desc: "A live confidence percentage is calculated and shown with every answer, so you always know how reliable the retrieved info is.",
  },
  {
    icon: "📝",
    title: "Interactive Quiz Generator",
    desc: "Instantly generate a multiple-choice quiz from any document to test your knowledge, complete with answer keys and page references.",
  },
  {
    icon: "🔄",
    title: "Multi-Document Comparison",
    desc: "Select two or more files to compare side-by-side. The AI easily highlights agreements, contradictions, and key differences.",
  },
  {
    icon: "📄",
    title: "Collapsible Citations",
    desc: "Every response points to exact pages and paragraphs, allowing you to click and read the direct source quote instantly.",
  },
  {
    icon: "📥",
    title: "Detailed PDF Report Export",
    desc: "Export entire conversation logs, cited sources, and key analysis insights into a clean, print-friendly PDF report with a single click.",
  },
];

export default function LandingPage({ onLogin, onSignup }) {
  const heroRef = useRef(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("lp-visible");
          }
        });
      },
      { threshold: 0.1 }
    );
    document.querySelectorAll(".lp-fade-up").forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, []);

  return (
    <div className="lp-root">
      <ParticleBackground />

      {/* NAV */}
      <nav className="lp-nav">
        <div className="lp-nav-logo">
          <span className="lp-logo-icon">🧠</span>
          <span className="lp-logo-text">DocMind AI</span>
        </div>
        <div className="lp-nav-actions">
          <button className="lp-btn-ghost" onClick={onLogin}>Sign In</button>
          <button className="lp-btn-primary" onClick={onSignup}>Get Started</button>
        </div>
      </nav>

      {/* HERO */}
      <section className="lp-hero" ref={heroRef}>
        <div className="lp-hero-badge">Next-Generation Document Assistant</div>
        <h1 className="lp-hero-title">
          Document Intelligence,<br />
          <span className="lp-gradient-text">Reimagined with AI</span>
        </h1>
        <p className="lp-hero-desc">
          Upload any PDF and get instant answers, key concept extraction, executive summaries,
          interactive quizzes, and multi-document comparison — all powered by advanced AI and smart search.
        </p>
        <div className="lp-hero-cta">
          <button className="lp-btn-primary lp-btn-lg" onClick={onSignup}>
            Start for Free →
          </button>
          <button className="lp-btn-ghost lp-btn-lg" onClick={onLogin}>
            Sign In
          </button>
        </div>
      </section>

      {/* FEATURES GRID */}
      <section className="lp-section">
        <div className="lp-section-tag lp-fade-up">Features</div>
        <h2 className="lp-section-title lp-fade-up">
          Unlock the Power of Your Documents
        </h2>
        <p className="lp-section-sub lp-fade-up">
          Go beyond simple search. DocMind AI extracts, summarizes, compares, and explains your files
          with incredible precision and smart insights.
        </p>
        <div className="lp-features-grid">
          {FEATURES.map((f) => (
            <div className="lp-feature-card lp-fade-up" key={f.title}>
              <div className="lp-feature-icon">{f.icon}</div>
              <div className="lp-feature-title">{f.title}</div>
              <div className="lp-feature-desc">{f.desc}</div>
            </div>
          ))}
        </div>
      </section>

      {/* HOW IT WORKS */}
      <section className="lp-section lp-howto-section">
        <div className="lp-section-tag lp-fade-up">How It Works</div>
        <h2 className="lp-section-title lp-fade-up">From PDF to insight in seconds</h2>
        <div className="lp-steps-row">
          {[
            { n: "01", title: "Upload PDF", desc: "Drag and drop any PDF. The system immediately processes and analyzes the document structure." },
            { n: "02", title: "AI Analysis", desc: "Key insights, summaries, alerts, and suggested questions are auto-generated in the background." },
            { n: "03", title: "Ask Anything", desc: "Chat in any of 4 modes. Every answer streams live with a reliability score and page references." },
            { n: "04", title: "Explore More", desc: "Generate quizzes, compare multiple documents, and export your chat as a PDF report." },
          ].map((step) => (
            <div className="lp-step-card lp-fade-up" key={step.n}>
              <div className="lp-step-number">{step.n}</div>
              <div className="lp-step-title">{step.title}</div>
              <div className="lp-step-desc">{step.desc}</div>
            </div>
          ))}
        </div>
      </section>

      {/* CTA BANNER */}
      <section className="lp-cta-banner lp-fade-up">
        <div className="lp-cta-glow" />
        <h2 className="lp-cta-title">Ready to chat with your documents?</h2>
        <p className="lp-cta-sub">Create a free account and upload your first PDF in under a minute.</p>
        <button className="lp-btn-primary lp-btn-lg" onClick={onSignup}>
          Create Free Account →
        </button>
      </section>

      {/* FOOTER */}
      <footer className="lp-footer">
        <span className="lp-logo-text" style={{ opacity: 0.5 }}>🧠 DocMind AI</span>
        <span style={{ opacity: 0.35, fontSize: "12px" }}>
          Intelligent Document AI • Built by Surya Sasank
        </span>
      </footer>
    </div>
  );
}
