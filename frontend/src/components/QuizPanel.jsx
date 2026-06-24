import React, { useState } from "react";
import { chatApi } from "../api";

export default function QuizPanel({ docId, docName }) {
  const [questions, setQuestions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [quizStarted, setQuizStarted] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [score, setScore] = useState(0);
  const [selectedOption, setSelectedOption] = useState(null);
  const [isAnswered, setIsAnswered] = useState(false);
  const [error, setError] = useState("");

  const startQuiz = async () => {
    setLoading(true);
    setError("");
    try {
      const data = await chatApi.quiz(docId);
      setQuestions(data.questions);
      setQuizStarted(true);
      setCurrentIndex(0);
      setScore(0);
      setSelectedOption(null);
      setIsAnswered(false);
    } catch (err) {
      setError("Failed to generate quiz. Make sure your backend and API keys are active.");
    } finally {
      setLoading(false);
    }
  };

  const handleOptionClick = (option) => {
    if (isAnswered) return;
    setSelectedOption(option);
    setIsAnswered(true);

    const isCorrect = option === questions[currentIndex].correct;
    if (isCorrect) {
      setScore((prev) => prev + 1);
    }
  };

  const handleNext = () => {
    setSelectedOption(null);
    setIsAnswered(false);
    setCurrentIndex((prev) => prev + 1);
  };

  const resetQuiz = () => {
    setQuizStarted(false);
    setQuestions([]);
  };

  if (!docId) {
    return (
      <div style={{ padding: "20px", textAlign: "center", color: "var(--text-muted)" }}>
        Select a document from the sidebar to take a quiz.
      </div>
    );
  }

  if (loading) {
    return (
      <div style={{ padding: "40px", textAlign: "center", color: "var(--text-muted)" }}>
        <div className="logo-icon" style={{ margin: "0 auto 16px auto", width: "40px", height: "40px", fontSize: "20px" }}>⏳</div>
        Analyzing document content to formulate questions...
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: "20px", textAlign: "center", color: "var(--color-error)" }}>
        <p>{error}</p>
        <button className="quiz-gen-btn" onClick={startQuiz}>Try Again</button>
      </div>
    );
  }

  if (!quizStarted) {
    return (
      <div className="quiz-intro-card">
        <h3>📄 Test Your Knowledge</h3>
        <p style={{ fontSize: "13px", color: "var(--text-secondary)", margin: "8px 0 16px 0", lineHeight: "1.4" }}>
          Ready to challenge yourself? Let DocMind AI parse <strong>{docName}</strong> and generate an interactive 10-question MCQ quiz for you.
        </p>
        <button className="quiz-gen-btn" onClick={startQuiz}>
          Generate Quiz
        </button>
      </div>
    );
  }

  // Quiz completion state
  if (currentIndex >= questions.length) {
    const pct = Math.round((score / questions.length) * 100);
    return (
      <div className="quiz-intro-card">
        <h3>🎉 Quiz Completed!</h3>
        <div style={{ fontSize: "36px", fontWeight: "800", color: pct >= 70 ? "var(--color-success)" : "var(--color-warning)", margin: "16px 0" }}>
          {score} / {questions.length}
        </div>
        <p style={{ fontSize: "14px", color: "var(--text-secondary)", marginBottom: "16px" }}>
          You scored a {pct}% on this document assessment!
        </p>
        <div style={{ display: "flex", gap: "10px", justifyContent: "center" }}>
          <button className="quiz-gen-btn" onClick={startQuiz}>
            Retake Quiz
          </button>
          <button className="btn-action-outline" style={{ marginTop: "12px" }} onClick={resetQuiz}>
            Exit
          </button>
        </div>
      </div>
    );
  }

  const currentQ = questions[currentIndex];

  return (
    <div className="quiz-container">
      <div className="quiz-question-card">
        <div className="quiz-progress">
          <span>Question {currentIndex + 1} of {questions.length}</span>
          <span className="entity-type date" style={{ fontSize: "10px" }}>{currentQ.difficulty}</span>
        </div>
        <div className="quiz-question-text">{currentQ.question}</div>
        <div className="quiz-options">
          {currentQ.options.map((option, idx) => {
            let btnClass = "";
            if (isAnswered) {
              if (option === currentQ.correct) {
                btnClass = "correct";
              } else if (option === selectedOption) {
                btnClass = "incorrect";
              }
            } else if (option === selectedOption) {
              btnClass = "selected";
            }

            return (
              <button
                key={idx}
                className={`quiz-option-btn ${btnClass}`}
                onClick={() => handleOptionClick(option)}
                disabled={isAnswered}
              >
                {option}
              </button>
            );
          })}
        </div>

        {isAnswered && (
          <div>
            <div className={`quiz-feedback ${selectedOption === currentQ.correct ? "success" : "error"}`}>
              {selectedOption === currentQ.correct
                ? "✨ Correct! Well done."
                : `❌ Incorrect. The correct answer was "${currentQ.correct}".`}
            </div>
            <p style={{ fontSize: "11px", color: "var(--text-muted)", marginTop: "6px" }}>
              Verifiable on Page {currentQ.page_ref} of the document.
            </p>
            <button
              className="quiz-gen-btn"
              style={{ width: "100%", marginTop: "12px" }}
              onClick={handleNext}
            >
              {currentIndex === questions.length - 1 ? "Finish Quiz" : "Next Question →"}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
