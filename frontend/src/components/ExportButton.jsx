import React from "react";
import { exportChatToPdf } from "../utils/exportPdf";

export default function ExportButton({ messages, docName, disabled }) {
  const handleExport = () => {
    exportChatToPdf(messages, docName);
  };

  return (
    <button
      className="btn-action-outline"
      onClick={handleExport}
      disabled={disabled || messages.length === 0}
      title="Download chat history as PDF"
    >
      📥 Export Chat PDF
    </button>
  );
}
