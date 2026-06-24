import { jsPDF } from "jspdf";

export const exportChatToPdf = (messages, docName) => {
  const doc = new jsPDF();
  let y = 20;
  
  // Header
  doc.setFont("helvetica", "bold");
  doc.setFontSize(18);
  doc.setTextColor(20, 184, 166); // Accent teal
  doc.text("DocMind AI - Chat Export", 20, y);
  
  y += 8;
  doc.setFont("helvetica", "normal");
  doc.setFontSize(10);
  doc.setTextColor(100, 116, 139); // Muted text
  doc.text(`Document: ${docName || "All Selected Documents"}`, 20, y);
  doc.text(`Date: ${new Date().toLocaleString()}`, 20, y + 5);
  
  y += 15;
  doc.setLineWidth(0.5);
  doc.setDrawColor(226, 232, 240);
  doc.line(20, y, 190, y);
  
  y += 10;
  
  messages.forEach((msg, idx) => {
    // Page breaking logic
    if (y > 270) {
      doc.addPage();
      y = 20;
    }
    
    // Role Label
    doc.setFont("helvetica", "bold");
    doc.setFontSize(11);
    if (msg.role === "user") {
      doc.setTextColor(20, 184, 166); // User color (Teal)
      doc.text("User:", 20, y);
    } else {
      doc.setTextColor(16, 185, 129); // AI color (Emerald)
      doc.text("AI Response:", 20, y);
    }
    
    y += 6;
    
    // Message Content
    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    doc.setTextColor(30, 41, 59); // Primary dark text
    
    const lines = doc.splitTextToSize(msg.content, 170);
    lines.forEach((line) => {
      if (y > 280) {
        doc.addPage();
        y = 20;
      }
      doc.text(line, 20, y);
      y += 5;
    });
    
    y += 5; // Spacing between bubbles
  });
  
  doc.save(`chat_export_${Date.now()}.pdf`);
};
