import { describe, test, expect, vi } from 'vitest'
import { jsPDF } from 'jspdf'
import { exportChatToPdf } from '../utils/exportPdf'

// Mock jsPDF
vi.mock('jspdf', () => {
  const mockJsPDFInstance = {
    setFont: vi.fn(),
    setFontSize: vi.fn(),
    setTextColor: vi.fn(),
    text: vi.fn(),
    setLineWidth: vi.fn(),
    setDrawColor: vi.fn(),
    line: vi.fn(),
    addPage: vi.fn(),
    splitTextToSize: vi.fn((text) => [text]), // Simple split mock
    save: vi.fn(),
  }
  return {
    jsPDF: vi.fn(() => mockJsPDFInstance)
  }
})

describe('exportChatToPdf Utility', () => {
  test('generates and saves PDF successfully', () => {
    const messages = [
      { role: 'user', content: 'What is Generative AI?' },
      { role: 'assistant', content: 'Generative AI is a form of AI...' }
    ]
    const docName = 'AI.pdf'
    
    exportChatToPdf(messages, docName)
    
    // Check jsPDF constructor was called
    expect(jsPDF).toHaveBeenCalled()
    
    // Get mock instance
    const instance = jsPDF.mock.results[0].value
    
    // Verify save was called
    expect(instance.save).toHaveBeenCalled()
    expect(instance.text).toHaveBeenCalledWith('Document: AI.pdf', 20, 28)
  })
})
