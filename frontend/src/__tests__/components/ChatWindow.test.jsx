import React from 'react'
import { render, screen } from '@testing-library/react'
import { describe, test, expect, vi } from 'vitest'
import ChatWindow from '../../components/ChatWindow'

// Mock sub-components using correct paths relative to the test file
vi.mock('../../components/MessageBubble', () => ({
  default: ({ message }) => <div data-testid="msg-bubble">{message.content}</div>
}))
vi.mock('../../components/TypingIndicator', () => ({
  default: () => <div data-testid="typing-indicator">Typing...</div>
}))
vi.mock('../../components/ChatModeSelector', () => ({
  default: () => <div data-testid="mode-selector">ModeSelector</div>
}))
vi.mock('../../components/ExportButton', () => ({
  default: () => <button data-testid="export-btn">Export</button>
}))
vi.mock('../../components/ProfileSection', () => ({
  default: () => <div data-testid="profile-sec">Profile</div>
}))

describe('ChatWindow Component', () => {
  const defaultProps = {
    messages: [],
    activeDoc: null,
    activeMode: 'qa',
    onChangeMode: vi.fn(),
    onSendMessage: vi.fn(),
    loading: false,
    onClearHistory: vi.fn(),
    showInsights: false,
    onToggleInsights: vi.fn(),
    username: '',
    fullName: '',
    email: '',
    onLogout: vi.fn(),
    onEditProfile: vi.fn(),
    onCloseChat: vi.fn(),
  }

  test('renders global mode header when activeDoc is null', () => {
    render(<ChatWindow {...defaultProps} />)
    expect(screen.getByText('Global Chat Mode')).toBeInTheDocument()
    expect(screen.getByText('Upload documents and start asking questions')).toBeInTheDocument()
  })

  test('renders active document header when activeDoc is set', () => {
    const activeDoc = { id: 'doc-123', name: 'sample_doc.pdf' }
    render(<ChatWindow {...defaultProps} activeDoc={activeDoc} />)
    expect(screen.getByText('sample_doc.pdf')).toBeInTheDocument()
    expect(screen.getByText('Querying selected document')).toBeInTheDocument()
  })

  test('shows typing indicator when loading is true', () => {
    render(<ChatWindow {...defaultProps} loading={true} />)
    expect(screen.getByTestId('typing-indicator')).toBeInTheDocument()
  })

  test('renders messages correctly', () => {
    const messages = [
      { id: '1', role: 'user', content: 'Hello RAG' },
      { id: '2', role: 'assistant', content: 'Hello User' }
    ]
    render(<ChatWindow {...defaultProps} messages={messages} />)
    const bubbles = screen.getAllByTestId('msg-bubble')
    expect(bubbles.length).toBe(2)
    expect(bubbles[0]).toHaveTextContent('Hello RAG')
    expect(bubbles[1]).toHaveTextContent('Hello User')
  })
})
