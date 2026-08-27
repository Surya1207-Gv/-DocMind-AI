import React from 'react'
import { render, screen, fireEvent } from '@testing-library/react'
import { describe, test, expect, vi } from 'vitest'
import ChatWindow from '../../components/ChatWindow'

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

const ANCHOR = { id: 'doc-a', name: 'policy-a.pdf' }
const EXTRA = { id: 'doc-b', name: 'policy-b.pdf' }

const baseProps = {
  messages: [],
  activeDoc: ANCHOR,
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

describe('ChatWindow multi-document selection', () => {
  test('a single selected document keeps the original subtitle', () => {
    // The existing single-document workflow must look exactly as it did.
    render(<ChatWindow {...baseProps} selectedDocs={[ANCHOR]} />)
    expect(screen.getByText('Querying selected document')).toBeInTheDocument()
    expect(screen.queryByTestId('selected-docs-bar')).not.toBeInTheDocument()
  })

  test('omitting selectedDocs entirely does not break the header', () => {
    render(<ChatWindow {...baseProps} />)
    expect(screen.getByText('Querying selected document')).toBeInTheDocument()
  })

  test('several selected documents are counted in the subtitle', () => {
    render(<ChatWindow {...baseProps} selectedDocs={[ANCHOR, EXTRA]} />)
    expect(screen.getByText('Querying 2 documents')).toBeInTheDocument()
  })

  test('each selected document is named so the scope is visible', () => {
    render(<ChatWindow {...baseProps} selectedDocs={[ANCHOR, EXTRA]} />)
    const bar = screen.getByTestId('selected-docs-bar')
    expect(bar).toHaveTextContent('policy-a.pdf')
    expect(bar).toHaveTextContent('policy-b.pdf')
  })

  test('an added document can be removed from the chat', () => {
    const onRemoveIncludedDoc = vi.fn()
    render(
      <ChatWindow
        {...baseProps}
        selectedDocs={[ANCHOR, EXTRA]}
        onRemoveIncludedDoc={onRemoveIncludedDoc}
      />
    )
    fireEvent.click(screen.getByLabelText('Remove policy-b.pdf from this chat'))
    expect(onRemoveIncludedDoc).toHaveBeenCalledWith('doc-b')
  })

  test('the anchor document offers no remove control', () => {
    // Removing it would leave the conversation with no owner; switching
    // documents is what the sidebar row is for.
    render(
      <ChatWindow
        {...baseProps}
        selectedDocs={[ANCHOR, EXTRA]}
        onRemoveIncludedDoc={vi.fn()}
      />
    )
    expect(screen.queryByLabelText('Remove policy-a.pdf from this chat')).not.toBeInTheDocument()
  })
})
