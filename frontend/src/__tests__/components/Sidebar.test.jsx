import React from 'react'
import { render, screen, fireEvent } from '@testing-library/react'
import { describe, test, expect, vi } from 'vitest'
import Sidebar from '../../components/Sidebar'

const DOCS = [
  { id: 'doc-a', name: 'policy-a.pdf', size: 1024 },
  { id: 'doc-b', name: 'policy-b.pdf', size: 2048 },
  { id: 'doc-c', name: 'notes.md', size: 512 },
]

function renderSidebar(overrides = {}) {
  const props = {
    documents: DOCS,
    activeDocId: 'doc-a',
    setActiveDocId: vi.fn(),
    onDeleteDoc: vi.fn(),
    onDeleteChat: vi.fn(),
    onUploadStart: vi.fn(),
    onUploadSuccess: vi.fn(),
    onUploadError: vi.fn(),
    activeChats: [],
    includedDocIds: [],
    onToggleIncludedDoc: vi.fn(),
    ...overrides,
  }
  render(<Sidebar {...props} />)
  return props
}

const checkboxFor = (name) => screen.getByRole('checkbox', { name: new RegExp(name) })

describe('Sidebar document selection', () => {
  test('renders a selection checkbox for every document', () => {
    renderSidebar()
    expect(screen.getAllByRole('checkbox')).toHaveLength(DOCS.length)
  })

  test('the open document is checked and cannot be unchecked', () => {
    // It is always part of the search; an empty box beside the open document
    // would read as "this one is excluded".
    renderSidebar()
    const anchor = checkboxFor('policy-a.pdf')
    expect(anchor).toBeChecked()
    expect(anchor).toBeDisabled()
  })

  test('toggling another document calls back with its id', () => {
    const props = renderSidebar()
    fireEvent.click(checkboxFor('Also search policy-b.pdf'))
    expect(props.onToggleIncludedDoc).toHaveBeenCalledWith('doc-b')
  })

  test('already-included documents render as checked', () => {
    renderSidebar({ includedDocIds: ['doc-b'] })
    expect(checkboxFor('Also search policy-b.pdf')).toBeChecked()
    expect(checkboxFor('Also search notes.md')).not.toBeChecked()
  })

  test('toggling inclusion does not switch the open conversation', () => {
    // Two different intents live on the same row; the checkbox must not also
    // navigate, or adding a document would silently discard the current thread.
    const props = renderSidebar()
    fireEvent.click(checkboxFor('Also search policy-b.pdf'))
    expect(props.setActiveDocId).not.toHaveBeenCalled()
  })

  test('clicking the row still opens that document (existing workflow)', () => {
    const props = renderSidebar()
    fireEvent.click(screen.getByText('policy-b.pdf'))
    expect(props.setActiveDocId).toHaveBeenCalledWith('doc-b')
  })

  test('checkboxes are disabled until a conversation is open', () => {
    renderSidebar({ activeDocId: null })
    screen.getAllByRole('checkbox').forEach((box) => expect(box).toBeDisabled())
  })

  test('deleting a document still works alongside the checkbox', () => {
    const props = renderSidebar()
    fireEvent.click(screen.getAllByTitle('Delete document')[1])
    expect(props.onDeleteDoc).toHaveBeenCalledWith('doc-b')
  })
})
