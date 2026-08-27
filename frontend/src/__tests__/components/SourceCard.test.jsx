import React from 'react'
import { render, screen, fireEvent } from '@testing-library/react'
import { describe, test, expect } from 'vitest'
import SourceCard from '../../components/SourceCard'

const base = {
  text: 'Audit logs are retained for 30 days.',
  page: 5,
  doc_id: 'd1',
  doc_name: 'policy.pdf',
  relevance: 88,
}

describe('SourceCard citations', () => {
  test('shows a single page when the passage does not cross a page break', () => {
    render(<SourceCard source={base} />)
    expect(screen.getByText(/Page 5/)).toBeInTheDocument()
    expect(screen.queryByText(/Pages 5/)).not.toBeInTheDocument()
  })

  test('shows a page range when the passage was expanded across a page break', () => {
    // Retrieval stitches a chunk together with the following one, which can sit
    // on the next page. Citing only "Page 5" would send the reader to a page
    // that does not contain the quoted text.
    render(<SourceCard source={{ ...base, page_end: 6 }} />)
    expect(screen.getByText(/Pages 5–6/)).toBeInTheDocument()
  })

  test('marks a passage that claim verification found supporting the answer', () => {
    render(<SourceCard source={{ ...base, supports_answer: true }} />)
    expect(screen.getByText(/Supports answer/)).toBeInTheDocument()
  })

  test('does not claim support when verification did not confirm it', () => {
    render(<SourceCard source={{ ...base, supports_answer: false }} />)
    expect(screen.queryByText(/Supports answer/)).not.toBeInTheDocument()
  })

  test('omits the support marker entirely when the backend did not report it', () => {
    render(<SourceCard source={base} />)
    expect(screen.queryByText(/Supports answer/)).not.toBeInTheDocument()
  })

  test('shows the section heading when expanded', () => {
    render(<SourceCard source={{ ...base, section: 'Retention' }} />)
    fireEvent.click(screen.getByText(/Page 5/))
    expect(screen.getByText(/SECTION: Retention/)).toBeInTheDocument()
  })

  test('links back to the origin URL for web-ingested documents', () => {
    render(
      <SourceCard source={{ ...base, source_url: 'https://example.com/policy' }} />
    )
    fireEvent.click(screen.getByText(/Page 5/))
    const link = screen.getByRole('link', { name: 'https://example.com/policy' })
    expect(link).toHaveAttribute('href', 'https://example.com/policy')
    expect(link).toHaveAttribute('rel', expect.stringContaining('noopener'))
  })

  test('names the source document when a turn spans several documents', () => {
    // With multiple documents in scope, "Page 5" alone does not tell the reader
    // which document to open.
    render(<SourceCard source={base} showDocName />)
    expect(screen.getByText(/policy\.pdf · Page 5/)).toBeInTheDocument()
  })

  test('omits the document name in a single-document chat', () => {
    render(<SourceCard source={base} />)
    expect(screen.queryByText(/policy\.pdf ·/)).not.toBeInTheDocument()
    expect(screen.getByText(/Page 5/)).toBeInTheDocument()
  })

  test('combines the document name with a page range', () => {
    render(<SourceCard source={{ ...base, page_end: 6 }} showDocName />)
    expect(screen.getByText(/policy\.pdf · Pages 5–6/)).toBeInTheDocument()
  })

  test('renders the quoted passage when expanded', () => {
    render(<SourceCard source={base} />)
    fireEvent.click(screen.getByText(/Page 5/))
    expect(screen.getByText(/Audit logs are retained for 30 days/)).toBeInTheDocument()
  })
})
