import React from 'react'
import { render, screen } from '@testing-library/react'
import { describe, test, expect } from 'vitest'
import ConfidenceMeter from '../../components/ConfidenceMeter'

describe('ConfidenceMeter Component', () => {
  test('renders high label when confidence is 90', () => {
    render(<ConfidenceMeter confidence={90} />)
    expect(screen.getByText('Confidence:')).toBeInTheDocument()
    expect(screen.getByText('High')).toBeInTheDocument()
  })

  test('renders medium label when confidence is 70', () => {
    render(<ConfidenceMeter confidence={70} />)
    expect(screen.getByText('Confidence:')).toBeInTheDocument()
    expect(screen.getByText('Medium')).toBeInTheDocument()
  })

  test('renders low label when confidence is 40', () => {
    render(<ConfidenceMeter confidence={40} />)
    expect(screen.getByText('Confidence:')).toBeInTheDocument()
    expect(screen.getByText('Low')).toBeInTheDocument()
  })

  test('uses custom confidenceLabel if provided', () => {
    render(<ConfidenceMeter confidence={10} confidenceLabel="Medium" />)
    expect(screen.getByText('Medium')).toBeInTheDocument()
  })
})
