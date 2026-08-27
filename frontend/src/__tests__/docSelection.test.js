import { describe, test, expect } from 'vitest'
import { buildChatDocIds, resolveSelectedDocs } from '../utils/docSelection'

const DOCS = [
  { id: 'doc-a', name: 'policy-a.pdf' },
  { id: 'doc-b', name: 'policy-b.pdf' },
  { id: 'doc-c', name: 'notes.md' },
]

describe('buildChatDocIds', () => {
  test('a single-document chat sends exactly the open document', () => {
    // The pre-existing workflow, unchanged.
    expect(buildChatDocIds('doc-a', [], DOCS)).toEqual(['doc-a'])
  })

  test('added documents are appended after the anchor', () => {
    expect(buildChatDocIds('doc-a', ['doc-b', 'doc-c'], DOCS)).toEqual([
      'doc-a', 'doc-b', 'doc-c',
    ])
  })

  test('the anchor is always first', () => {
    // The backend files each turn under doc_ids[0]; if the anchor slipped out of
    // first place the conversation would fork into a different history.
    expect(buildChatDocIds('doc-c', ['doc-a', 'doc-b'], DOCS)[0]).toBe('doc-c')
  })

  test('the anchor is not duplicated when it is also in the included list', () => {
    expect(buildChatDocIds('doc-a', ['doc-a', 'doc-b'], DOCS)).toEqual(['doc-a', 'doc-b'])
  })

  test('repeated ids are de-duplicated', () => {
    expect(buildChatDocIds('doc-a', ['doc-b', 'doc-b'], DOCS)).toEqual(['doc-a', 'doc-b'])
  })

  test('ids that do not belong to this user are dropped', () => {
    // Scoping is enforced server-side; dropping them here is what keeps one
    // stale id from turning the whole request into a 404.
    expect(buildChatDocIds('doc-a', ['someone-elses-doc'], DOCS)).toEqual(['doc-a'])
  })

  test('a deleted anchor yields no selection rather than a doomed request', () => {
    expect(buildChatDocIds('deleted-doc', ['doc-b'], DOCS)).toEqual([])
  })

  test('global mode (no open document) selects nothing', () => {
    // The backend reads an empty selection as "all of MY documents"; it never
    // falls back to every index on disk.
    expect(buildChatDocIds(null, ['doc-a'], DOCS)).toEqual([])
  })

  test('tolerates missing arguments', () => {
    expect(buildChatDocIds('doc-a', undefined, DOCS)).toEqual(['doc-a'])
    expect(buildChatDocIds('doc-a')).toEqual([])
  })
})

describe('resolveSelectedDocs', () => {
  test('resolves ids to documents in order, anchor first', () => {
    const resolved = resolveSelectedDocs(['doc-b', 'doc-a'], DOCS)
    expect(resolved.map((d) => d.name)).toEqual(['policy-b.pdf', 'policy-a.pdf'])
  })

  test('skips ids with no matching document', () => {
    expect(resolveSelectedDocs(['doc-a', 'ghost'], DOCS)).toHaveLength(1)
  })

  test('empty selection resolves to an empty list', () => {
    expect(resolveSelectedDocs([], DOCS)).toEqual([])
  })
})
