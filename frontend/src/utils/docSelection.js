/**
 * Which documents the current conversation searches.
 *
 * The anchor (`activeDocId`) is the conversation itself: chat history,
 * analytics and the export filename are all keyed by it, and the backend
 * persists each turn under `doc_ids[0]`. So the anchor is always present and
 * always FIRST -- that ordering is what keeps a multi-document turn filed under
 * the same thread as a single-document one, rather than forking a new history.
 *
 * Everything is filtered against the documents the server returned for the
 * authenticated user. A stale id (a document deleted in another tab) or an id
 * that never belonged to this user cannot reach the request: the backend would
 * reject the whole call with a 404, so a client-side filter is the difference
 * between "the extra document quietly drops out" and "the chat stops working".
 *
 * @param {string|null} activeDocId  the open document, or null in global mode
 * @param {string[]} includedDocIds  extra documents added to this conversation
 * @param {{id: string}[]} documents this user's documents, as returned by the API
 * @returns {string[]} document ids to send as `doc_ids`, anchor first
 */
export function buildChatDocIds(activeDocId, includedDocIds = [], documents = []) {
  if (!activeDocId) return [];

  const owned = new Set(documents.map((doc) => doc.id));
  if (!owned.has(activeDocId)) return [];

  const extras = [];
  for (const id of includedDocIds) {
    // Skip the anchor (it is already first) and anything not owned, and
    // de-duplicate so a repeated id is not sent twice.
    if (id !== activeDocId && owned.has(id) && !extras.includes(id)) {
      extras.push(id);
    }
  }

  return [activeDocId, ...extras];
}

/**
 * The same selection resolved to document objects, for display.
 * Index 0 is the anchor.
 */
export function resolveSelectedDocs(chatDocIds = [], documents = []) {
  return chatDocIds
    .map((id) => documents.find((doc) => doc.id === id))
    .filter(Boolean);
}
