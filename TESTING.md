# DocMind AI - Testing Documentation

This document describes the testing infrastructure, frameworks, and methodologies implemented in the DocMind AI platform.

---

## 1. Backend Testing Suite (`backend/tests/`)

The backend testing suite is built using **pytest** and utilizes FastAPI's `TestClient` for endpoint integration testing. All database calls and LLM/Embedding API queries are fully mocked, allowing the tests to run quickly and reliably without external network connectivity or keys.

### Running Backend Tests
Ensure your python virtual environment is active, then run:

```bash
# In the repository root
$env:PYTHONPATH="."
python -m pytest backend/tests/ -v
```

### Backend Coverage
- **Unit Tests:**
  - `test_unit_auth.py`: Tests password hashing, bcrypt verification, token generation, and signature decay.
  - `test_unit_bm25.py`: Verifies custom BM25 TF-IDF logic, stopping-word exclusion, and CamelCase splitting.
  - `test_unit_chat_engine.py`: Verifies question classifiers, greetings, and temperature logic.
  - `test_unit_pdf_processor.py`: Verifies RecursiveCharacterTextSplitter and chunk page tracking with mocked PDF readers.
- **API Endpoint Tests (FastAPI TestClient):**
  - `test_api_auth.py`: Verifies user registration (Gmail constraint, duplicate prevention) and token retrieval.
  - `test_api_documents.py`: Tests secure file uploads, validation of files, list indexing, and cascading deletes.
  - `test_api_chat.py`: Tests Server-Sent Events (SSE) chat streams, MCQ assessment generation, history tracking, and clear operations.
- **Integration Tests:**
  - `test_integration_rag.py`: Verifies confidence index computations, low-relevance threshold filtering, and LLM fallback transitions.

---

## 2. Frontend Testing Suite (`frontend/src/__tests__/`)

The React application uses **Vitest** + **React Testing Library** + **JSDOM** to test components in virtual browser environments.

### Running Frontend Tests
Navigate to the `frontend` folder and run:

```bash
cd frontend
npm run test
```

### Frontend Coverage
- `ConfidenceMeter.test.jsx`: Confirms exact color triggers, width values, and responsive text mappings for different confidence levels.
- `ChatWindow.test.jsx`: Validates global versus document-scoped view states, loading indicators, and clean bubble message rendering.
- `utils.test.js`: Confirms jsPDF integration builds document pages and downloads exports correctly.

---

## 3. Postman Collection (`postman/`)

A pre-configured Postman JSON collection (`DocMind_API.postman_collection.json`) containing all 10 platform API calls is stored in the repository.

- **Automated Login Script:** Parses the token response and updates the variable `{{jwt_token}}` automatically.
- **Variables:** Includes base URL, JWT token, and doc ID mappings.

---

## 4. CI/CD Pipeline

The `.github/workflows/tests.yml` configuration triggers on every Git push and pull request to `main`. It:
1. Provisions Python and Node.js runners.
2. Installs dependencies.
3. Executes backend tests and outputs coverage reports.
4. Installs frontend dependencies and executes frontend Vitest tests.
