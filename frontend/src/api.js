import axios from "axios";

const API_BASE = "http://localhost:8000/api";

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    "Content-Type": "application/json",
  },
});

// Request Interceptor: Attach JWT token dynamically
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem("token");
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response Interceptor: Catch 401 Unauthorized errors and force logout
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      localStorage.removeItem("token");
      localStorage.removeItem("username");
      // Fire custom event to notify React app state to reset
      window.dispatchEvent(new Event("auth-logout"));
    }
    return Promise.reject(error);
  }
);

export const authApi = {
  login: async (username, password) => {
    const response = await api.post("/auth/login", { username, password });
    return response.data;
  },
  register: async (username, password, email, fullName) => {
    const response = await api.post("/auth/register", {
      username,
      password,
      email,
      full_name: fullName,
    });
    return response.data;
  },
  updateProfile: async (username, email, fullName, password = "") => {
    const response = await api.put("/users/me", {
      username,
      email,
      full_name: fullName,
      password,
    });
    return response.data;
  },
};

export const documentApi = {
  list: async () => {
    const response = await api.get("/documents");
    return response.data;
  },
  upload: async (file, onProgress) => {
    const formData = new FormData();
    formData.append("file", file);
    const response = await api.post("/upload", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(percent);
        }
      },
    });
    return response.data;
  },
  delete: async (docId) => {
    const response = await api.delete(`/documents/${docId}`);
    return response.data;
  },
  getAnalytics: async (docId) => {
    const response = await api.get(`/analytics/${docId}`);
    return response.data;
  },
};

export const chatApi = {
  send: async (question, docIds, history, mode = "qa") => {
    const response = await api.post("/chat", {
      question,
      doc_ids: docIds,
      history,
      mode,
    });
    return response.data;
  },
  quiz: async (docId) => {
    const response = await api.post(`/quiz/${docId}`);
    return response.data;
  },
  compare: async (docIds, question) => {
    const response = await api.post("/compare", {
      doc_ids: docIds,
      question,
    });
    return response.data;
  },
  getHistory: async (docId) => {
    const response = await api.get(`/chat/history/${docId}`);
    return response.data;
  },
  clearHistory: async (docId) => {
    const response = await api.delete(`/chat/history/${docId}`);
    return response.data;
  },
  getActiveChats: async () => {
    const response = await api.get("/chats/active");
    return response.data;
  },
};

export default api;
