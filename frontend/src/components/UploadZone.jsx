import React, { useState, useRef } from "react";
import { documentApi } from "../api";

export default function UploadZone({ onUploadSuccess, onUploadStart, onUploadError }) {
  const [dragActive, setDragActive] = useState(false);
  const [progress, setProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      await uploadFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = async (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      await uploadFile(e.target.files[0]);
    }
  };

  const onButtonClick = () => {
    fileInputRef.current.click();
  };

  const uploadFile = async (file) => {
    if (!file.name.endsWith(".pdf")) {
      onUploadError("Only PDF files are supported!");
      return;
    }

    setUploading(true);
    setProgress(0);
    onUploadStart();

    try {
      const data = await documentApi.upload(file, (percent) => {
        setProgress(percent);
      });
      onUploadSuccess(data);
    } catch (err) {
      const errMsg = err.response?.data?.detail || "Upload failed. Please try again.";
      onUploadError(errMsg);
    } finally {
      setUploading(false);
      setProgress(0);
    }
  };

  return (
    <div className="upload-container">
      <form
        className={`upload-zone ${dragActive ? "dragging" : ""}`}
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
        onClick={onButtonClick}
      >
        <input
          ref={fileInputRef}
          type="file"
          className="hidden-file-input"
          style={{ display: "none" }}
          accept=".pdf"
          onChange={handleChange}
          disabled={uploading}
        />
        <div className="upload-icon">📂</div>
        {uploading ? (
          <div>
            <p className="upload-text">Processing Document...</p>
            <div className="upload-progress">
              <div className="progress-bar-bg">
                <div className="progress-bar-fill" style={{ width: `${progress}%` }} />
              </div>
              <span className="doc-size">{progress}%</span>
            </div>
          </div>
        ) : (
          <div>
            <p className="upload-text">Drag & drop your PDF here</p>
            <p className="upload-subtext">or click to browse local files</p>
          </div>
        )}
      </form>
    </div>
  );
}
