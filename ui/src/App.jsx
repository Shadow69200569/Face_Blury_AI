import { useState, useRef } from 'react';
import { UploadCloud, Settings, FileVideo, Download, AlertCircle, PlayCircle } from 'lucide-react';

function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  
  // Settings state
  const [detector, setDetector] = useState('scrfd');
  const [blurType, setBlurType] = useState('gaussian');
  const [blurStrength, setBlurStrength] = useState(99);
  
  // App state
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);

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

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (newFile) => {
    setFile(newFile);
    setResult(null);
    setError(null);
    
    // Create preview URL for images
    if (newFile.type.startsWith('image/')) {
      const url = URL.createObjectURL(newFile);
      setPreviewUrl(url);
    } else {
      setPreviewUrl(null); // Will show generic video icon
    }
  };

  const processMedia = async () => {
    if (!file) return;
    
    setIsProcessing(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('detector', detector);
    formData.append('blur_type', blurType);
    
    // Ensure blur strength is odd
    const safeStrength = blurStrength % 2 === 0 ? blurStrength + 1 : blurStrength;
    formData.append('blur_strength', safeStrength);

    try {
      const response = await fetch('http://127.0.0.1:8000/api/process', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Processing failed');
      }

      setResult({
        ...data,
        url: `http://127.0.0.1:8000${data.download_url}`
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const resetState = () => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>Blurify AI</h1>
        <p>Next-gen privacy & anonymization for your media</p>
      </header>

      <main className="main-content">
        {/* Left Column: Main View */}
        <div className="panel">
          {error && (
            <div className="error-message">
              <AlertCircle size={20} />
              <span>{error}</span>
            </div>
          )}

          {isProcessing ? (
            <div className="loading-overlay">
              <div className="spinner"></div>
              <h3 className="loading-text">Processing Media...</h3>
              <p className="loading-subtext">This may take a few minutes for videos.</p>
            </div>
          ) : result ? (
            <div className="preview-container">
              <h2>Result Ready</h2>
              
              {result.media_type === 'image' ? (
                <img src={result.url} alt="Processed output" className="media-preview" />
              ) : (
                <video src={result.url} controls className="media-preview" />
              )}
              
              <div className="file-info" style={{marginBottom: '2rem'}}>
                <FileVideo size={20} />
                <span style={{flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap'}}>
                  {result.filename}
                </span>
              </div>
              
              <div style={{display: 'flex', gap: '1rem', width: '100%'}}>
                <a href={result.url} download={result.filename} style={{flex: 1, textDecoration: 'none'}}>
                  <button className="btn btn-success">
                    <Download size={20} /> Download
                  </button>
                </a>
                <button className="btn" style={{flex: 1, background: 'var(--bg-tertiary)', color: 'white'}} onClick={resetState}>
                  Process Another
                </button>
              </div>
            </div>
          ) : file ? (
            <div className="preview-container">
              <h2>Ready to Process</h2>
              
              {previewUrl ? (
                <img src={previewUrl} alt="Preview" className="media-preview" style={{maxHeight: '400px', objectFit: 'contain'}} />
              ) : (
                <div className="media-preview" style={{width: '100%', height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--bg-tertiary)'}}>
                  <PlayCircle size={64} color="var(--text-secondary)" />
                </div>
              )}
              
              <div className="file-info">
                <FileVideo size={20} />
                <span style={{flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap'}}>
                  {file.name}
                </span>
                <span style={{color: 'var(--text-secondary)', fontSize: '0.9rem'}}>
                  {(file.size / (1024 * 1024)).toFixed(2)} MB
                </span>
              </div>

              <div style={{display: 'flex', gap: '1rem', width: '100%', marginTop: '1rem'}}>
                <button className="btn btn-primary" onClick={processMedia}>
                  Start Processing
                </button>
                <button className="btn" style={{background: 'var(--bg-tertiary)', color: 'white'}} onClick={resetState}>
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <div 
              className={`dropzone ${dragActive ? 'active' : ''}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                style={{ display: 'none' }}
                accept="video/*,image/*"
                onChange={handleChange}
              />
              <UploadCloud />
              <h3 className="dropzone-text">Click or drag file to this area</h3>
              <p className="dropzone-hint">Supports MP4, AVI, MOV, JPG, PNG, WEBP</p>
            </div>
          )}
        </div>

        {/* Right Column: Settings */}
        <div className="panel settings-panel">
          <h2><Settings size={20} /> Configuration</h2>
          
          <div className="control-group">
            <label>Detector Engine</label>
            <select 
              className="select-input"
              value={detector}
              onChange={(e) => setDetector(e.target.value)}
              disabled={isProcessing}
            >
              <option value="scrfd">SCRFD (Highest Accuracy)</option>
              <option value="retinaface">RetinaFace (Gold Standard)</option>
              <option value="yunet">YuNet (Fastest CPU)</option>
            </select>
          </div>

          <div className="control-group">
            <label>Blur Type</label>
            <select 
              className="select-input"
              value={blurType}
              onChange={(e) => setBlurType(e.target.value)}
              disabled={isProcessing}
            >
              <option value="gaussian">Gaussian (Smooth)</option>
              <option value="pixelate">Pixelation (Mosaic)</option>
              <option value="adaptive">Adaptive (Confidence Based)</option>
            </select>
          </div>

          <div className="control-group">
            <label>Blur Intensity: {blurStrength}</label>
            <input 
              type="range" 
              className="range-input" 
              min="11" 
              max="251" 
              step="2"
              value={blurStrength}
              onChange={(e) => setBlurStrength(parseInt(e.target.value))}
              disabled={isProcessing}
            />
            <div style={{display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.5rem'}}>
              <span>Subtle</span>
              <span>Aggressive</span>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
