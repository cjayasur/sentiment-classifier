import { useState } from 'react'
import './App.css'

function App() {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const analyzeSentiment = async () => {
    if (!text.trim()) return

    setLoading(true)
    setError(null)

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.trim() })
      })

      if (!response.ok) throw new Error('API request failed')

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError('Could not connect to API. Make sure the server is running.')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      analyzeSentiment()
    }
  }

  const getSentimentEmoji = (sentiment) => {
    return sentiment === 'positive' ? '😊' : '😞'
  }

  const getConfidenceColor = (confidence) => {
    if (confidence > 0.9) return '#22c55e'
    if (confidence > 0.7) return '#eab308'
    return '#ef4444'
  }

  return (
    <div className="app">
      <header>
        <h1>Sentiment Analyzer</h1>
        <p className="subtitle">Powered by a transformer built from scratch</p>
      </header>

      <main>
        <div className="input-section">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Enter text to analyze sentiment..."
            rows={4}
          />
          <button
            onClick={analyzeSentiment}
            disabled={loading || !text.trim()}
          >
            {loading ? 'Analyzing...' : 'Analyze'}
          </button>
        </div>

        {error && (
          <div className="error">
            {error}
          </div>
        )}

        {result && !error && (
          <div className={`result ${result.sentiment}`}>
            <div className="sentiment-display">
              <span className="emoji">{getSentimentEmoji(result.sentiment)}</span>
              <span className="label">{result.sentiment.toUpperCase()}</span>
            </div>

            <div className="confidence">
              <div className="confidence-bar">
                <div
                  className="confidence-fill"
                  style={{
                    width: `${result.confidence * 100}%`,
                    backgroundColor: getConfidenceColor(result.confidence)
                  }}
                />
              </div>
              <span className="confidence-text">
                {(result.confidence * 100).toFixed(1)}% confident
              </span>
            </div>

            <div className="probabilities">
              <div className="prob-item">
                <span>Positive</span>
                <span>{(result.probabilities.positive * 100).toFixed(1)}%</span>
              </div>
              <div className="prob-item">
                <span>Negative</span>
                <span>{(result.probabilities.negative * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>
        )}
      </main>

      <footer>
        <p>
          Built with PyTorch + FastAPI + React
          <br />
          <span className="tech-details">640K params • 3 layers • 4 attention heads</span>
        </p>
      </footer>
    </div>
  )
}

export default App
