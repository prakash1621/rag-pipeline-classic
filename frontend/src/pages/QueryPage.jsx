import { useState, useRef, useEffect } from 'react';
import api from '../api';

function QueryPage() {
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || loading) return;

    const userMessage = { role: 'user', content: trimmed };
    setMessages((prev) => [...prev, userMessage]);
    setQuestion('');
    setLoading(true);

    try {
      const { data } = await api.post('/api/query', { question: trimmed });
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: data.answer, citations: data.citations },
      ]);
    } catch (err) {
      const errorText =
        err.response?.data?.detail || 'Something went wrong. Please try again.';
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: errorText },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="query-page">
      <h1>Query</h1>

      <div className="chat-container" role="log" aria-live="polite">
        {messages.map((msg, idx) => (
          <div key={idx} className={`chat-message chat-message--${msg.role}`}>
            <span className="chat-role">
              {msg.role === 'user' ? 'You' : 'Assistant'}
            </span>
            <p className="chat-content">{msg.content}</p>
            {msg.role === 'assistant' && msg.citations && (
              <p className="chat-citations">
                <strong>Sources:</strong> {msg.citations}
              </p>
            )}
          </div>
        ))}
        {loading && (
          <div className="chat-message chat-message--assistant">
            <span className="chat-role">Assistant</span>
            <p className="chat-content thinking">Thinking…</p>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>

      <form onSubmit={handleSubmit} className="query-input-area" aria-label="Ask a question">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question…"
          disabled={loading}
          aria-label="Question input"
        />
        <button type="submit" disabled={loading || !question.trim()}>
          Send
        </button>
      </form>
    </div>
  );
}

export default QueryPage;
