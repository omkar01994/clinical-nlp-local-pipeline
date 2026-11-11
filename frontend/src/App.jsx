import { useEffect, useRef, useState } from "react";
import { ask, getHealth } from "./api.js";

const TONES = [
  { id: "empathetic", label: "Empathetic" },
  { id: "reassuring", label: "Reassuring" },
  { id: "professional", label: "Professional" },
  { id: "simple", label: "Simple" },
];

const EXAMPLES = [
  "How can I lower my blood pressure?",
  "I have a fever of 39°C, what should I do?",
  "What are the warning signs of a stroke?",
  "I can't sleep at night",
];

// Light formatting for LLM output: "* item" / "- item" -> bullet points.
function formatAnswer(text) {
  return text
    .split("\n")
    .map((line) => line.replace(/^\s*[*-]\s+/, "• "))
    .join("\n");
}

function StatusDot({ health }) {
  if (health === null) return <span className="status">Connecting…</span>;
  if (health.error) return <span className="status down">● Backend offline</span>;
  return health.tone_llm_ready ? (
    <span className="status up">● ClinicalBERT + LLM ready</span>
  ) : (
    <span className="status warn">● LLM not ready (answers won't be styled)</span>
  );
}

function AssistantMessage({ data }) {
  const emergency = data.red_flags.length > 0;
  return (
    <div className={`bubble assistant ${emergency ? "emergency" : ""}`}>
      {emergency && (
        <div className="flag">⚠ Emergency warning signs: {data.red_flags.join(", ")}</div>
      )}
      <div className="answer">{formatAnswer(data.answer)}</div>

      <div className="meta">
        <span className="tag">{data.tone}</span>
        {!data.is_relevant && <span className="tag muted">Not in knowledge base</span>}
        {!data.tone_applied && data.is_relevant && <span className="tag warn">Unstyled</span>}
      </div>
      {data.warnings.map((w) => (
        <div key={w} className="warning">{w}</div>
      ))}

      {data.evidence.length > 0 && (
        <details className="evidence">
          <summary>ClinicalBERT evidence ({data.evidence.length})</summary>
          <ul>
            {data.evidence.map((e, i) => (
              <li key={i}>
                <div className="ev-head">
                  <strong>{e.title}</strong>
                  <span className="score">score {e.score.toFixed(2)}</span>
                </div>
                <div>{e.sentence}</div>
                <div className="source">{e.source}</div>
              </li>
            ))}
          </ul>
        </details>
      )}
      <div className="disclaimer">{data.disclaimer}</div>
    </div>
  );
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [tone, setTone] = useState("empathetic");
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const endRef = useRef(null);

  useEffect(() => {
    const check = () =>
      getHealth()
        .then(setHealth)
        .catch((e) => setHealth({ error: e.message }));
    check();
    const id = setInterval(check, 30_000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function send(text) {
    const question = (text ?? input).trim();
    if (!question || loading) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", text: question }]);
    setLoading(true);
    try {
      const data = await ask(question, tone);
      setMessages((m) => [...m, { role: "assistant", data }]);
    } catch (err) {
      setMessages((m) => [...m, { role: "error", text: err.message }]);
    } finally {
      setLoading(false);
    }
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  return (
    <div className="app">
      <header>
        <div>
          <h1>Clinical Q&amp;A</h1>
          <StatusDot health={health} />
        </div>
        <div className="tones" role="radiogroup" aria-label="Answer tone">
          {TONES.map((t) => (
            <button
              key={t.id}
              role="radio"
              aria-checked={tone === t.id}
              className={tone === t.id ? "active" : ""}
              onClick={() => setTone(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>
      </header>

      <main>
        {messages.length === 0 && (
          <div className="empty">
            <p>Ask a general health question. Answers come only from the local medical knowledge base.</p>
            <div className="examples">
              {EXAMPLES.map((q) => (
                <button key={q} onClick={() => send(q)}>{q}</button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) =>
          m.role === "user" ? (
            <div key={i} className="bubble user">{m.text}</div>
          ) : m.role === "error" ? (
            <div key={i} className="bubble error">⚠ {m.text}</div>
          ) : (
            <AssistantMessage key={i} data={m.data} />
          )
        )}

        {loading && <div className="bubble assistant typing">Thinking<span>.</span><span>.</span><span>.</span></div>}
        <div ref={endRef} />
      </main>

      <footer>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Type your question… (Enter to send, Shift+Enter for a new line)"
          rows={2}
          maxLength={2000}
          disabled={loading}
        />
        <button className="send" onClick={() => send()} disabled={loading || !input.trim()}>
          Send
        </button>
      </footer>
      <p className="footnote">
        Not a substitute for professional medical advice. In an emergency, call your local emergency number.
      </p>
    </div>
  );
}
