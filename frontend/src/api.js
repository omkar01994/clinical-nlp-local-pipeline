const TIMEOUT_MS = 180_000; // the tone LLM can be slow on CPU

export async function getHealth() {
  const res = await fetch("/api/health");
  if (!res.ok) throw new Error(`Health check failed (${res.status})`);
  return res.json();
}

export async function ask(question, tone) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, tone }),
      signal: controller.signal,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = Array.isArray(data.detail)
        ? data.detail.map((d) => d.msg).join("; ")
        : data.detail;
      throw new Error(detail || `Request failed (${res.status})`);
    }
    return data;
  } catch (err) {
    if (err.name === "AbortError") throw new Error("The request timed out. Please try again.");
    throw err;
  } finally {
    clearTimeout(timer);
  }
}
