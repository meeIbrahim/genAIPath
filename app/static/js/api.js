async function getPipelineStatus() {
  const response = await fetch("/pipeline/status");
  if (!response.ok) {
    throw new Error(`GET /pipeline/status failed: ${response.status}`);
  }
  return response.json();
}

async function postQuery(query, topK) {
  const response = await fetch("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK ?? null }),
  });
  if (!response.ok) {
    throw new Error(`POST /query failed: ${response.status}`);
  }
  return response.json();
}
