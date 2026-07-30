async function postIngest(urls) {
  const response = await fetch("/ingest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ urls }),
  });
  if (!response.ok) {
    throw new Error(`POST /ingest failed: ${response.status}`);
  }
  return response.json();
}

async function getIngestStatus(jobId) {
  const response = await fetch(`/ingest/${jobId}/status`);
  if (!response.ok) {
    throw new Error(`GET /ingest/${jobId}/status failed: ${response.status}`);
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
