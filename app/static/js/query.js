let lastChunks = [];
let currentFilter = "all";
let citationsByMarker = {};

function matchesFilter(chunk, filter) {
  if (filter === "all") return true;
  if (filter === "both") return chunk.matched_methods.length === 2;
  return chunk.matched_methods.length === 1 && chunk.matched_methods[0] === filter;
}

function scrollToChunkByMarker(marker) {
  const chunkId = citationsByMarker[marker];
  if (!chunkId) return;
  let card = document.querySelector(`[data-chunk-id="${chunkId}"]`);

  // If card is filtered out, reset filter to "all" and re-render
  if (!card) {
    currentFilter = "all";
    for (const btn of document.querySelectorAll("#filter-toggle button")) {
      btn.setAttribute("aria-pressed", String(btn.dataset.filter === "all"));
    }
    renderChunks();
    card = document.querySelector(`[data-chunk-id="${chunkId}"]`);
  }

  if (!card) return; // Still not found (shouldn't happen with valid citationsByMarker)
  card.scrollIntoView({ behavior: "smooth", block: "center" });
  card.classList.add("highlight");
  setTimeout(() => card.classList.remove("highlight"), 1500);
}

function renderPreferences(preferences) {
  const container = document.getElementById("preferences");
  container.innerHTML = "";
  const entries = [];
  if (preferences.city) entries.push(`City: ${preferences.city}`);
  if (preferences.budget != null) entries.push(`Budget: <= ${preferences.budget}`);
  if (preferences.interests.length > 0) entries.push(`Interests: ${preferences.interests.join(", ")}`);
  for (const entry of entries) {
    const badge = document.createElement("span");
    badge.className = "badge";
    badge.textContent = entry;
    container.appendChild(badge);
  }
}

function renderFilteredNote(count) {
  const container = document.getElementById("filtered-note");
  container.textContent = count > 0 ? `${count} chunk(s) excluded by filter` : "";
}

function renderAnswer(answer) {
  const container = document.getElementById("answer");
  container.innerHTML = "";
  const parts = answer.split(/(\[\d+\])/g);
  for (const part of parts) {
    const match = part.match(/^\[(\d+)\]$/);
    if (match) {
      const marker = document.createElement("span");
      marker.className = "citation-marker";
      marker.textContent = part;
      marker.onclick = () => scrollToChunkByMarker(match[1]);
      container.appendChild(marker);
    } else {
      container.appendChild(document.createTextNode(part));
    }
  }
}

function renderChunks() {
  const container = document.getElementById("chunks");
  container.innerHTML = "";
  for (const chunk of lastChunks) {
    if (!matchesFilter(chunk, currentFilter)) continue;

    const card = document.createElement("div");
    card.className = `chunk-card ${chunk.used_in_synthesis ? "used" : "not-used"}`;
    card.dataset.chunkId = chunk.chunk_id;

    const source = document.createElement("div");
    source.textContent = `${chunk.source_url} — page ${chunk.page_number}`;

    const badges = document.createElement("div");
    const badgeParts = [`<span class="badge">Fused #${chunk.fused_rank} (${chunk.rrf_score.toFixed(3)})</span>`];
    if (chunk.bm25_rank != null) {
      badgeParts.unshift(`<span class="badge">BM25 #${chunk.bm25_rank} (${chunk.bm25_score.toFixed(2)})</span>`);
    }
    if (chunk.semantic_rank != null) {
      badgeParts.splice(1, 0, `<span class="badge">Semantic #${chunk.semantic_rank} (${chunk.semantic_score.toFixed(2)})</span>`);
    }
    badges.innerHTML = badgeParts.join("");

    const text = document.createElement("p");
    text.textContent = chunk.text;

    const usedNote = document.createElement("div");
    usedNote.textContent = chunk.used_in_synthesis ? "Used in answer" : "Retrieved but not used";

    card.appendChild(source);
    card.appendChild(badges);
    card.appendChild(text);
    card.appendChild(usedNote);
    container.appendChild(card);
  }
}

function renderJudgePanel(judgeAttempts) {
  const container = document.getElementById("judge-panel");
  container.innerHTML = "";
  for (const attempt of judgeAttempts) {
    const block = document.createElement("div");
    block.className = "judge-attempt";

    const heading = document.createElement("h3");
    heading.textContent = `Attempt ${attempt.attempt}: ${attempt.verdict}`;

    const raw = document.createElement("p");
    raw.textContent = attempt.raw_response;

    block.appendChild(heading);
    block.appendChild(raw);
    container.appendChild(block);
  }
}

document.getElementById("filter-toggle").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-filter]");
  if (!button) return;
  currentFilter = button.dataset.filter;
  for (const btn of document.querySelectorAll("#filter-toggle button")) {
    btn.setAttribute("aria-pressed", String(btn === button));
  }
  renderChunks();
});

document.getElementById("ask").onclick = async () => {
  const query = document.getElementById("query-input").value.trim();
  if (!query) return;
  const result = await postQuery(query);
  lastChunks = result.retrieved_chunks;
  citationsByMarker = Object.fromEntries(result.citations.map((c) => [String(c.marker), c.chunk_id]));
  renderPreferences(result.preferences);
  renderFilteredNote(result.filtered_out_count);
  renderJudgePanel(result.judge_attempts);
  renderAnswer(result.answer);
  renderChunks();
};
