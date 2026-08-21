async function renderPipelineStatus() {
  const status = await getPipelineStatus();
  const container = document.getElementById("doc-counts");
  container.innerHTML = "";
  for (const [strategy, count] of Object.entries(status.doc_counts)) {
    const row = document.createElement("div");
    row.className = "row";

    const label = document.createElement("span");
    label.textContent = strategy;

    const badge = document.createElement("span");
    badge.className = "badge";
    badge.textContent = `${count} doc(s) indexed`;

    row.appendChild(label);
    row.appendChild(badge);
    container.appendChild(row);
  }
}

renderPipelineStatus();
