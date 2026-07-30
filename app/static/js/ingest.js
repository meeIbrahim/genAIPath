const urlRows = document.getElementById("url-rows");
const progress = document.getElementById("progress");
let pollTimer = null;

function addUrlRow(value = "") {
  const row = document.createElement("div");
  row.className = "row";

  const input = document.createElement("input");
  input.type = "url";
  input.placeholder = "https://example.com/article";
  input.value = value;

  const remove = document.createElement("button");
  remove.textContent = "Remove";
  remove.onclick = () => row.remove();

  row.appendChild(input);
  row.appendChild(remove);
  urlRows.appendChild(row);
}

function collectUrls() {
  return Array.from(urlRows.querySelectorAll("input"))
    .map((input) => input.value.trim())
    .filter((value) => value.length > 0);
}

function renderStatus(status) {
  progress.innerHTML = "";
  for (const urlStatus of status.urls) {
    const row = document.createElement("div");
    row.className = "row";

    const label = document.createElement("span");
    label.textContent = urlStatus.url;

    const badge = document.createElement("span");
    badge.className = `stage-badge ${urlStatus.stage}`;
    const pages = urlStatus.pages_total
      ? `${urlStatus.pages_fetched}/${urlStatus.pages_total}`
      : `${urlStatus.pages_fetched}`;
    badge.textContent = urlStatus.error ? `${urlStatus.stage}: ${urlStatus.error}` : `${urlStatus.stage} (${pages})`;

    row.appendChild(label);
    row.appendChild(badge);
    progress.appendChild(row);
  }
}

async function pollStatus(jobId) {
  const status = await getIngestStatus(jobId);
  renderStatus(status);
  const allTerminal = status.urls.every((u) => u.stage === "done" || u.stage === "error");
  if (allTerminal && pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

document.getElementById("add-row").onclick = () => addUrlRow();
document.getElementById("submit").onclick = async () => {
  const urls = collectUrls();
  if (urls.length === 0) return;
  const { job_id } = await postIngest(urls);
  if (pollTimer) clearInterval(pollTimer);
  await pollStatus(job_id);
  pollTimer = setInterval(() => pollStatus(job_id), 1500);
};

addUrlRow();
