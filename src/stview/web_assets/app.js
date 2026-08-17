const state = {
  info: null,
  offset: 0,
  limit: 50,
  total: 0,
  sort: "name",
  order: "asc",
  query: "",
  dtype: "",
  rank: "",
  shard: "",
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const formatNumber = (value) => new Intl.NumberFormat("en-US").format(value);

function humanCount(value) {
  const units = ["", "K", "M", "B", "T"];
  let number = Number(value);
  let unit = 0;
  while (Math.abs(number) >= 1000 && unit < units.length - 1) {
    number /= 1000;
    unit += 1;
  }
  return unit === 0 ? formatNumber(number) : `${number.toFixed(2)}${units[unit]}`;
}

function humanBytes(value) {
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let number = Number(value);
  let unit = 0;
  while (Math.abs(number) >= 1024 && unit < units.length - 1) {
    number /= 1024;
    unit += 1;
  }
  return unit === 0 ? `${number} B` : `${number.toFixed(2)} ${units[unit]}`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      message = body.detail || message;
    } catch (_) {}
    throw new Error(message);
  }
  return response.json();
}

async function loadInfo() {
  const info = await api("/api/info");
  state.info = info;
  const sourceParts = info.source.split(/[\\/]/).filter(Boolean);
  $("#model-name").textContent = sourceParts.at(-1) || info.source;
  $("#model-path").textContent = info.source;
  $("#index-badge").textContent = info.index ? "HF SHARDED INDEX" : info.shard_count > 1 ? "MULTI-FILE MODEL" : "SINGLE SAFETENSORS";
  $("#metric-tensors").textContent = formatNumber(info.tensor_count);
  $("#metric-params").textContent = humanCount(info.total_params);
  $("#metric-params-exact").textContent = `${formatNumber(info.total_params)} exact`;
  $("#metric-size").textContent = humanBytes(info.total_size_bytes);
  $("#metric-size-exact").textContent = `${formatNumber(info.total_size_bytes)} bytes`;
  $("#metric-shards").textContent = formatNumber(info.shard_count);
  $("#metric-disk").textContent = `${humanBytes(info.disk_size_bytes)} on disk`;
  document.title = `${sourceParts.at(-1) || "Model"} · Safetensors Viewer`;

  renderDtypes(info.dtypes);
  renderRanks(info.ranks);
  renderShards(info.shards, info.total_size_bytes);
  populateFilters(info);
}

function renderDtypes(dtypes) {
  const entries = Object.entries(dtypes);
  const total = entries.reduce((sum, [, count]) => sum + count, 0);
  $("#dtype-bars").innerHTML = entries.map(([name, count]) => {
    const percent = total ? (count / total) * 100 : 0;
    return `<div>
      <div class="bar-label"><strong>${escapeHtml(name)}</strong><span>${formatNumber(count)} · ${percent.toFixed(1)}%</span></div>
      <div class="bar-track"><div class="bar-fill" style="width:${Math.max(percent, 0.5)}%"></div></div>
    </div>`;
  }).join("");
}

function renderRanks(ranks) {
  $("#rank-list").innerHTML = Object.entries(ranks).map(([rank, count]) => `
    <div class="rank-item"><span>${escapeHtml(rank)}D tensors</span><strong>${formatNumber(count)}</strong></div>
  `).join("");
}

function renderShards(shards, totalSize) {
  $("#shard-list").innerHTML = shards.map((shard) => {
    const percent = totalSize ? (shard.data_size / totalSize) * 100 : 0;
    return `<article class="shard-card">
      <div class="shard-top"><strong title="${escapeHtml(shard.name)}">${escapeHtml(shard.name)}</strong><span>${percent.toFixed(1)}%</span></div>
      <div class="shard-meter"><div style="width:${Math.max(percent, 0.5)}%"></div></div>
      <div class="shard-stats"><span>${formatNumber(shard.tensor_count)} tensors</span><span>${humanBytes(shard.data_size)} data</span><span>${humanBytes(shard.file_size)} file</span></div>
    </article>`;
  }).join("");
}

function populateFilters(info) {
  $("#dtype-filter").insertAdjacentHTML("beforeend", Object.keys(info.dtypes).map((dtype) => `<option value="${escapeHtml(dtype)}">${escapeHtml(dtype)}</option>`).join(""));
  $("#rank-filter").insertAdjacentHTML("beforeend", Object.keys(info.ranks).map((rank) => `<option value="${escapeHtml(rank)}">${escapeHtml(rank)}D</option>`).join(""));
  $("#shard-filter").insertAdjacentHTML("beforeend", info.shards.map((shard) => `<option value="${escapeHtml(shard.name)}">${escapeHtml(shard.name)}</option>`).join(""));
}

async function loadModules() {
  const modules = await api("/api/modules?depth=2");
  const top = [...modules].sort((a, b) => b.params - a.params).slice(0, 12);
  $("#module-list").innerHTML = top.map((module) => `
    <div class="module-item" title="${escapeHtml(module.name)}">
      <span class="module-name">${escapeHtml(module.name)}</span>
      <span class="module-meta"><span>${formatNumber(module.tensor_count)} tensors</span><span>${humanBytes(module.size_bytes)}</span></span>
    </div>
  `).join("");
}

async function loadTensors() {
  $("#tensor-body").innerHTML = '<tr><td colspan="6" class="loading-row">Loading tensor metadata…</td></tr>';
  const params = new URLSearchParams({
    offset: state.offset,
    limit: state.limit,
    sort: state.sort,
    order: state.order,
  });
  if (state.query) params.set("q", state.query);
  if (state.dtype) params.set("dtype", state.dtype);
  if (state.rank !== "") params.set("rank", state.rank);
  if (state.shard) params.set("shard", state.shard);
  try {
    const result = await api(`/api/tensors?${params}`);
    state.total = result.total;
    renderTensorRows(result.items);
    updatePagination();
  } catch (error) {
    $("#tensor-body").innerHTML = `<tr><td colspan="6" class="empty-row">${escapeHtml(error.message)}</td></tr>`;
  }
}

function renderTensorRows(items) {
  if (!items.length) {
    $("#tensor-body").innerHTML = '<tr><td colspan="6" class="empty-row">No tensors match these filters.</td></tr>';
    return;
  }
  $("#tensor-body").innerHTML = items.map((tensor) => `
    <tr class="data-row" data-name="${escapeHtml(tensor.name)}" tabindex="0">
      <td class="name" title="${escapeHtml(tensor.name)}">${escapeHtml(tensor.name)}</td>
      <td class="shape">[${tensor.shape.map(formatNumber).join(", ")}]</td>
      <td><span class="dtype-pill">${escapeHtml(tensor.dtype)}</span></td>
      <td class="number">${formatNumber(tensor.params)}</td>
      <td class="number">${humanBytes(tensor.size_bytes)}</td>
      <td><span class="shard-name" title="${escapeHtml(tensor.shard)}">${escapeHtml(tensor.shard)}</span></td>
    </tr>
  `).join("");
  $$("#tensor-body .data-row").forEach((row) => {
    row.addEventListener("click", () => openTensor(row.dataset.name));
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") openTensor(row.dataset.name);
    });
  });
}

function updatePagination() {
  const start = state.total === 0 ? 0 : state.offset + 1;
  const end = Math.min(state.offset + state.limit, state.total);
  $("#result-count").textContent = `${formatNumber(state.total)} matching tensors`;
  $("#page-range").textContent = `${formatNumber(start)}–${formatNumber(end)} of ${formatNumber(state.total)}`;
  $("#prev-page").disabled = state.offset === 0;
  $("#next-page").disabled = state.offset + state.limit >= state.total;
}

async function openTensor(name) {
  try {
    const tensor = await api(`/api/tensors/${encodeURIComponent(name)}`);
    $("#detail-name").textContent = tensor.name;
    $("#detail-grid").innerHTML = [
      ["Shape", `[${tensor.shape.map(formatNumber).join(", ")}]`],
      ["Dtype", tensor.dtype],
      ["Rank", tensor.rank],
      ["Parameters", formatNumber(tensor.params)],
      ["Data size", `${humanBytes(tensor.size_bytes)} · ${formatNumber(tensor.size_bytes)} bytes`],
      ["Data offsets", `[${tensor.data_offsets.map(formatNumber).join(", ")}]`],
      ["Shard", tensor.shard, true],
      ["Path", tensor.path, true],
    ].map(([label, value, wide]) => `<div class="detail-item${wide ? " wide" : ""}"><span>${label}</span><strong>${escapeHtml(value)}</strong></div>`).join("");
    $("#tensor-dialog").showModal();
  } catch (error) {
    showToast(error.message);
  }
}

async function runVerification() {
  const button = $("#verify-button");
  const resultBox = $("#verify-result");
  button.disabled = true;
  button.textContent = "Verifying…";
  resultBox.hidden = true;
  try {
    const result = await api("/api/verify", { method: "POST" });
    resultBox.className = `verify-result ${result.ok ? "ok" : "fail"}`;
    const headline = result.ok
      ? `✓ ${formatNumber(result.tensor_count)} tensors across ${result.shard_count} shard(s) passed verification.`
      : `Verification found ${result.errors} error(s) and ${result.warnings} warning(s).`;
    const issues = result.issues.length
      ? `<ul class="issue-list">${result.issues.slice(0, 20).map((issue) => `<li>${escapeHtml(issue.level.toUpperCase())}: ${escapeHtml(issue.message)}</li>`).join("")}</ul>`
      : "";
    resultBox.innerHTML = `<strong>${headline}</strong>${issues}`;
    resultBox.hidden = false;
  } catch (error) {
    resultBox.className = "verify-result fail";
    resultBox.textContent = error.message;
    resultBox.hidden = false;
  } finally {
    button.disabled = false;
    button.textContent = "Run verification";
  }
}

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove("show"), 1800);
}

function bindEvents() {
  let searchTimer;
  $("#search").addEventListener("input", (event) => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      state.query = event.target.value.trim();
      state.offset = 0;
      loadTensors();
    }, 180);
  });
  [["#dtype-filter", "dtype"], ["#rank-filter", "rank"], ["#shard-filter", "shard"]].forEach(([selector, key]) => {
    $(selector).addEventListener("change", (event) => {
      state[key] = event.target.value;
      state.offset = 0;
      loadTensors();
    });
  });
  $$('[data-sort]').forEach((button) => button.addEventListener("click", () => {
    const sort = button.dataset.sort;
    state.order = state.sort === sort && state.order === "asc" ? "desc" : "asc";
    state.sort = sort;
    state.offset = 0;
    $$('[data-sort]').forEach((item) => item.classList.toggle("active", item === button));
    loadTensors();
  }));
  $("#prev-page").addEventListener("click", () => {
    state.offset = Math.max(0, state.offset - state.limit);
    loadTensors();
  });
  $("#next-page").addEventListener("click", () => {
    state.offset += state.limit;
    loadTensors();
  });
  $("#verify-button").addEventListener("click", runVerification);
  $("#close-dialog").addEventListener("click", () => $("#tensor-dialog").close());
  $("#tensor-dialog").addEventListener("click", (event) => {
    if (event.target === $("#tensor-dialog")) $("#tensor-dialog").close();
  });
  $("#copy-name").addEventListener("click", async () => {
    await navigator.clipboard.writeText($("#detail-name").textContent);
    showToast("Tensor name copied");
  });
  $$('nav a').forEach((link) => link.addEventListener("click", () => {
    $$('nav a').forEach((item) => item.classList.toggle("active", item === link));
  }));
}

async function init() {
  bindEvents();
  try {
    await Promise.all([loadInfo(), loadModules()]);
    await loadTensors();
  } catch (error) {
    $("#model-name").textContent = "Unable to load model";
    $("#model-path").textContent = error.message;
    showToast(error.message);
  }
}

init();
