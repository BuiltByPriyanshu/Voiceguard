export async function getConfig() {
  const res = await fetch("/config");
  if (!res.ok) throw new Error(`GET /config failed: ${res.status}`);
  return res.json();
}

export async function setConfig(threshold) {
  const res = await fetch("/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ threshold }),
  });
  if (!res.ok) throw new Error(`POST /config failed: ${res.status}`);
  return res.json();
}

export async function analyzeFile(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/analyze", { method: "POST", body: form });
  if (!res.ok) throw new Error(`POST /analyze failed: ${res.status}`);
  return res.json();
}
