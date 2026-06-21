export interface ActiveConfig {
  preset: string | null;
  strategy: string;
  aggregator: string;
  panel: string[];
  judge: string | null;
  tools: { web_search: boolean; web_fetch: boolean };
  allow_request_overrides: boolean;
  allow_ui_api_key: boolean;
  needs_api_key: boolean;
  api_key_set: boolean;
  presets: Record<string, { panel: string[]; judge: string }>;
  fusion_model: string;
}

const UNREACHABLE =
  "Couldn't reach the openfusion server. Open the page that the running server serves " +
  "(e.g. http://localhost:8000) — not a static file or a different port.";

async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(path, init);
  } catch {
    throw new Error(UNREACHABLE);
  }
}

export async function getConfig(): Promise<ActiveConfig> {
  const res = await apiFetch("/v1/config");
  if (!res.ok) throw new Error("Could not load config (" + res.status + ")");
  return res.json();
}

export async function setApiKey(key: string): Promise<{ api_key_set: boolean }> {
  const res = await apiFetch("/v1/runtime/api-key", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ api_key: key }),
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body?.error?.message || `Failed to set key (${res.status})`);
  return body;
}

export interface ProgressEvent {
  stage: string;
  message?: string;
  models?: string[];
  judge?: string | null;
  total?: number;
  completed?: number;
  ok?: boolean;
  panel_count?: number;
  failed_count?: number;
}

export interface PanelAnswer {
  model: string;
  label: string;
  content: string;
}

export interface StreamHandlers {
  onProgress?: (event: ProgressEvent) => void;
  onPanelAnswer?: (answer: PanelAnswer) => void;
  onContent?: (text: string) => void;
  onAnalysis?: (analysis: Record<string, unknown>) => void;
  onUsage?: (usage: any) => void;
  onError?: (msg: string) => void;
}

function flushBlock(block: string, h: StreamHandlers) {
  let event: string | null = null;
  const dataLines: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (!dataLines.length) return;
  const raw = dataLines.join("\n");
  if (raw === "[DONE]") return;
  let data: any;
  try {
    data = JSON.parse(raw);
  } catch {
    return;
  }
  if (event === "progress") {
    h.onProgress?.(data as ProgressEvent);
  } else if (event === "panel_answer") {
    h.onPanelAnswer?.(data as PanelAnswer);
  } else if (event === "analysis") {
    h.onAnalysis?.(data);
  } else if (event === "usage") {
    h.onUsage?.(data);
  } else if (data.error) {
    h.onError?.(data.error.message || "upstream error");
  } else {
    const delta = data.choices?.[0]?.delta;
    if (delta?.content) h.onContent?.(delta.content);
  }
}

export async function streamFusion(payload: any, token: string | undefined, h: StreamHandlers) {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token) headers.Authorization = "Bearer " + token;
  let res: Response;
  try {
    res = await apiFetch("/v1/chat/completions", {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
    });
  } catch (err) {
    h.onError?.((err as Error).message);
    return;
  }
  if (!res.ok || !res.body) {
    const body = await res.json().catch(() => ({}));
    h.onError?.(body?.error?.message || `Request failed (${res.status})`);
    return;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      flushBlock(buffer.slice(0, idx), h);
      buffer = buffer.slice(idx + 2);
    }
  }
  if (buffer.trim()) flushBlock(buffer, h);
}
