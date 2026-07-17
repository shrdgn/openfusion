import { useEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";
import {
  ArrowUp,
  Check,
  ChevronDown,
  Copy,
  Github,
  GitBranch,
  FileText,
  Image,
  KeyRound,
  Loader2,
  MessageSquare,
  Paperclip,
  Plus,
  Settings,
  Sparkles,
  X,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { Markdown } from "@/components/markdown";
import {
  type ActiveConfig,
  type ChatPayload,
  type ChatMessage,
  type ContentBlock,
  errorMessage,
  type Estimate,
  getConfig,
  getEstimate,
  type PanelAnswer,
  type ProgressEvent,
  setApiKey,
  streamFusion,
  type UsagePayload,
} from "@/lib/api";

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

type Preset = "quality" | "budget" | "custom";

interface ConversationTurn {
  id: string;
  prompt: string;
  answer: string;
  panelAnswers: PanelAnswer[];
  usage: UsagePayload | null;
  analysis: Record<string, unknown> | null;
  timestamp: number;
}

interface Conversation {
  id: string;
  title: string;
  turns: ConversationTurn[];
  createdAt: number;
}

function newId() {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

function newConversation(): Conversation {
  return { id: newId(), title: "New conversation", turns: [], createdAt: Date.now() };
}

const STORAGE_KEY = "openfusion_conversations_v1";
const ACTIVE_KEY = "openfusion_active_conversation_v1";

function loadConversations(): Conversation[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveConversations(convs: Conversation[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(convs));
  } catch {
    // quota exceeded — drop oldest until it fits
    const trimmed = convs.slice(-20);
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed)); } catch { /* ignore */ }
  }
}

function titleFromPrompt(prompt: string): string {
  const first = prompt.trim().replace(/\s+/g, " ").slice(0, 60);
  return first.length < prompt.trim().length ? first + "…" : first;
}

// ---------------------------------------------------------------------------
// Attached files
// ---------------------------------------------------------------------------

interface AttachedFile {
  id: string;
  name: string;
  content: string;
  isImage: boolean;
}

const IMAGE_TYPES = ["image/png", "image/jpeg", "image/gif", "image/webp"];
const MAX_FILE_BYTES = 10 * 1024 * 1024;

function readFileAsText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result as string);
    r.onerror = reject;
    r.readAsText(file);
  });
}

function readFileAsDataURL(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result as string);
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}

// ---------------------------------------------------------------------------
// Progress state
// ---------------------------------------------------------------------------

interface ProgressState {
  stage: "panel" | "synthesis";
  models: string[];
  judge: string | null;
  total: number;
  completed: number;
  failed: number;
  streaming: boolean;
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

export default function App() {
  const [config, setConfig] = useState<ActiveConfig | null>(null);
  const [preset, setPreset] = useState<Preset>("quality");
  const [panel, setPanel] = useState<string[]>([]);
  const [judge, setJudge] = useState("");
  const [webSearch, setWebSearch] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [token, setToken] = useState("");

  const [needsKey, setNeedsKey] = useState(false);
  const [keySet, setKeySet] = useState(false);
  const [keyInput, setKeyInput] = useState("");
  const [keySaving, setKeySaving] = useState(false);
  const [keyError, setKeyError] = useState("");
  const [settingsOpen, setSettingsOpen] = useState(false);

  const [maxTokens, setMaxTokens] = useState(1024);

  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [progress, setProgress] = useState<ProgressState | null>(null);
  const [panelAnswers, setPanelAnswers] = useState<PanelAnswer[]>([]);
  const [streamingAnswer, setStreamingAnswer] = useState("");
  const [analysis, setAnalysis] = useState<Record<string, unknown> | null>(null);
  const [usage, setUsage] = useState<UsagePayload | null>(null);
  const answerRef = useRef("");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [attachedFiles, setAttachedFiles] = useState<AttachedFile[]>([]);

  // Conversations
  const [conversations, setConversations] = useState<Conversation[]>(() => loadConversations());
  const [activeId, setActiveId] = useState<string>(() => {
    const saved = localStorage.getItem(ACTIVE_KEY);
    if (saved) {
      const convs = loadConversations();
      if (convs.find((c) => c.id === saved)) return saved;
    }
    return "";
  });
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const activeConversation = conversations.find((c) => c.id === activeId) ?? null;
  const turns = activeConversation?.turns ?? [];

  const bottomRef = useRef<HTMLDivElement>(null);

  const allowOverrides = config?.allow_request_overrides ?? false;

  const modelSuggestions = (() => {
    const set = new Set<string>();
    for (const p of Object.values(config?.presets ?? {})) {
      (p.panel || []).forEach((m) => set.add(m));
      if (p.judge) set.add(p.judge);
    }
    [...panel, judge].forEach((m) => m && set.add(m));
    return [...set].sort();
  })();

  const [estimate, setEstimate] = useState<Estimate | null>(null);

  useEffect(() => {
    getConfig()
      .then((cfg) => {
        setConfig(cfg);
        setPanel(cfg.panel);
        setJudge(cfg.judge || "");
        setWebSearch(cfg.tools?.web_search ?? false);
        setPreset((cfg.preset as Preset) || "custom");
        setNeedsKey(cfg.needs_api_key && cfg.allow_ui_api_key);
        setKeySet(cfg.api_key_set || !cfg.needs_api_key);
      })
      .catch((err) => setStatus(err.message));
  }, []);

  useEffect(() => {
    saveConversations(conversations);
  }, [conversations]);

  useEffect(() => {
    if (activeId) localStorage.setItem(ACTIVE_KEY, activeId);
  }, [activeId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns.length, streamingAnswer]);

  // Debounced pre-run cost estimate.
  useEffect(() => {
    if (!prompt.trim() || !config) {
      setEstimate(null);
      return;
    }
    const messages = buildMessages(turns, prompt, attachedFiles);
    const payload: ChatPayload = { messages };
    if (allowOverrides) {
      payload.openfusion = {
        panel: panel.filter(Boolean),
        judge,
        tools: { web_search: webSearch },
        max_tokens: maxTokens,
      };
    }
    const id = setTimeout(() => {
      getEstimate(payload)
        .then(setEstimate)
        .catch(() => setEstimate(null));
    }, 500);
    return () => clearTimeout(id);
  }, [prompt, panel, judge, webSearch, maxTokens, allowOverrides, config, turns]);

  function applyPreset(name: Preset) {
    setPreset(name);
    if (name !== "custom" && config?.presets?.[name]) {
      setPanel(config.presets[name].panel.slice());
      setJudge(config.presets[name].judge);
      setWebSearch(true);
    }
  }

  async function saveKey() {
    setKeySaving(true);
    setKeyError("");
    try {
      const res = await setApiKey(keyInput.trim());
      setNeedsKey(false);
      setKeySet(res.api_key_set);
      setKeyInput("");
    } catch (err) {
      setKeyError(errorMessage(err));
    } finally {
      setKeySaving(false);
    }
  }

  function ensureActiveConversation(): string {
    if (activeId && conversations.find((c) => c.id === activeId)) return activeId;
    const conv = newConversation();
    setConversations((prev) => [conv, ...prev]);
    setActiveId(conv.id);
    return conv.id;
  }

  async function run() {
    if (busy || !prompt.trim()) return;

    const convId = ensureActiveConversation();
    setBusy(true);
    setStatus("Sending…");
    setProgress(null);
    setStreamingAnswer("");
    setPanelAnswers([]);
    answerRef.current = "";
    setAnalysis(null);
    setUsage(null);

    const currentTurns = conversations.find((c) => c.id === convId)?.turns ?? [];
    const messages = buildMessages(currentTurns, prompt, attachedFiles);
    setAttachedFiles([]);

    const payload: ChatPayload = {
      model: config?.fusion_model || "openfusion",
      messages,
      stream: true,
    };
    if (allowOverrides) {
      payload.openfusion = {
        panel: panel.filter(Boolean),
        judge,
        tools: { web_search: webSearch },
        max_tokens: maxTokens,
        expose_panel: true,
      };
    }

    const submittedPrompt = prompt;
    const submittedFiles = attachedFiles;
    setPrompt("");

    let finalAnalysis: Record<string, unknown> | null = null;
    let finalUsage: UsagePayload | null = null;
    const capturedAnswers: PanelAnswer[] = [];

    await streamFusion(payload, token.trim() || undefined, {
      onProgress: (e: ProgressEvent) => handleProgress(e),
      onPanelAnswer: (a: PanelAnswer) => {
        capturedAnswers.push(a);
        flushSync(() => setPanelAnswers((prev) => [...prev.filter((p) => p.model !== a.model), a]));
      },
      onContent: (text) => {
        answerRef.current += text;
        flushSync(() => {
          setStreamingAnswer(answerRef.current);
          setProgress((p) => (p ? { ...p, streaming: true } : p));
        });
      },
      onAnalysis: (a) => {
        finalAnalysis = a;
        setAnalysis(a);
      },
      onUsage: (u) => {
        finalUsage = u;
        setUsage(u);
      },
      onError: (msg) => { setStatus("Error: " + msg); setBusy(false); setPrompt(submittedPrompt); setAttachedFiles(submittedFiles); },
    });

    const finalAnswer = answerRef.current;

    // Persist the turn
    const turn: ConversationTurn = {
      id: newId(),
      prompt: submittedPrompt,
      answer: finalAnswer,
      panelAnswers: capturedAnswers,
      usage: finalUsage,
      analysis: finalAnalysis,
      timestamp: Date.now(),
    };

    setConversations((prev) =>
      prev.map((c) => {
        if (c.id !== convId) return c;
        const newTurns = [...c.turns, turn];
        const title = c.turns.length === 0 ? titleFromPrompt(submittedPrompt) : c.title;
        return { ...c, title, turns: newTurns };
      }),
    );

    setBusy(false);
    setProgress((p) => (p ? { ...p, streaming: true } : p));
    setStatus((s) => (s.startsWith("Error") ? s : "Done."));
  }

  function handleProgress(e: ProgressEvent) {
    flushSync(() => {
      if (e.stage === "panel") {
        setProgress({
          stage: "panel",
          models: e.models || [],
          judge: e.judge ?? null,
          total: e.total ?? (e.models ? e.models.length : 0),
          completed: 0,
          failed: 0,
          streaming: false,
        });
      } else if (e.stage === "panel_member") {
        setProgress((p) => {
          if (!p) return p;
          const total = e.total != null && e.total > p.total ? e.total : p.total;
          return { ...p, completed: e.completed ?? p.completed, total, failed: p.failed + (e.ok ? 0 : 1) };
        });
      } else if (e.stage === "synthesis" || e.stage === "vote" || e.stage === "ranked") {
        setProgress((p) => (p ? { ...p, stage: "synthesis", judge: e.judge ?? p.judge } : p));
      }
      setStatus(e.message || "");
    });
  }

  async function handleFilesPicked(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? []);
    e.target.value = "";
    const loaded: AttachedFile[] = [];
    for (const file of files) {
      if (file.size > MAX_FILE_BYTES) {
        setStatus(`File "${file.name}" is too large (max 10 MB)`);
        continue;
      }
      const isImage = IMAGE_TYPES.includes(file.type);
      const content = isImage ? await readFileAsDataURL(file) : await readFileAsText(file);
      loaded.push({ id: newId(), name: file.name, content, isImage });
    }
    setAttachedFiles((prev) => [...prev, ...loaded]);
  }

  function resetTurnState() {
    setStreamingAnswer("");
    setPanelAnswers([]);
    setProgress(null);
    setStatus("");
    setAnalysis(null);
    setUsage(null);
    answerRef.current = "";
  }

  function startNewConversation() {
    const conv = newConversation();
    setConversations((prev) => [conv, ...prev]);
    setActiveId(conv.id);
    resetTurnState();
  }

  function switchConversation(id: string) {
    setActiveId(id);
    resetTurnState();
    setBusy(false);
  }

  function deleteConversation(id: string) {
    setConversations((prev) => prev.filter((c) => c.id !== id));
    if (activeId === id) {
      const remaining = conversations.filter((c) => c.id !== id);
      if (remaining.length > 0) switchConversation(remaining[0].id);
      else setActiveId("");
    }
  }

  // Branch: create a new conversation with turns up to (and including) turnIndex
  function branchFrom(convId: string, turnIndex: number) {
    const source = conversations.find((c) => c.id === convId);
    if (!source) return;
    const branchedTurns = source.turns.slice(0, turnIndex + 1);
    const branchTitle = "Branch of: " + source.title;
    const newConv: Conversation = {
      id: newId(),
      title: branchTitle,
      turns: branchedTurns,
      createdAt: Date.now(),
    };
    setConversations((prev) => [newConv, ...prev]);
    switchConversation(newConv.id);
  }

  const isStreaming = busy && streamingAnswer.length > 0;
  const hasAnyResult = busy || streamingAnswer || (turns.length > 0);

  return (
    <div className="flex min-h-screen flex-col">
      <header className="flex items-center gap-3 border-b bg-card px-5 py-3">
        <a href="/" className="flex items-center gap-2 font-semibold">
          <span className="grid h-7 w-7 place-items-center rounded-lg bg-primary text-xs text-primary-foreground">
            of
          </span>
          openfusion
        </a>
        <Badge>playground</Badge>
        <nav className="ml-auto flex items-center gap-1">
          <button
            onClick={() => setSidebarOpen((o) => !o)}
            className="flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-sm text-muted-foreground hover:bg-accent hover:text-foreground"
            title="Toggle history"
          >
            <MessageSquare className="h-4 w-4" />
          </button>
          <button
            onClick={() => setSettingsOpen(true)}
            className="flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-sm text-muted-foreground hover:bg-accent hover:text-foreground"
          >
            <Settings className="h-4 w-4" /> Settings
          </button>
          <a
            href="https://github.com/shahar-dagan/openfusion"
            className="rounded-md p-1.5 text-muted-foreground hover:bg-accent hover:text-foreground"
            aria-label="GitHub"
          >
            <Github className="h-5 w-5" />
          </a>
        </nav>
      </header>

      <SettingsDialog
        open={settingsOpen}
        onOpenChange={setSettingsOpen}
        config={config}
        keySet={keySet}
        token={token}
        onToken={setToken}
        keyInput={keyInput}
        onKeyInput={setKeyInput}
        onSaveKey={saveKey}
        keySaving={keySaving}
        keyError={keyError}
        maxTokens={maxTokens}
        onMaxTokens={setMaxTokens}
      />

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        {sidebarOpen && (
          <aside className="flex w-64 flex-col border-r bg-card">
            <div className="flex items-center justify-between border-b px-3 py-2">
              <span className="text-sm font-medium text-muted-foreground">History</span>
              <button
                onClick={startNewConversation}
                className="flex items-center gap-1 rounded-md px-2 py-1 text-xs text-muted-foreground hover:bg-accent hover:text-foreground"
                title="New conversation"
              >
                <Plus className="h-3.5 w-3.5" /> New
              </button>
            </div>
            <nav className="flex-1 overflow-y-auto p-2">
              {conversations.length === 0 && (
                <p className="px-2 py-4 text-center text-xs text-muted-foreground">
                  No conversations yet
                </p>
              )}
              {conversations.map((conv) => (
                <div
                  key={conv.id}
                  className={
                    "group flex items-start justify-between rounded-md px-2 py-2 " +
                    (conv.id === activeId
                      ? "bg-accent text-accent-foreground"
                      : "text-muted-foreground hover:bg-accent/50 hover:text-foreground")
                  }
                >
                  <button
                    className="flex-1 text-left"
                    onClick={() => switchConversation(conv.id)}
                  >
                    <div className="truncate text-sm font-medium leading-tight">{conv.title}</div>
                    <div className="mt-0.5 text-xs opacity-60">
                      {conv.turns.length} turn{conv.turns.length !== 1 ? "s" : ""} ·{" "}
                      {new Date(conv.createdAt).toLocaleDateString()}
                    </div>
                  </button>
                  <button
                    onClick={() => deleteConversation(conv.id)}
                    className="ml-1 mt-0.5 shrink-0 rounded p-0.5 opacity-0 group-hover:opacity-100 hover:text-destructive"
                    title="Delete"
                  >
                    <X className="h-3 w-3" />
                  </button>
                </div>
              ))}
            </nav>
          </aside>
        )}

        {/* Main area */}
        <main className="flex flex-1 flex-col overflow-hidden">
          <div className="flex-1 overflow-y-auto px-4 pb-4 pt-6">
            <div className="mx-auto w-full max-w-3xl">
              {/* Empty state */}
              {turns.length === 0 && !busy && (
                <div className="mb-8 text-center">
                  <h1 className="flex items-center justify-center gap-2 text-4xl font-bold tracking-tight">
                    <Sparkles className="h-7 w-7 text-primary" /> Model Fusion
                  </h1>
                  <p className="mt-2 text-muted-foreground">
                    Run a panel of models, analyze them, and fuse into one stronger answer.
                  </p>
                </div>
              )}

              {needsKey && (
                <Card className="mb-6 border-primary/30">
                  <CardContent className="flex flex-col gap-3">
                    <div className="flex items-center gap-2 font-medium">
                      <KeyRound className="h-4 w-4 text-primary" /> Add your OpenRouter API key to start
                    </div>
                    <p className="text-sm text-muted-foreground">
                      The key is held only in this server's memory and used to call models on your
                      behalf — it never leaves your machine.
                    </p>
                    <div className="flex gap-2">
                      <Input
                        type="password"
                        placeholder="sk-or-v1-…"
                        value={keyInput}
                        onChange={(e) => setKeyInput(e.target.value)}
                        onKeyDown={(e) => e.key === "Enter" && saveKey()}
                      />
                      <Button onClick={saveKey} disabled={keySaving || !keyInput.trim()}>
                        {keySaving ? <Loader2 className="h-4 w-4 animate-spin" /> : "Save"}
                      </Button>
                    </div>
                    {keyError && <p className="text-sm text-destructive">{keyError}</p>}
                  </CardContent>
                </Card>
              )}

              {/* Past turns */}
              {turns.map((turn, i) => (
                <TurnView
                  key={turn.id}
                  turn={turn}
                  onBranch={() => branchFrom(activeConversation!.id, i)}
                />
              ))}

              {/* In-flight turn */}
              {busy && (
                <div className="mb-6 flex flex-col gap-4">
                  {progress ? (
                    <ProgressPanel progress={progress} />
                  ) : (
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      {status}
                    </div>
                  )}
                  {panelAnswers.length > 0 && <PanelGrid answers={panelAnswers} />}
                  {(streamingAnswer || isStreaming) && (
                    <Card>
                      <CardContent>
                        <div className="mb-2 text-xs font-medium uppercase tracking-wide text-primary">
                          Fused answer
                        </div>
                        <div className="relative">
                          {streamingAnswer && <CopyButton text={streamingAnswer} />}
                          {streamingAnswer ? (
                            <Markdown>{streamingAnswer}</Markdown>
                          ) : (
                            <div className="flex items-center gap-2 text-muted-foreground">
                              <Loader2 className="h-4 w-4 animate-spin" /> Waiting for the panel…
                            </div>
                          )}
                        </div>
                      </CardContent>
                    </Card>
                  )}
                </div>
              )}

              {!busy && status.startsWith("Error") && (
                <p className="mb-4 text-sm text-destructive">{status}</p>
              )}

              <div ref={bottomRef} />
            </div>
          </div>

          {/* Input bar */}
          <div className="border-t bg-card px-4 py-4">
            <div className="mx-auto w-full max-w-3xl">
              <Card>
                <CardContent className="flex flex-col gap-4">
                  <Tabs value={preset} onValueChange={(v) => applyPreset(v as Preset)}>
                    <TabsList>
                      <TabsTrigger value="quality">Quality</TabsTrigger>
                      <TabsTrigger value="budget">Budget</TabsTrigger>
                      <TabsTrigger value="custom">Custom</TabsTrigger>
                    </TabsList>
                  </Tabs>

                  <div className="flex flex-wrap items-center gap-2">
                    {panel.map((model, i) => (
                      <ModelChip
                        key={i}
                        value={model}
                        editable={allowOverrides}
                        options={modelSuggestions}
                        onChange={(v) => setPanel((p) => p.map((m, j) => (j === i ? v : m)))}
                        onRemove={() => {
                          setPanel((p) => p.filter((_, j) => j !== i));
                          setPreset("custom");
                        }}
                      />
                    ))}
                    {allowOverrides && (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => {
                          setPanel((p) => [...p, ""]);
                          setPreset("custom");
                        }}
                      >
                        <Plus className="h-3.5 w-3.5" /> Add model
                      </Button>
                    )}
                  </div>

                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-sm text-muted-foreground">Fuse with</span>
                    <ModelChip
                      value={judge}
                      editable={allowOverrides}
                      options={modelSuggestions}
                      onChange={setJudge}
                    />
                  </div>

                  <Textarea
                    className="min-h-[80px] border-0 px-1 text-base shadow-none focus-visible:ring-0"
                    placeholder={turns.length > 0 ? "Continue the conversation…" : "Ask anything…"}
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    onKeyDown={(e) => {
                      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") run();
                    }}
                  />

                  {attachedFiles.length > 0 && (
                    <div className="flex flex-wrap gap-2">
                      {attachedFiles.map((f) => (
                        <div
                          key={f.id}
                          className="flex items-center gap-1.5 rounded-full border bg-accent px-2.5 py-1 text-xs"
                        >
                          {f.isImage ? (
                            <Image className="h-3 w-3 shrink-0 text-muted-foreground" />
                          ) : (
                            <FileText className="h-3 w-3 shrink-0 text-muted-foreground" />
                          )}
                          <span className="max-w-[160px] truncate">{f.name}</span>
                          <button
                            onClick={() =>
                              setAttachedFiles((prev) => prev.filter((x) => x.id !== f.id))
                            }
                            className="ml-0.5 text-muted-foreground hover:text-foreground"
                            aria-label={`Remove ${f.name}`}
                          >
                            <X className="h-3 w-3" />
                          </button>
                        </div>
                      ))}
                    </div>
                  )}

                  <input
                    ref={fileInputRef}
                    type="file"
                    multiple
                    accept="text/*,image/png,image/jpeg,image/gif,image/webp,.md,.json,.csv,.yaml,.yml,.toml,.ts,.tsx,.js,.jsx,.py,.rs,.go,.java,.c,.cpp,.sh"
                    className="hidden"
                    onChange={handleFilesPicked}
                  />

                  <div className="flex items-center gap-4 border-t pt-3">
                    <label className="flex items-center gap-2 text-sm text-muted-foreground">
                      <Switch checked={webSearch} onCheckedChange={setWebSearch} />
                      Web search
                    </label>
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="rounded-md p-1.5 text-muted-foreground hover:bg-accent hover:text-foreground"
                      title="Attach files"
                      aria-label="Attach files"
                    >
                      <Paperclip className="h-4 w-4" />
                    </button>
                    <div className="flex-1" />
                    {estimate && (
                      <span className="text-xs text-muted-foreground" title="Estimated pre-run cost">
                        ≈ {estimate.calls} call{estimate.calls === 1 ? "" : "s"}
                        {estimate.cost_usd != null
                          ? ` · ~$${estimate.cost_usd.toFixed(estimate.cost_usd < 0.01 ? 4 : 2)}`
                          : ` · ~${estimate.input_tokens} in-tok`}
                      </span>
                    )}
                    <Button onClick={run} disabled={busy || !prompt.trim()}>
                      {busy ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <ArrowUp className="h-4 w-4" />
                      )}
                      Fuse
                    </Button>
                  </div>

                  {!allowOverrides && config && (
                    <p className="text-xs text-muted-foreground">
                      This server uses a fixed config. Set{" "}
                      <code>allow_request_overrides: true</code> to edit the panel from here.
                    </p>
                  )}
                </CardContent>
              </Card>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Build messages array for the API (include prior turns as context)
// ---------------------------------------------------------------------------

function buildMessages(
  turns: ConversationTurn[],
  currentPrompt: string,
  files: AttachedFile[] = [],
): ChatMessage[] {
  const messages: ChatMessage[] = [];
  for (const turn of turns) {
    messages.push({ role: "user", content: turn.prompt });
    if (turn.answer) messages.push({ role: "assistant", content: turn.answer });
  }
  if (files.length === 0) {
    messages.push({ role: "user", content: currentPrompt });
  } else {
    const blocks: ContentBlock[] = [{ type: "text", text: currentPrompt }];
    for (const f of files) {
      if (f.isImage) {
        blocks.push({ type: "image_url", image_url: { url: f.content } });
      } else {
        blocks.push({ type: "text", text: `\n\nFile: ${f.name}\n\`\`\`\n${f.content}\n\`\`\`` });
      }
    }
    messages.push({ role: "user", content: blocks });
  }
  return messages;
}

// ---------------------------------------------------------------------------
// TurnView — a completed turn in history
// ---------------------------------------------------------------------------

function TurnView({ turn, onBranch }: { turn: ConversationTurn; onBranch: () => void }) {
  const [showPanel, setShowPanel] = useState(false);
  return (
    <div className="mb-8 flex flex-col gap-3">
      {/* User prompt */}
      <div className="flex items-start justify-end gap-2">
        <div className="group relative max-w-xl rounded-2xl bg-primary px-4 py-3 text-primary-foreground">
          <p className="whitespace-pre-wrap text-sm">{turn.prompt}</p>
          <button
            onClick={onBranch}
            className="absolute -bottom-5 right-0 flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] text-muted-foreground opacity-0 transition-opacity hover:bg-accent hover:text-foreground group-hover:opacity-100"
            title="Branch from here"
          >
            <GitBranch className="h-3 w-3" /> Branch
          </button>
        </div>
      </div>

      {/* Fused answer */}
      {turn.answer && (
        <Card>
          <CardContent>
            <div className="mb-2 text-xs font-medium uppercase tracking-wide text-primary">
              Fused answer
            </div>
            <div className="relative">
              <CopyButton text={turn.answer} />
              <Markdown>{turn.answer}</Markdown>
            </div>

            <div className="mt-3 flex items-center gap-3 border-t pt-2">
              {turn.panelAnswers.length > 0 && (
                <button
                  onClick={() => setShowPanel((o) => !o)}
                  className="text-xs text-muted-foreground hover:text-foreground"
                >
                  {showPanel ? "Hide" : "Show"} {turn.panelAnswers.length} panel answers
                </button>
              )}
              <div className="flex-1" />
              <span className="text-xs text-muted-foreground">
                {new Date(turn.timestamp).toLocaleTimeString()}
              </span>
            </div>
            {showPanel && turn.panelAnswers.length > 0 && (
              <div className="mt-3">
                <PanelGrid answers={turn.panelAnswers} />
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Panel grid shown directly when there is no fused answer (e.g. panel-only run) */}
      {!turn.answer && turn.panelAnswers.length > 0 && (
        <PanelGrid answers={turn.panelAnswers} />
      )}

      {/* Usage and analysis always shown when present */}
      {turn.usage && <UsageBar usage={turn.usage} />}
      {turn.analysis && <AnalysisCard analysis={turn.analysis} />}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Existing components (unchanged)
// ---------------------------------------------------------------------------

function SettingsDialog({
  open,
  onOpenChange,
  config,
  keySet,
  token,
  onToken,
  keyInput,
  onKeyInput,
  onSaveKey,
  keySaving,
  keyError,
  maxTokens,
  onMaxTokens,
}: {
  open: boolean;
  onOpenChange: (v: boolean) => void;
  config: ActiveConfig | null;
  keySet: boolean;
  token: string;
  onToken: (v: string) => void;
  keyInput: string;
  onKeyInput: (v: string) => void;
  onSaveKey: () => void;
  keySaving: boolean;
  keyError: string;
  maxTokens: number;
  onMaxTokens: (v: number) => void;
}) {
  const canSetKey = config?.allow_ui_api_key ?? false;
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Settings</DialogTitle>
          <DialogDescription>
            Keys are kept only in this server's memory and never reach the browser of other users.
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-2">
          <Label htmlFor="settings-key">OpenRouter API key</Label>
          {canSetKey ? (
            <>
              <div className="flex gap-2">
                <Input
                  id="settings-key"
                  type="password"
                  placeholder={keySet ? "•••••••• (set) — paste to replace" : "sk-or-v1-…"}
                  value={keyInput}
                  onChange={(e) => onKeyInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && onSaveKey()}
                />
                <Button onClick={onSaveKey} disabled={keySaving || !keyInput.trim()}>
                  {keySaving ? <Loader2 className="h-4 w-4 animate-spin" /> : "Save"}
                </Button>
              </div>
              {keySet && !keyInput && (
                <p className="flex items-center gap-1 text-sm text-muted-foreground">
                  <Check className="h-3.5 w-3.5 text-primary" /> A key is configured.
                </p>
              )}
              {keyError && <p className="text-sm text-destructive">{keyError}</p>}
            </>
          ) : (
            <p className="text-sm text-muted-foreground">
              This server is configured with a fixed key (UI key entry is disabled).
            </p>
          )}
        </div>

        <div className="flex flex-col gap-2">
          <Label htmlFor="settings-tokens">Response length (max tokens per call)</Label>
          <select
            id="settings-tokens"
            value={maxTokens}
            onChange={(e) => onMaxTokens(Number(e.target.value))}
            className="h-10 rounded-md border bg-card px-3 text-sm"
          >
            <option value={512}>Short (~512)</option>
            <option value={1024}>Medium (~1024)</option>
            <option value={2048}>Long (~2048)</option>
            <option value={4096}>Very long (~4096)</option>
          </select>
          <p className="text-xs text-muted-foreground">
            Caps every panel and judge call — lower is faster and cheaper.
          </p>
        </div>

        <div className="flex flex-col gap-2">
          <Label htmlFor="settings-token">Gateway token (optional)</Label>
          <Input
            id="settings-token"
            type="password"
            placeholder="Bearer token if your server requires one"
            value={token}
            onChange={(e) => onToken(e.target.value)}
          />
          <p className="text-xs text-muted-foreground">
            Sent as <code>Authorization: Bearer …</code> to a server with a gateway allowlist.
          </p>
        </div>

        {config && (
          <div className="rounded-md border bg-muted/40 p-3 text-xs text-muted-foreground">
            Active server: {config.panel.length} panel models · judge {config.judge || "—"} ·
            overrides {config.allow_request_overrides ? "on" : "off"}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}

function ProgressPanel({ progress }: { progress: ProgressState }) {
  const panelDone = progress.completed >= progress.total && progress.total > 0;
  const Step = ({
    active,
    done,
    children,
  }: {
    active: boolean;
    done: boolean;
    children: React.ReactNode;
  }) => (
    <div className="flex items-center gap-2 text-sm">
      {done ? (
        <Check className="h-4 w-4 text-primary" />
      ) : active ? (
        <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
      ) : (
        <span className="h-4 w-4 rounded-full border" />
      )}
      <span className={done || active ? "text-foreground" : "text-muted-foreground"}>
        {children}
      </span>
    </div>
  );
  return (
    <Card>
      <CardContent className="flex flex-col gap-3 py-4">
        <Step active={progress.stage === "panel"} done={panelDone}>
          Querying panel — {progress.completed}/{progress.total} answered
          {progress.failed > 0 ? ` · ${progress.failed} failed` : ""}
        </Step>
        {progress.models.length > 0 && (
          <div className="flex flex-wrap gap-1.5 pl-6">
            {progress.models.map((m, i) => (
              <span
                key={i}
                className={
                  "rounded-md border px-2 py-0.5 text-xs " +
                  (i < progress.completed
                    ? "bg-accent text-accent-foreground"
                    : "text-muted-foreground")
                }
              >
                {m}
              </span>
            ))}
          </div>
        )}
        <Step
          active={progress.stage === "synthesis" && !progress.streaming}
          done={progress.streaming}
        >
          Synthesizing{progress.judge ? ` with ${progress.judge}` : ""}
        </Step>
      </CardContent>
    </Card>
  );
}

function PanelGrid({ answers }: { answers: PanelAnswer[] }) {
  return (
    <div>
      <div className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
        Panel · {answers.length} model{answers.length === 1 ? "" : "s"} answered
      </div>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {answers.map((a) => (
          <Card key={a.model} className="overflow-hidden">
            <CardContent className="p-3">
              <div
                className="mb-2 truncate text-xs font-medium text-muted-foreground"
                title={a.model}
              >
                {a.model}
              </div>
              <div className="max-h-64 overflow-auto text-sm">
                <Markdown>{a.content}</Markdown>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      onClick={() => {
        navigator.clipboard?.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      }}
      className="absolute right-0 top-0 flex items-center gap-1 rounded-md border bg-card px-2 py-1 text-xs text-muted-foreground hover:text-foreground"
      aria-label="Copy answer"
    >
      {copied ? <Check className="h-3.5 w-3.5 text-primary" /> : <Copy className="h-3.5 w-3.5" />}
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

function ModelChip({
  value,
  editable,
  onChange,
  onRemove,
  options = [],
}: {
  value: string;
  editable: boolean;
  onChange: (v: string) => void;
  onRemove?: () => void;
  options?: string[];
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);

  if (!editable) {
    return (
      <span className="inline-flex items-center rounded-lg border bg-card px-2.5 py-1.5 text-sm">
        {value || "—"}
      </span>
    );
  }

  const q = value.toLowerCase();
  const matches = options.filter((o) => o !== value && o.toLowerCase().includes(q));

  return (
    <span
      ref={ref}
      className="relative inline-flex items-center gap-1 rounded-lg border bg-card py-1.5 pl-2.5 pr-1.5 text-sm"
    >
      <input
        className="min-w-[150px] bg-transparent outline-none"
        value={value}
        placeholder="provider/model"
        onChange={(e) => {
          onChange(e.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
      />
      {options.length > 0 && (
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          className="text-muted-foreground hover:text-foreground"
          aria-label="Choose a model"
        >
          <ChevronDown className="h-3.5 w-3.5" />
        </button>
      )}
      {onRemove && (
        <button
          onClick={onRemove}
          aria-label="Remove model"
          className="text-muted-foreground hover:text-destructive"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      )}
      {open && matches.length > 0 && (
        <div className="absolute left-0 top-full z-20 mt-1 max-h-60 w-72 overflow-auto rounded-md border bg-card p-1 shadow-md">
          {matches.map((o) => (
            <button
              key={o}
              type="button"
              onClick={() => {
                onChange(o);
                setOpen(false);
              }}
              className="block w-full truncate rounded px-2 py-1.5 text-left hover:bg-accent"
            >
              {o}
            </button>
          ))}
        </div>
      )}
    </span>
  );
}

function AnalysisCard({ analysis }: { analysis: Record<string, unknown> }) {
  const [open, setOpen] = useState(true);
  const entries: [string, unknown][] =
    "raw" in analysis ? [["analysis", analysis.raw]] : Object.entries(analysis);
  return (
    <Card>
      <CardContent>
        <button
          className="flex w-full items-center justify-between font-semibold"
          onClick={() => setOpen((o) => !o)}
        >
          Analysis — consensus, contradictions, blind spots
          <span className="text-muted-foreground">{open ? "–" : "+"}</span>
        </button>
        {open && (
          <div className="mt-3 flex flex-col gap-3 text-sm">
            {entries.map(([key, value]) => (
              <div key={key}>
                <div className="font-medium capitalize">{key.replace(/_/g, " ")}</div>
                {Array.isArray(value) ? (
                  <ul className="mt-1 list-disc pl-5 text-muted-foreground">
                    {value.map((item, i) => (
                      <li key={i}>{typeof item === "string" ? item : JSON.stringify(item)}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="mt-1 whitespace-pre-wrap text-muted-foreground">{String(value)}</p>
                )}
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function UsageBar({ usage }: { usage: UsagePayload }) {
  const total = usage.total || usage.panel_total || usage;
  const parts: string[] = [];
  if (total?.total_tokens != null) parts.push(`${total.total_tokens} tokens`);
  if (total?.cost != null) parts.push(`$${Number(total.cost).toFixed(4)}`);
  if (Array.isArray(usage.panel)) parts.push(`${usage.panel.length} panel members + judge`);
  if (!parts.length) return null;
  return (
    <div className="flex flex-wrap gap-4 text-xs text-muted-foreground">
      {parts.map((p, i) => (
        <span key={i}>{p}</span>
      ))}
    </div>
  );
}
