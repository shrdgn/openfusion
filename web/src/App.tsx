import { useEffect, useRef, useState } from "react";
import {
  ArrowUp,
  Check,
  ChevronDown,
  Copy,
  Github,
  KeyRound,
  Loader2,
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
  getConfig,
  type PanelAnswer,
  type ProgressEvent,
  setApiKey,
  streamFusion,
} from "@/lib/api";

type Preset = "quality" | "budget" | "custom";

interface ProgressState {
  stage: "panel" | "synthesis";
  models: string[];
  judge: string | null;
  total: number;
  completed: number;
  failed: number;
  streaming: boolean;
}

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
  const [answer, setAnswer] = useState("");
  const [analysis, setAnalysis] = useState<Record<string, unknown> | null>(null);
  const [usage, setUsage] = useState<any>(null);
  const [hasRun, setHasRun] = useState(false);
  const answerRef = useRef("");

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
    } catch (err: any) {
      setKeyError(err.message);
    } finally {
      setKeySaving(false);
    }
  }

  async function run() {
    if (busy || !prompt.trim()) return;
    setBusy(true);
    setHasRun(true);
    setStatus("Sending…");
    setProgress(null);
    setAnswer("");
    setPanelAnswers([]);
    answerRef.current = "";
    setAnalysis(null);
    setUsage(null);

    const payload: any = {
      model: config?.fusion_model || "openfusion",
      messages: [{ role: "user", content: prompt }],
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

    await streamFusion(payload, token.trim() || undefined, {
      onProgress: (e: ProgressEvent) => handleProgress(e),
      onPanelAnswer: (a: PanelAnswer) =>
        setPanelAnswers((prev) => [...prev.filter((p) => p.model !== a.model), a]),
      onContent: (text) => {
        answerRef.current += text;
        setAnswer(answerRef.current);
        setProgress((p) => (p ? { ...p, streaming: true } : p));
      },
      onAnalysis: setAnalysis,
      onUsage: setUsage,
      onError: (msg) => setStatus("Error: " + msg),
    });

    setBusy(false);
    setProgress((p) => (p ? { ...p, streaming: true } : p));
    setStatus((s) => (s.startsWith("Error") ? s : "Done."));
  }

  function handleProgress(e: ProgressEvent) {
    if (e.stage === "panel") {
      setProgress({
        stage: "panel",
        models: e.models || [],
        judge: e.judge ?? null,
        total: e.total || (e.models ? e.models.length : 0),
        completed: 0,
        failed: 0,
        streaming: false,
      });
    } else if (e.stage === "panel_member") {
      setProgress((p) =>
        p
          ? { ...p, completed: e.completed ?? p.completed, failed: p.failed + (e.ok ? 0 : 1) }
          : p,
      );
    } else if (e.stage === "synthesis" || e.stage === "vote" || e.stage === "ranked") {
      setProgress((p) => (p ? { ...p, stage: "synthesis", judge: e.judge ?? p.judge } : p));
    }
    setStatus(e.message || "");
  }

  return (
    <div className="min-h-full">
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

      <main className="mx-auto w-full max-w-3xl px-4 pb-24 pt-12">
        <div className="mb-8 text-center">
          <h1 className="flex items-center justify-center gap-2 text-4xl font-bold tracking-tight">
            <Sparkles className="h-7 w-7 text-primary" /> Model Fusion
          </h1>
          <p className="mt-2 text-muted-foreground">
            Run a panel of models, analyze them, and fuse into one stronger answer.
          </p>
        </div>

        {needsKey && (
          <Card className="mb-6 border-primary/30">
            <CardContent className="flex flex-col gap-3">
              <div className="flex items-center gap-2 font-medium">
                <KeyRound className="h-4 w-4 text-primary" /> Add your OpenRouter API key to start
              </div>
              <p className="text-sm text-muted-foreground">
                The key is held only in this server's memory and used to call models on your behalf —
                it never leaves your machine.
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
              className="min-h-[120px] border-0 px-1 text-base shadow-none focus-visible:ring-0"
              placeholder="Ask anything…"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={(e) => {
                if ((e.metaKey || e.ctrlKey) && e.key === "Enter") run();
              }}
            />

            <div className="flex items-center gap-4 border-t pt-3">
              <label className="flex items-center gap-2 text-sm text-muted-foreground">
                <Switch checked={webSearch} onCheckedChange={setWebSearch} />
                Web search
              </label>
              <div className="flex-1" />
              <Button onClick={run} disabled={busy || !prompt.trim()}>
                {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <ArrowUp className="h-4 w-4" />}
                Fuse
              </Button>
            </div>

            {!allowOverrides && config && (
              <p className="text-xs text-muted-foreground">
                This server uses a fixed config. Set <code>allow_request_overrides: true</code> to
                edit the panel from here.
              </p>
            )}
          </CardContent>
        </Card>

        {hasRun && (
          <div className="mt-6 flex flex-col gap-4">
            {progress ? (
              <ProgressPanel progress={progress} />
            ) : (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                {busy && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
                {status}
              </div>
            )}
            {panelAnswers.length > 0 && <PanelGrid answers={panelAnswers} />}
            <Card>
              <CardContent>
                {answer && (
                  <div className="mb-2 text-xs font-medium uppercase tracking-wide text-primary">
                    Fused answer
                  </div>
                )}
                {answer ? (
                  <div className="relative">
                    <CopyButton text={answer} />
                    <Markdown>{answer}</Markdown>
                  </div>
                ) : (
                  <div className="flex items-center gap-2 text-muted-foreground">
                    {busy ? (
                      <>
                        <Loader2 className="h-4 w-4 animate-spin" /> Waiting for the panel…
                      </>
                    ) : (
                      "—"
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
            {analysis && <AnalysisCard analysis={analysis} />}
            {usage && <UsageBar usage={usage} />}
          </div>
        )}
      </main>
    </div>
  );
}

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
        <Step active={progress.stage === "synthesis" && !progress.streaming} done={progress.streaming}>
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
              <div className="mb-2 truncate text-xs font-medium text-muted-foreground" title={a.model}>
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
        <button onClick={onRemove} className="text-muted-foreground hover:text-destructive">
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
  const entries =
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

function UsageBar({ usage }: { usage: any }) {
  const total = usage.total || usage.panel_total || usage;
  const parts: string[] = [];
  if (total?.total_tokens != null) parts.push(`${total.total_tokens} tokens`);
  if (total?.cost != null) parts.push(`$${Number(total.cost).toFixed(4)}`);
  if (Array.isArray(usage.panel)) parts.push(`${usage.panel.length} panel members + judge`);
  if (!parts.length) return null;
  return (
    <div className="flex flex-wrap gap-4 text-sm text-muted-foreground">
      {parts.map((p, i) => (
        <span key={i}>{p}</span>
      ))}
    </div>
  );
}
