import { useEffect, useRef, useState } from "react";
import { ArrowUp, Github, KeyRound, Loader2, Plus, Sparkles, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { type ActiveConfig, getConfig, setApiKey, streamFusion } from "@/lib/api";

type Preset = "quality" | "budget" | "custom";

export default function App() {
  const [config, setConfig] = useState<ActiveConfig | null>(null);
  const [preset, setPreset] = useState<Preset>("quality");
  const [panel, setPanel] = useState<string[]>([]);
  const [judge, setJudge] = useState("");
  const [webSearch, setWebSearch] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [token, setToken] = useState("");
  const [showSettings, setShowSettings] = useState(false);

  const [needsKey, setNeedsKey] = useState(false);
  const [keyInput, setKeyInput] = useState("");
  const [keySaving, setKeySaving] = useState(false);
  const [keyError, setKeyError] = useState("");

  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [answer, setAnswer] = useState("");
  const [analysis, setAnalysis] = useState<Record<string, unknown> | null>(null);
  const [usage, setUsage] = useState<any>(null);
  const [hasRun, setHasRun] = useState(false);
  const answerRef = useRef("");

  const allowOverrides = config?.allow_request_overrides ?? false;

  useEffect(() => {
    getConfig()
      .then((cfg) => {
        setConfig(cfg);
        setPanel(cfg.panel);
        setJudge(cfg.judge || "");
        setWebSearch(cfg.tools?.web_search ?? false);
        setPreset((cfg.preset as Preset) || "custom");
        setNeedsKey(cfg.needs_api_key && cfg.allow_ui_api_key);
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
      await setApiKey(keyInput.trim());
      setNeedsKey(false);
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
    setAnswer("");
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
      };
    }

    await streamFusion(payload, token.trim() || undefined, {
      onProgress: setStatus,
      onContent: (text) => {
        answerRef.current += text;
        setAnswer(answerRef.current);
      },
      onAnalysis: setAnalysis,
      onUsage: setUsage,
      onError: (msg) => setStatus("Error: " + msg),
    });

    setBusy(false);
    setStatus((s) => (s.startsWith("Error") ? s : "Done."));
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
        <a
          href="https://github.com/shahar-dagan/openfusion"
          className="ml-auto text-muted-foreground hover:text-foreground"
          aria-label="GitHub"
        >
          <Github className="h-5 w-5" />
        </a>
      </header>

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
              <ModelChip value={judge} editable={allowOverrides} onChange={setJudge} />
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
              <button
                className="text-sm text-muted-foreground hover:text-foreground"
                onClick={() => setShowSettings((s) => !s)}
              >
                Settings
              </button>
              <div className="flex-1" />
              <Button onClick={run} disabled={busy || !prompt.trim()}>
                {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <ArrowUp className="h-4 w-4" />}
                Fuse
              </Button>
            </div>

            {showSettings && (
              <div className="flex flex-col gap-1.5 border-t pt-3">
                <Label htmlFor="gw">Gateway token (optional)</Label>
                <Input
                  id="gw"
                  type="password"
                  placeholder="Bearer token if your server requires one"
                  value={token}
                  onChange={(e) => setToken(e.target.value)}
                />
              </div>
            )}

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
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              {busy && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
              {status}
            </div>
            <Card>
              <CardContent>
                <div className="whitespace-pre-wrap leading-relaxed">
                  {answer || (busy ? "" : "—")}
                </div>
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

function ModelChip({
  value,
  editable,
  onChange,
  onRemove,
}: {
  value: string;
  editable: boolean;
  onChange: (v: string) => void;
  onRemove?: () => void;
}) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-lg border bg-card px-2.5 py-1.5 text-sm">
      {editable ? (
        <input
          className="min-w-[120px] bg-transparent outline-none"
          value={value}
          placeholder="provider/model"
          onChange={(e) => onChange(e.target.value)}
        />
      ) : (
        <span>{value || "—"}</span>
      )}
      {editable && onRemove && (
        <button onClick={onRemove} className="text-muted-foreground hover:text-destructive">
          <X className="h-3.5 w-3.5" />
        </button>
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
