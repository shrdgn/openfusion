import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import type { ActiveConfig, StreamHandlers } from "../lib/api";

const { getConfig, getEstimate, setApiKey, streamFusion } = vi.hoisted(() => ({
  getConfig: vi.fn(),
  getEstimate: vi.fn(),
  setApiKey: vi.fn(),
  streamFusion: vi.fn(),
}));

vi.mock("../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../lib/api")>();
  return { ...actual, getConfig, getEstimate, setApiKey, streamFusion };
});

function baseConfig(overrides: Partial<ActiveConfig> = {}): ActiveConfig {
  return {
    preset: "quality",
    strategy: "panel",
    aggregator: "synthesize",
    panel: ["openai/gpt-4o", "anthropic/claude-3.5-sonnet"],
    judge: "google/gemini-1.5-pro",
    tools: { web_search: false, web_fetch: false },
    allow_request_overrides: true,
    allow_ui_api_key: true,
    needs_api_key: false,
    api_key_set: true,
    presets: {
      quality: { panel: ["openai/gpt-4o"], judge: "openai/gpt-4o" },
      budget: { panel: ["openai/gpt-4o-mini"], judge: "openai/gpt-4o-mini" },
    },
    fusion_model: "openfusion",
    ...overrides,
  };
}

beforeEach(() => {
  localStorage.clear();
  getConfig.mockReset();
  getEstimate.mockReset();
  setApiKey.mockReset();
  streamFusion.mockReset();
  getEstimate.mockResolvedValue({
    calls: 1,
    models: [],
    input_tokens: 10,
    max_output_tokens: 100,
    cost_usd: null,
  });
  streamFusion.mockResolvedValue(undefined);
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("App", () => {
  it("loads the active config and renders the panel/judge models", async () => {
    getConfig.mockResolvedValue(baseConfig());
    render(<App />);

    expect(await screen.findByDisplayValue("openai/gpt-4o")).toBeInTheDocument();
    expect(screen.getByDisplayValue("anthropic/claude-3.5-sonnet")).toBeInTheDocument();
    expect(screen.getByText("Fuse with")).toBeInTheDocument();
  });

  it("doesn't crash when the config fails to load, and stays usable", async () => {
    getConfig.mockRejectedValue(new Error("Could not load config (500)"));
    render(<App />);

    // No config-load banner is rendered pre-run (the error only surfaces via `status`,
    // which is shown once a run starts) -- but the prompt box must still work.
    expect(await screen.findByPlaceholderText("Ask anything…")).toBeInTheDocument();
    expect(screen.queryByDisplayValue(/gpt-4o/)).not.toBeInTheDocument();
  });

  it("shows the API key card when the server needs one, and hides it once saved", async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue(
      baseConfig({ needs_api_key: true, allow_ui_api_key: true, api_key_set: false }),
    );
    setApiKey.mockResolvedValue({ api_key_set: true });

    render(<App />);

    expect(await screen.findByText(/Add your OpenRouter API key to start/)).toBeInTheDocument();

    await user.type(screen.getByPlaceholderText("sk-or-v1-…"), "sk-or-v1-test");
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => expect(setApiKey).toHaveBeenCalledWith("sk-or-v1-test"));
    await waitFor(() =>
      expect(screen.queryByText(/Add your OpenRouter API key to start/)).not.toBeInTheDocument(),
    );
  });

  it("shows the save error when the key fails to save", async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue(
      baseConfig({ needs_api_key: true, allow_ui_api_key: true, api_key_set: false }),
    );
    setApiKey.mockRejectedValue(new Error("Failed to set key (400)"));

    render(<App />);

    await user.type(await screen.findByPlaceholderText("sk-or-v1-…"), "bad-key");
    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(await screen.findByText("Failed to set key (400)")).toBeInTheDocument();
  });

  it("disables the Fuse button until a prompt is entered, then runs the fusion request", async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue(baseConfig());
    streamFusion.mockImplementation(
      async (_payload: unknown, _token: unknown, handlers: StreamHandlers) => {
        handlers.onContent?.("Hello ");
        handlers.onContent?.("world");
      },
    );

    render(<App />);
    await screen.findByDisplayValue("openai/gpt-4o");

    const fuseButton = screen.getByRole("button", { name: /Fuse/ });
    expect(fuseButton).toBeDisabled();

    await user.type(screen.getByPlaceholderText("Ask anything…"), "What is the meaning of life?");
    expect(fuseButton).not.toBeDisabled();

    await user.click(fuseButton);

    expect(await screen.findByText("Hello world")).toBeInTheDocument();
    expect(streamFusion).toHaveBeenCalledTimes(1);
    const [payload] = streamFusion.mock.calls[0];
    expect(payload.model).toBe("openfusion");
    expect(payload.stream).toBe(true);
    expect(payload.messages).toEqual([
      { role: "user", content: "What is the meaning of life?" },
    ]);
  });

  it("surfaces a stream error via onError without leaving the UI stuck busy", async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue(baseConfig());
    streamFusion.mockImplementation(
      async (_payload: unknown, _token: unknown, handlers: StreamHandlers) => {
        handlers.onError?.("upstream exploded");
      },
    );

    render(<App />);
    await screen.findByDisplayValue("openai/gpt-4o");
    await user.type(screen.getByPlaceholderText("Ask anything…"), "hi");
    await user.click(screen.getByRole("button", { name: /Fuse/ }));

    expect(await screen.findByText("Error: upstream exploded")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Fuse/ })).not.toBeDisabled();
  });

  it("attaches a text file and includes it in the payload", async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue(baseConfig());
    streamFusion.mockImplementation((_p: unknown, _t: unknown, h: StreamHandlers) => {
      h.onContent?.("done");
      return Promise.resolve();
    });

    render(<App />);
    await screen.findByDisplayValue("openai/gpt-4o");

    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    const file = new File(["hello file content"], "notes.txt", { type: "text/plain" });
    Object.defineProperty(fileInput, "files", { value: [file], configurable: true });
    fireEvent.change(fileInput);

    expect(await screen.findByText("notes.txt")).toBeInTheDocument();

    await user.type(screen.getByPlaceholderText("Ask anything…"), "summarise this");
    await user.click(screen.getByRole("button", { name: /Fuse/ }));

    await waitFor(() => expect(streamFusion).toHaveBeenCalledTimes(1));
    const [payload] = streamFusion.mock.calls[0];
    const lastMsg = payload.messages[payload.messages.length - 1];
    expect(Array.isArray(lastMsg.content)).toBe(true);
    const blocks = lastMsg.content as Array<{ type: string; text?: string }>;
    expect(blocks[0]).toEqual({ type: "text", text: "summarise this" });
    expect(blocks[1].type).toBe("text");
    expect(blocks[1].text).toContain("hello file content");
  });

  it("removes an attached file when its × button is clicked", async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue(baseConfig());

    render(<App />);
    await screen.findByDisplayValue("openai/gpt-4o");

    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    const file = new File(["data"], "remove-me.txt", { type: "text/plain" });
    Object.defineProperty(fileInput, "files", { value: [file], configurable: true });
    fireEvent.change(fileInput);

    expect(await screen.findByText("remove-me.txt")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Remove remove-me.txt" }));
    expect(screen.queryByText("remove-me.txt")).not.toBeInTheDocument();
  });

  it("opens the settings dialog from the header button", async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue(baseConfig());

    render(<App />);
    await screen.findByDisplayValue("openai/gpt-4o");

    expect(screen.queryByText("Settings")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Settings/ }));

    expect(await screen.findByRole("heading", { name: "Settings" })).toBeInTheDocument();
    expect(
      screen.getByText(/Keys are kept only in this server's memory/),
    ).toBeInTheDocument();
  });

  it("tracks panel progress and renders the fused answer once complete", async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue(baseConfig());
    streamFusion.mockImplementation(
      async (_payload: unknown, _token: unknown, handlers: StreamHandlers) => {
        handlers.onProgress?.({ stage: "panel", models: ["openai/gpt-4o", "anthropic/claude-3.5-sonnet"], total: 2 });
        handlers.onProgress?.({ stage: "panel_member", completed: 1, ok: true });
        handlers.onProgress?.({ stage: "panel_member", completed: 2, ok: false });
        handlers.onProgress?.({ stage: "synthesis", judge: "google/gemini-1.5-pro" });
        handlers.onContent?.("final answer");
      },
    );

    render(<App />);
    await screen.findByDisplayValue("openai/gpt-4o");
    await user.type(screen.getByPlaceholderText("Ask anything…"), "hi");
    await user.click(screen.getByRole("button", { name: /Fuse/ }));

    // After the run completes, the fused answer is stored in the conversation turn.
    expect(await screen.findByText("final answer")).toBeInTheDocument();
  });

  it("renders each panel member's answer in the side-by-side grid", async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue(baseConfig());
    streamFusion.mockImplementation(
      async (_payload: unknown, _token: unknown, handlers: StreamHandlers) => {
        handlers.onPanelAnswer?.({ model: "openai/gpt-4o", label: "A", content: "answer one" });
        handlers.onPanelAnswer?.({
          model: "anthropic/claude-3.5-sonnet",
          label: "B",
          content: "answer two",
        });
      },
    );

    render(<App />);
    await screen.findByDisplayValue("openai/gpt-4o");
    await user.type(screen.getByPlaceholderText("Ask anything…"), "hi");
    await user.click(screen.getByRole("button", { name: /Fuse/ }));

    expect(await screen.findByText("Panel · 2 models answered")).toBeInTheDocument();
    expect(screen.getByText("answer one")).toBeInTheDocument();
    expect(screen.getByText("answer two")).toBeInTheDocument();
  });

  it("copies the fused answer to the clipboard", async () => {
    const user = userEvent.setup();
    const writeText = vi.fn();
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText },
      configurable: true,
    });
    getConfig.mockResolvedValue(baseConfig());
    streamFusion.mockImplementation(
      async (_payload: unknown, _token: unknown, handlers: StreamHandlers) => {
        handlers.onContent?.("copy me");
      },
    );

    render(<App />);
    await screen.findByDisplayValue("openai/gpt-4o");
    await user.type(screen.getByPlaceholderText("Ask anything…"), "hi");
    await user.click(screen.getByRole("button", { name: /Fuse/ }));

    await screen.findByText("copy me");
    await user.click(screen.getByRole("button", { name: "Copy answer" }));

    expect(writeText).toHaveBeenCalledWith("copy me");
    expect(await screen.findByText("Copied")).toBeInTheDocument();
  });

  it("renders analysis entries, including list values, and toggles the card open/closed", async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue(baseConfig());
    streamFusion.mockImplementation(
      async (_payload: unknown, _token: unknown, handlers: StreamHandlers) => {
        handlers.onAnalysis?.({
          consensus: "Everyone agrees",
          blind_spots: ["missed edge case", "no citations"],
        });
      },
    );

    render(<App />);
    await screen.findByDisplayValue("openai/gpt-4o");
    await user.type(screen.getByPlaceholderText("Ask anything…"), "hi");
    await user.click(screen.getByRole("button", { name: /Fuse/ }));

    // Analysis card starts open — content visible immediately.
    expect(await screen.findByText("Everyone agrees")).toBeInTheDocument();
    expect(screen.getByText("missed edge case")).toBeInTheDocument();
    expect(screen.getByText("blind spots")).toBeInTheDocument();

    // Clicking the header collapses the card.
    await user.click(
      screen.getByRole("button", { name: /Analysis — consensus, contradictions, blind spots/ }),
    );
    expect(screen.queryByText("Everyone agrees")).not.toBeInTheDocument();
  });

  it("falls back to a single raw entry when analysis isn't structured", async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue(baseConfig());
    streamFusion.mockImplementation(
      async (_payload: unknown, _token: unknown, handlers: StreamHandlers) => {
        handlers.onAnalysis?.({ raw: "unstructured analysis text" });
      },
    );

    render(<App />);
    await screen.findByDisplayValue("openai/gpt-4o");
    await user.type(screen.getByPlaceholderText("Ask anything…"), "hi");
    await user.click(screen.getByRole("button", { name: /Fuse/ }));

    expect(await screen.findByText("unstructured analysis text")).toBeInTheDocument();
  });

  it("shows token/cost/panel usage totals once the run reports usage", async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue(baseConfig());
    streamFusion.mockImplementation(
      async (_payload: unknown, _token: unknown, handlers: StreamHandlers) => {
        handlers.onUsage?.({
          total: { total_tokens: 500, cost: 0.0123 },
          panel: ["a", "b"],
        });
      },
    );

    render(<App />);
    await screen.findByDisplayValue("openai/gpt-4o");
    await user.type(screen.getByPlaceholderText("Ask anything…"), "hi");
    await user.click(screen.getByRole("button", { name: /Fuse/ }));

    expect(await screen.findByText("500 tokens")).toBeInTheDocument();
    expect(screen.getByText("$0.0123")).toBeInTheDocument();
    expect(screen.getByText("2 panel members + judge")).toBeInTheDocument();
  });

  it("lets the user pick a suggested model, add a chip, and remove one", async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue(baseConfig());

    render(<App />);
    const firstChip = await screen.findByDisplayValue("openai/gpt-4o");
    await user.click(firstChip);

    const suggestion = await screen.findByRole("button", { name: "openai/gpt-4o-mini" });
    await user.click(suggestion);
    expect(screen.getByDisplayValue("openai/gpt-4o-mini")).toBeInTheDocument();

    const chipsBefore = screen.getAllByPlaceholderText("provider/model");
    await user.click(screen.getByRole("button", { name: "Add model" }));
    expect(screen.getAllByPlaceholderText("provider/model")).toHaveLength(chipsBefore.length + 1);

    const removeButtons = screen.getAllByRole("button", { name: "Remove model" });
    await user.click(removeButtons[0]);
    expect(screen.queryByDisplayValue("openai/gpt-4o-mini")).not.toBeInTheDocument();
  });

  it("does not let the panel be edited when the server disallows overrides", async () => {
    getConfig.mockResolvedValue(baseConfig({ allow_request_overrides: false }));
    render(<App />);

    await screen.findByText("openai/gpt-4o");
    expect(screen.queryByRole("button", { name: "Add model" })).not.toBeInTheDocument();
    expect(
      await screen.findByText(/This server uses a fixed config/),
    ).toBeInTheDocument();
  });

  it("switches to the budget preset and back to custom on manual edits", async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue(baseConfig());

    render(<App />);
    await screen.findByDisplayValue("openai/gpt-4o");

    await user.click(screen.getByRole("tab", { name: "Budget" }));
    expect(screen.getAllByDisplayValue("openai/gpt-4o-mini")).toHaveLength(2);
    expect(screen.getByRole("tab", { name: "Budget" })).toHaveAttribute("aria-selected", "true");

    await user.click(screen.getByRole("button", { name: "Add model" }));
    expect(screen.getByRole("tab", { name: "Custom" })).toHaveAttribute("aria-selected", "true");
  });

  it("settings dialog: disables key entry on a fixed-key server and edits max tokens / gateway token", async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue(baseConfig({ allow_ui_api_key: false }));

    render(<App />);
    await screen.findByDisplayValue("openai/gpt-4o");
    await user.click(screen.getByRole("button", { name: /Settings/ }));

    const dialog = await screen.findByRole("dialog");
    expect(
      within(dialog).getByText(/This server is configured with a fixed key/),
    ).toBeInTheDocument();
    expect(within(dialog).queryByPlaceholderText("sk-or-v1-…")).not.toBeInTheDocument();

    await user.selectOptions(within(dialog).getByLabelText(/Response length/), "2048");
    expect(within(dialog).getByLabelText(/Response length/)).toHaveValue("2048");

    await user.type(within(dialog).getByLabelText(/Gateway token/), "secret-token");
    expect(within(dialog).getByLabelText(/Gateway token/)).toHaveValue("secret-token");

    expect(within(dialog).getByText(/Active server: 2 panel models/)).toBeInTheDocument();
  });

  it("starts a new conversation when 'New' is clicked in the sidebar", async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue(baseConfig());
    streamFusion.mockImplementation(
      async (_payload: unknown, _token: unknown, handlers: StreamHandlers) => {
        handlers.onContent?.("first answer");
      },
    );

    render(<App />);
    await screen.findByDisplayValue("openai/gpt-4o");

    await user.type(screen.getByPlaceholderText("Ask anything…"), "first prompt");
    await user.click(screen.getByRole("button", { name: /Fuse/ }));
    await screen.findByText("first answer");

    // Start a new conversation via the sidebar "New" button (title="New conversation").
    await user.click(screen.getByTitle("New conversation"));

    // Main area should show empty state — old answer no longer in the turn list.
    expect(screen.queryByText("first answer")).not.toBeInTheDocument();
    expect(screen.getByPlaceholderText("Ask anything…")).toHaveValue("");
  });

  it("saves completed turns to localStorage and restores them on remount", async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue(baseConfig());
    streamFusion.mockImplementation(
      async (_payload: unknown, _token: unknown, handlers: StreamHandlers) => {
        handlers.onContent?.("saved answer");
      },
    );

    const { unmount } = render(<App />);
    await screen.findByDisplayValue("openai/gpt-4o");
    await user.type(screen.getByPlaceholderText("Ask anything…"), "persisted prompt");
    await user.click(screen.getByRole("button", { name: /Fuse/ }));
    await screen.findByText("saved answer");
    unmount();

    // Remount — TurnView should show the saved turn.
    getConfig.mockResolvedValue(baseConfig());
    render(<App />);
    await screen.findByDisplayValue("openai/gpt-4o");
    // The turn text appears in TurnView (not just the sidebar title).
    expect(await screen.findByText("saved answer")).toBeInTheDocument();
  });

  it("switches between conversations in the sidebar", async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue(baseConfig());
    streamFusion.mockImplementation(
      async (_payload: unknown, _token: unknown, handlers: StreamHandlers) => {
        handlers.onContent?.("answer one");
      },
    );

    render(<App />);
    await screen.findByDisplayValue("openai/gpt-4o");

    await user.type(screen.getByPlaceholderText("Ask anything…"), "question one");
    await user.click(screen.getByRole("button", { name: /Fuse/ }));
    await screen.findByText("answer one");

    // Start a second conversation.
    streamFusion.mockImplementation(
      async (_payload: unknown, _token: unknown, handlers: StreamHandlers) => {
        handlers.onContent?.("answer two");
      },
    );
    await user.click(screen.getByTitle("New conversation"));
    await user.type(screen.getByPlaceholderText("Ask anything…"), "question two");
    await user.click(screen.getByRole("button", { name: /Fuse/ }));
    await screen.findByText("answer two");

    // Switch back to the first conversation by clicking its sidebar entry.
    // The sidebar button text is the conversation title (truncated first prompt).
    const sidebarButtons = screen.getAllByText("question one");
    // Click the one inside the sidebar nav (not the TurnView — TurnView is now hidden).
    await user.click(sidebarButtons[0]);
    expect(await screen.findByText("answer one")).toBeInTheDocument();
    expect(screen.queryByText("answer two")).not.toBeInTheDocument();
  });

  it("branches from a completed turn and preserves context", async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue(baseConfig());
    streamFusion.mockImplementation(
      async (_payload: unknown, _token: unknown, handlers: StreamHandlers) => {
        handlers.onContent?.("original answer");
      },
    );

    render(<App />);
    await screen.findByDisplayValue("openai/gpt-4o");
    await user.type(screen.getByPlaceholderText("Ask anything…"), "original question");
    await user.click(screen.getByRole("button", { name: /Fuse/ }));
    await screen.findByText("original answer");

    // The Branch button is on the turn bubble (title="Branch from here").
    await user.click(screen.getByTitle("Branch from here"));

    // A "Branch of:" conversation should appear in the sidebar.
    expect(await screen.findByText(/Branch of:/)).toBeInTheDocument();
  });

  it("toggles the history sidebar open and closed", async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue(baseConfig());

    render(<App />);
    await screen.findByDisplayValue("openai/gpt-4o");

    // Sidebar open by default — "History" heading visible.
    expect(screen.getByText("History")).toBeInTheDocument();

    // Click the toggle (title="Toggle history") to close it.
    await user.click(screen.getByTitle("Toggle history"));
    expect(screen.queryByText("History")).not.toBeInTheDocument();

    // Click again to reopen.
    await user.click(screen.getByTitle("Toggle history"));
    expect(screen.getByText("History")).toBeInTheDocument();
  });
});
