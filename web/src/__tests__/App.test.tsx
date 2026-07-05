import { render, screen, waitFor } from "@testing-library/react";
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

vi.mock("../lib/api", () => ({ getConfig, getEstimate, setApiKey, streamFusion }));

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
});
