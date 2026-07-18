import { afterEach, describe, expect, it, vi } from "vitest";
import {
  errorMessage,
  getConfig,
  getEstimate,
  setApiKey,
  streamFusion,
  type ChatPayload,
  type StreamHandlers,
} from "../api";

/** Builds a fetch Response whose body streams the given text chunks one read() at a time. */
function streamResponse(chunks: string[], init: { ok?: boolean; status?: number } = {}): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
  return new Response(body, { status: init.status ?? 200 });
}

function handlers(): StreamHandlers & { events: unknown[] } {
  const events: unknown[] = [];
  return {
    events,
    onProgress: (e) => events.push(["progress", e]),
    onPanelAnswer: (a) => events.push(["panel_answer", a]),
    onContent: (t) => events.push(["content", t]),
    onAnalysis: (a) => events.push(["analysis", a]),
    onUsage: (u) => events.push(["usage", u]),
    onError: (m) => events.push(["error", m]),
  };
}

const PAYLOAD: ChatPayload = { messages: [{ role: "user", content: "hi" }] };

describe("errorMessage", () => {
  it("returns the message of an Error instance", () => {
    expect(errorMessage(new Error("boom"))).toBe("boom");
  });

  it("stringifies non-Error values", () => {
    expect(errorMessage("plain string")).toBe("plain string");
    expect(errorMessage(404)).toBe("404");
  });
});

describe("streamFusion", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("parses content deltas across chunk boundaries", async () => {
    // The SSE block is split mid-line across two network reads.
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        streamResponse([
          'data: {"choices":[{"delta":{"content":"hel',
          'lo"},"finish_reason":null}]}\n\n',
          "data: [DONE]\n\n",
        ])
      )
    );
    const h = handlers();
    await streamFusion(PAYLOAD, undefined, h);
    expect(h.events).toEqual([["content", "hello"]]);
  });

  it("routes progress, panel_answer, analysis, and usage events by SSE event name", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        streamResponse([
          'event: progress\ndata: {"stage":"panel","total":2}\n\n',
          'event: panel_answer\ndata: {"model":"m1","label":"a","content":"x"}\n\n',
          'event: analysis\ndata: {"consensus":"agree"}\n\n',
          'event: usage\ndata: {"total":{"prompt_tokens":1}}\n\n',
          "data: [DONE]\n\n",
        ])
      )
    );
    const h = handlers();
    await streamFusion(PAYLOAD, undefined, h);
    expect(h.events).toEqual([
      ["progress", { stage: "panel", total: 2 }],
      ["panel_answer", { model: "m1", label: "a", content: "x" }],
      ["analysis", { consensus: "agree" }],
      ["usage", { total: { prompt_tokens: 1 } }],
    ]);
  });

  it("reports an inline error payload via onError instead of onContent", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        streamResponse([
          'data: {"error":{"message":"judge failed"}}\n\n',
          "data: [DONE]\n\n",
        ])
      )
    );
    const h = handlers();
    await streamFusion(PAYLOAD, undefined, h);
    expect(h.events).toEqual([["error", "judge failed"]]);
  });

  it("ignores malformed JSON blocks instead of throwing", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        streamResponse(["data: {not json\n\n", 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'])
      )
    );
    const h = handlers();
    await streamFusion(PAYLOAD, undefined, h);
    expect(h.events).toEqual([["content", "ok"]]);
  });

  it("calls onError with the server error message on a non-2xx response", async () => {
    const res = new Response(JSON.stringify({ error: { message: "bad request" } }), {
      status: 400,
    });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(res));
    const h = handlers();
    await streamFusion(PAYLOAD, undefined, h);
    expect(h.events).toEqual([["error", "bad request"]]);
  });

  it("calls onError with a reachability message when fetch itself throws", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("network down")));
    const h = handlers();
    await streamFusion(PAYLOAD, undefined, h);
    expect(h.events).toHaveLength(1);
    const [kind, message] = h.events[0] as [string, string];
    expect(kind).toBe("error");
    expect(message).toMatch(/couldn't reach the openfusion server/i);
  });

  it("falls back to a generic message when a non-2xx response body isn't valid JSON", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("not json", { status: 500 })));
    const h = handlers();
    await streamFusion(PAYLOAD, undefined, h);
    expect(h.events).toEqual([["error", "Request failed (500)"]]);
  });

  it("falls back to a generic upstream error message when the error chunk has no message", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(streamResponse(['data: {"error":{}}\n\n', "data: [DONE]\n\n"]))
    );
    const h = handlers();
    await streamFusion(PAYLOAD, undefined, h);
    expect(h.events).toEqual([["error", "upstream error"]]);
  });

  it("ignores delta chunks that carry no content (e.g. a finish_reason-only chunk)", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        streamResponse([
          'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
          "data: [DONE]\n\n",
        ])
      )
    );
    const h = handlers();
    await streamFusion(PAYLOAD, undefined, h);
    expect(h.events).toEqual([]);
  });

  it("ignores blank lines within an SSE block", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        streamResponse([
          '\ndata: {"choices":[{"delta":{"content":"ok"}}]}\n\n',
        ])
      )
    );
    const h = handlers();
    await streamFusion(PAYLOAD, undefined, h);
    expect(h.events).toEqual([["content", "ok"]]);
  });

  it("ignores an SSE block that has no data line", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        streamResponse([
          "event: progress\n\n",
          'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n',
        ])
      )
    );
    const h = handlers();
    await streamFusion(PAYLOAD, undefined, h);
    expect(h.events).toEqual([["content", "ok"]]);
  });

  it("flushes a final SSE block that has no trailing blank line", async () => {
    // No `\n\n` terminator on the last chunk — relies on the post-loop flush of `buffer`.
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(streamResponse(['data: {"choices":[{"delta":{"content":"tail"}}]}']))
    );
    const h = handlers();
    await streamFusion(PAYLOAD, undefined, h);
    expect(h.events).toEqual([["content", "tail"]]);
  });

  it("sends an Authorization header only when a token is provided", async () => {
    const fetchMock = vi.fn().mockImplementation(() =>
      Promise.resolve(streamResponse(["data: [DONE]\n\n"]))
    );
    vi.stubGlobal("fetch", fetchMock);

    await streamFusion(PAYLOAD, "secret-token", handlers());
    const [, initWithToken] = fetchMock.mock.calls[0];
    expect((initWithToken.headers as Record<string, string>).Authorization).toBe(
      "Bearer secret-token"
    );

    fetchMock.mockClear();
    await streamFusion(PAYLOAD, undefined, handlers());
    const [, initWithoutToken] = fetchMock.mock.calls[0];
    expect((initWithoutToken.headers as Record<string, string>).Authorization).toBeUndefined();
  });
});

describe("getEstimate", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("posts the payload to /v1/estimate and returns the parsed estimate", async () => {
    const estimate = {
      calls: 3,
      models: ["m1", "m2"],
      input_tokens: 42,
      max_output_tokens: 512,
      cost_usd: 0.01,
    };
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(estimate)));
    vi.stubGlobal("fetch", fetchMock);

    const result = await getEstimate(PAYLOAD);

    expect(result).toEqual(estimate);
    const [path, init] = fetchMock.mock.calls[0];
    expect(path).toBe("/v1/estimate");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body as string)).toEqual(PAYLOAD);
  });

  it("throws with the status code on a non-2xx response", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("nope", { status: 500 })));
    await expect(getEstimate(PAYLOAD)).rejects.toThrow("estimate 500");
  });

  it("rejects with a reachability error when fetch itself throws", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("network down")));
    await expect(getEstimate(PAYLOAD)).rejects.toThrow(/couldn't reach the openfusion server/i);
  });
});

describe("getConfig", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("fetches /v1/config and returns the parsed config", async () => {
    const config = {
      preset: "budget",
      strategy: "panel",
      aggregator: "judge",
      panel: ["m1"],
      judge: "m1",
      tools: { web_search: false, web_fetch: false },
      allow_request_overrides: true,
      allow_ui_api_key: true,
      needs_api_key: false,
      api_key_set: true,
      presets: {},
      fusion_model: "openfusion",
    };
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(config)));
    vi.stubGlobal("fetch", fetchMock);

    const result = await getConfig();

    expect(result).toEqual(config);
    expect(fetchMock.mock.calls[0][0]).toBe("/v1/config");
  });

  it("throws a descriptive error on a non-2xx response", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("nope", { status: 503 })));
    await expect(getConfig()).rejects.toThrow("Could not load config (503)");
  });
});

describe("setApiKey", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("posts the key to /v1/runtime/api-key and returns the parsed body", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(new Response(JSON.stringify({ api_key_set: true })));
    vi.stubGlobal("fetch", fetchMock);

    const result = await setApiKey("sk-secret");

    expect(result).toEqual({ api_key_set: true });
    const [path, init] = fetchMock.mock.calls[0];
    expect(path).toBe("/v1/runtime/api-key");
    expect(JSON.parse(init.body as string)).toEqual({ api_key: "sk-secret" });
  });

  it("throws the server's error message on a non-2xx response with a JSON body", async () => {
    const res = new Response(JSON.stringify({ error: { message: "invalid key" } }), {
      status: 400,
    });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(res));
    await expect(setApiKey("bad")).rejects.toThrow("invalid key");
  });

  it("falls back to a generic message when the error body isn't valid JSON", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("not json", { status: 502 })));
    await expect(setApiKey("bad")).rejects.toThrow("Failed to set key (502)");
  });
});
