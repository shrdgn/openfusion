import { afterEach, describe, expect, it, vi } from "vitest";
import { streamFusion, type ChatPayload, type StreamHandlers } from "../api";

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
