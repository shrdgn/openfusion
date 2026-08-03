import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Markdown } from "../markdown";

describe("Markdown", () => {
  it("renders plain text content", () => {
    render(<Markdown>{"hello world"}</Markdown>);
    expect(screen.getByText("hello world")).toBeInTheDocument();
  });

  it("renders GFM formatting (bold, list, table) as real elements", () => {
    const source = [
      "**bold** and _italic_",
      "",
      "- one",
      "- two",
      "",
      "| a | b |",
      "| - | - |",
      "| 1 | 2 |",
    ].join("\n");
    const { container } = render(<Markdown>{source}</Markdown>);

    expect(screen.getByText("bold").tagName).toBe("STRONG");
    expect(screen.getAllByRole("listitem")).toHaveLength(2);
    expect(container.querySelector("table")).not.toBeNull();
  });

  it("renders a raw <script> tag as inert escaped text, never a live element", () => {
    // react-markdown (no rehype-raw plugin wired up) renders raw HTML in the
    // source as literal escaped text rather than parsing it into DOM nodes —
    // this is the entire XSS-safety property this component relies on for
    // untrusted LLM output. Lock in both halves of that behavior so a future
    // change (e.g. adding rehype-raw for HTML tables) trips a test instead of
    // silently reintroducing script execution.
    const { container } = render(
      <Markdown>{"before <script>window.__pwned = true;</script> after"}</Markdown>
    );

    expect(container.querySelector("script")).toBeNull();
    expect(container.textContent).toContain("<script>window.__pwned = true;</script>");
    expect((window as unknown as { __pwned?: boolean }).__pwned).toBeUndefined();
  });

  it("renders a raw <img onerror> handler as inert escaped text, never a live element", () => {
    const { container } = render(
      <Markdown>{'before <img src="x" onerror="window.__pwned = true"> after'}</Markdown>
    );

    expect(container.querySelector("img")).toBeNull();
    expect(container.textContent).toContain('<img src="x" onerror="window.__pwned = true">');
  });

  it("renders a markdown link as an anchor without executing javascript: hrefs as scripts", () => {
    render(<Markdown>{"[click me](https://example.com)"}</Markdown>);
    const link = screen.getByRole("link", { name: "click me" });
    expect(link).toHaveAttribute("href", "https://example.com");
  });
});
