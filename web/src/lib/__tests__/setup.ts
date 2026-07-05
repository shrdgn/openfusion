import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";
import "@testing-library/jest-dom/vitest";

// testing-library's auto-cleanup only self-registers when vitest's `globals`
// option is on; this project keeps globals off and imports test functions
// explicitly, so unmount each rendered component ourselves between tests.
afterEach(() => {
  cleanup();
});
