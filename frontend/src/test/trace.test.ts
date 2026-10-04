import { describe, expect, it } from "vitest";
import type { StepRecord } from "@/lib/api";
import { downstream, explainReason, statusOf, summarize } from "@/lib/trace";

const step = (over: Partial<StepRecord>): StepRecord => ({
  step_idx: 0,
  type: "retrieve",
  input: {},
  output: {},
  state_before: {},
  parents: [],
  cache_hit: false,
  latency_ms: 0,
  error: null,
  blame_score: 0,
  blame_rank: null,
  reasons: [],
  direct_evidence: [],
  ...over,
});

describe("trace helpers", () => {
  it("summarises each step type from its recorded output", () => {
    expect(summarize("retrieve", { title: "Zenit Labs" })).toBe("Zenit Labs");
    expect(summarize("extract", { field: "founded", value: 2015 })).toBe("founded = 2015");
    expect(summarize("calculate", { expr: "a - b", result: 6 })).toBe("a - b = 6");
    expect(
      summarize("plan", {
        actions: [{ action: "retrieve", name: "Zenith" }, { action: "answer" }],
      }),
    ).toBe("retrieve(Zenith) → answer");
  });

  it("turns SHAP reasons into sentences grounded in the step", () => {
    const s = step({ input: { name: "Zenith" }, output: { title: "Zenit Labs" } });
    expect(explainReason({ feature: "grounding_match", value: 0.625, shap_value: 4.4 }, s)).toBe(
      "retrieved “Zenit Labs” for requested “Zenith” (name match 0.63)",
    );
    expect(explainReason({ feature: "extracted_in_source", value: 0, shap_value: 2 }, s)).toBe(
      "not an extract step, so extracted_in_source does not apply",
    );
  });

  it("follows the data-flow graph downstream", () => {
    const steps = [
      step({ step_idx: 0 }),
      step({ step_idx: 1, parents: [0] }),
      step({ step_idx: 2, parents: [1] }),
      step({ step_idx: 3 }),
    ];
    expect(downstream(steps, 0)).toEqual([1, 2]);
  });

  it("maps blame scores to statuses", () => {
    expect([statusOf(0.9), statusOf(0.3), statusOf(0.1)]).toEqual(["suspect", "watch", "ok"]);
  });
});
