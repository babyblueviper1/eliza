// Real desktop generic-GGUF generation (#8808 C3).
//
// Proves a non-Eliza-1 single-file GGUF loads + generates real text on desktop
// through `GenericGgufBackend` → `DesktopGenericLlamaLoader` → the libllama shim
// (built/staged by `native/eliza-generic-llama/build.mjs`). Runs only in the
// post-merge lane (a `.real.test.ts`). Honest-skip (never a false pass) when the
// shim or a test model is absent — point `ELIZA_GENERIC_GGUF_TEST_MODEL` at any
// GGUF (e.g. a Gemma `*.gguf`) to exercise it.
import { existsSync } from "node:fs";
import { describe, expect, it } from "vitest";
import type { BackendPlan } from "../backend";
import { GenericGgufBackend } from "../generic-gguf-backend";
import { genericLlamaShimPath } from "./desktop-generic-llama";

const MODEL = process.env.ELIZA_GENERIC_GGUF_TEST_MODEL ?? "";
const ready =
	MODEL !== "" && existsSync(MODEL) && existsSync(genericLlamaShimPath());
const lane = ready ? describe : describe.skip;

lane("GenericGgufBackend — desktop libllama real generation (#8808 C3)", () => {
	it("loads a non-eliza-1 GGUF and generates real text on desktop", async () => {
		const backend = new GenericGgufBackend();
		expect(await backend.available()).toBe(true);

		const plan = {
			modelId: "generic-real-test",
			modelPath: MODEL,
			runtimeClass: "generic-gguf",
			overrides: { contextSize: 2048, gpuLayers: 99 },
		} as unknown as BackendPlan;

		await backend.load(plan);
		expect(backend.hasLoadedModel()).toBe(true);
		expect(backend.currentModelPath()).toBe(MODEL);

		const result = await backend.generateWithUsage({
			prompt: "Q: What is the capital of France? A:",
			maxTokens: 24,
			temperature: 0,
		});
		// Real generated text, not gibberish (the fused path would gibberish-
		// tokenize a foreign GGUF; the libllama shim uses the model's own vocab).
		expect(result.text.trim().length).toBeGreaterThan(0);
		expect(result.usage.completion_tokens).toBeGreaterThan(0);
		expect(result.usage.prompt_tokens).toBeGreaterThan(0);

		await backend.unload();
		expect(backend.hasLoadedModel()).toBe(false);
	}, 120_000);
});
