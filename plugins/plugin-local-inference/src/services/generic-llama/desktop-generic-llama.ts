/**
 * Desktop generic-GGUF runtime (#8808 C3).
 *
 * The explicit-`modelPath` text runtime for a non-Eliza-1 single-file GGUF on
 * **desktop/server** — the piece the shipping fused `libelizainference` could
 * not provide (it tokenizes with the eliza-1 bundle vocab, so a foreign GGUF
 * would gibberish-tokenize). This binds a thin C ABI over the vendored
 * llama.cpp (`native/eliza-generic-llama/shim.cpp`, built + staged by its
 * `build.mjs`) through `bun:ffi`, loading the model's OWN tokenizer/vocab.
 *
 * It implements the same `ExplicitModelPathLoader` → `CapacitorLlamaContext`
 * contract the mobile `llama-cpp-capacitor` binding does, so `GenericGgufBackend`
 * routes through it unchanged. Available only under Bun, on a non-mobile
 * platform, when the staged shim is present; otherwise `available()` is false
 * and the dispatcher raises `GenericRuntimeUnavailableError` as before.
 */

import { existsSync } from "node:fs";
import path from "node:path";
import { resolveStateDir } from "@elizaos/core";
import {
	CapacitorLlamaUnsupportedError,
	type CapacitorLlamaBenchResult,
	type CapacitorLlamaCompletionParams,
	type CapacitorLlamaCompletionResult,
	type CapacitorLlamaContext,
	type CapacitorLlamaEmbeddingResult,
	type CapacitorLlamaModelDescriptor,
	type CapacitorLlamaTokenData,
	type CapacitorLlamaTokenizeResult,
} from "../../adapters/capacitor-llama/types";
import type { ExplicitModelPathLoader } from "../generic-gguf-backend";

/** Native handle is an opaque pointer (a number under bun:ffi). */
type ShimHandle = number;

interface ShimSymbols {
	egl_init(): void;
	egl_load(
		path: Uint8Array,
		gpuLayers: number,
		nCtx: number,
	): ShimHandle | null;
	egl_generate(
		handle: ShimHandle,
		prompt: Uint8Array,
		maxTokens: number,
		temperature: number,
		topP: number,
		out: Uint8Array,
		outCap: number,
		nEval: Int32Array,
		nPred: Int32Array,
	): number;
	egl_free(handle: ShimHandle): void;
}

declare const Bun: unknown;

/**
 * Minimal surface of the Bun `bun:ffi` runtime builtin used here. Imported
 * through a non-literal specifier (below) so the typechecker doesn't need
 * `@types/bun` in its type roots — the module only loads at runtime under Bun.
 */
interface BunFfi {
	dlopen(
		path: string,
		symbols: Record<string, { args: unknown[]; returns: unknown }>,
	): { symbols: Record<string, (...args: never[]) => unknown> };
	FFIType: Record<"i32" | "f32" | "ptr" | "void", unknown>;
}

function isMobilePlatform(): boolean {
	const platform = process.env.ELIZA_PLATFORM?.trim().toLowerCase();
	return platform === "android" || platform === "ios";
}

/** Resolve the staged shim dylib (built by native/eliza-generic-llama/build.mjs). */
export function genericLlamaShimPath(): string {
	const override = process.env.ELIZA_GENERIC_LLAMA_LIB?.trim();
	if (override) return override;
	return path.join(
		resolveStateDir(),
		"local-inference",
		"lib",
		"generic-llama",
		"libeglshim.dylib",
	);
}

let cachedSymbols: ShimSymbols | null = null;

async function openShim(): Promise<ShimSymbols | null> {
	if (cachedSymbols) return cachedSymbols;
	if (typeof Bun === "undefined") return null;
	const libPath = genericLlamaShimPath();
	if (!existsSync(libPath)) return null;
	const specifier = "bun:ffi";
	const ffi = (await import(specifier)) as unknown as BunFfi;
	const { i32, f32, ptr: P, void: V } = ffi.FFIType;
	const lib = ffi.dlopen(libPath, {
		egl_init: { args: [], returns: V },
		egl_load: { args: [P, i32, i32], returns: P },
		egl_generate: {
			args: [P, P, i32, f32, f32, P, i32, P, P],
			returns: i32,
		},
		egl_free: { args: [P], returns: V },
	});
	(lib.symbols.egl_init as () => void)();
	cachedSymbols = lib.symbols as unknown as ShimSymbols;
	return cachedSymbols;
}

const encoder = new TextEncoder();
const decoder = new TextDecoder();

/** NUL-terminate a JS string as bytes for the C ABI. */
function cstr(s: string): Uint8Array {
	return encoder.encode(`${s}\0`);
}

const NO_TOOL_CAPS = {
	tools: false,
	toolCalls: false,
	toolResponses: false,
	systemRole: false,
	parallelToolCalls: false,
	toolCallId: false,
} as const;

const DESKTOP_MODEL_DESCRIPTOR: CapacitorLlamaModelDescriptor = {
	desc: "generic-gguf (desktop libllama)",
	size: 0,
	nEmbd: 0,
	nParams: 0,
	chatTemplates: {
		llamaChat: false,
		minja: {
			default: false,
			defaultCaps: { ...NO_TOOL_CAPS },
			toolUse: false,
			toolUseCaps: { ...NO_TOOL_CAPS },
		},
	},
	metadata: {},
	isChatTemplateSupported: false,
};

let nextId = 1;

class DesktopGenericLlamaContext implements CapacitorLlamaContext {
	readonly id = nextId++;
	readonly gpu = true;
	readonly reasonNoGPU = "";
	readonly model = DESKTOP_MODEL_DESCRIPTOR;
	private handle: ShimHandle | null;

	constructor(
		handle: ShimHandle,
		private readonly syms: ShimSymbols,
	) {
		this.handle = handle;
	}

	async completion(
		params: CapacitorLlamaCompletionParams,
		callback?: (data: CapacitorLlamaTokenData) => void,
	): Promise<CapacitorLlamaCompletionResult> {
		if (this.handle === null) {
			throw new Error("[desktop-generic-llama] completion after release()");
		}
		const prompt = params.prompt ?? "";
		const nPredict = params.n_predict ?? 2048;
		const temperature = params.temperature ?? 0.7;
		const topP = params.top_p ?? 0.9;
		// ~16 bytes/token is a generous upper bound for multibyte pieces.
		const cap = Math.max(4096, nPredict * 16 + 256);
		const out = new Uint8Array(cap);
		const nEval = new Int32Array(1);
		const nPred = new Int32Array(1);
		const written = this.syms.egl_generate(
			this.handle,
			cstr(prompt),
			nPredict,
			temperature,
			topP,
			out,
			cap,
			nEval,
			nPred,
		);
		if (written < 0) {
			throw new Error("[desktop-generic-llama] generation failed");
		}
		let text = decoder.decode(out.subarray(0, written));
		// The shim is stateless wrt stop strings; truncate at the first one.
		let stoppedWord = "";
		for (const stop of params.stop ?? []) {
			if (!stop) continue;
			const idx = text.indexOf(stop);
			if (idx >= 0) {
				text = text.slice(0, idx);
				stoppedWord = stop;
				break;
			}
		}
		// No token-level streaming from the synchronous shim: emit the whole
		// completion as a single chunk so `onTextChunk` consumers still fire.
		if (callback && text) {
			callback({ token: text, content: text, accumulated_text: text });
		}
		return {
			text,
			content: text,
			reasoning_content: "",
			tool_calls: [],
			chat_format: 0,
			tokens_predicted: nPred[0] ?? 0,
			tokens_evaluated: nEval[0] ?? 0,
			truncated: false,
			stopped_eos: stoppedWord === "",
			stopped_word: stoppedWord,
			stopped_limit: stoppedWord === "" ? 1 : 0,
			stopping_word: stoppedWord,
			context_full: false,
			interrupted: false,
			tokens_cached: 0,
			timings: {
				prompt_n: nEval[0] ?? 0,
				prompt_ms: 0,
				prompt_per_token_ms: 0,
				prompt_per_second: 0,
				predicted_n: nPred[0] ?? 0,
				predicted_ms: 0,
				predicted_per_token_ms: 0,
				predicted_per_second: 0,
			},
		};
	}

	async stopCompletion(): Promise<void> {
		// The shim generates synchronously per call; there is no in-flight loop
		// to interrupt. No-op.
	}

	tokenize(): Promise<CapacitorLlamaTokenizeResult> {
		throw new CapacitorLlamaUnsupportedError("tokenize", "desktop-ffi");
	}

	detokenize(): Promise<string> {
		throw new CapacitorLlamaUnsupportedError("detokenize", "desktop-ffi");
	}

	embedding(): Promise<CapacitorLlamaEmbeddingResult> {
		throw new CapacitorLlamaUnsupportedError("embedding", "desktop-ffi");
	}

	bench(): Promise<CapacitorLlamaBenchResult> {
		throw new CapacitorLlamaUnsupportedError("bench", "desktop-ffi");
	}

	async release(): Promise<void> {
		if (this.handle === null) return;
		this.syms.egl_free(this.handle);
		this.handle = null;
	}
}

/**
 * Desktop explicit-`modelPath` loader. `available()` is true only under Bun, on
 * a non-mobile platform, with the staged shim present.
 */
export class DesktopGenericLlamaLoader implements ExplicitModelPathLoader {
	async available(): Promise<boolean> {
		if (isMobilePlatform()) return false;
		if (typeof Bun === "undefined") return false;
		return existsSync(genericLlamaShimPath());
	}

	async load(args: {
		modelPath: string;
		contextSize?: number;
		gpuLayers?: number;
	}): Promise<CapacitorLlamaContext> {
		const syms = await openShim();
		if (!syms) {
			throw new Error(
				"[desktop-generic-llama] shim unavailable — build it via " +
					"`node plugins/plugin-local-inference/native/eliza-generic-llama/build.mjs`",
			);
		}
		const handle = syms.egl_load(
			cstr(args.modelPath),
			args.gpuLayers ?? 99,
			args.contextSize ?? 4096,
		);
		if (!handle) {
			throw new Error(
				`[desktop-generic-llama] failed to load GGUF: ${args.modelPath}`,
			);
		}
		return new DesktopGenericLlamaContext(handle, syms);
	}
}
