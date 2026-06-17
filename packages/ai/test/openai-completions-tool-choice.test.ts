import { Type } from "typebox";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { getModel } from "../src/models.ts";
import { convertMessages } from "../src/providers/openai-completions.ts";
import { stream, streamSimple } from "../src/stream.ts";
import type { AssistantMessage, Model, Tool, ToolResultMessage } from "../src/types.ts";

const mockState = vi.hoisted(() => ({
	lastParams: undefined as unknown,
	chunks: undefined as
		| Array<null | {
				id?: string;
				choices?: Array<{ delta: Record<string, unknown>; finish_reason: string | null; usage?: unknown }>;
				usage?: {
					prompt_tokens: number;
					completion_tokens: number;
					prompt_tokens_details: { cached_tokens: number; cache_write_tokens?: number };
					completion_tokens_details: { reasoning_tokens: number };
				};
		  }>
		| undefined,
}));

vi.mock("openai", () => {
	class FakeOpenAI {
		chat = {
			completions: {
				create: (params: unknown) => {
					mockState.lastParams = params;
					const stream = {
						async *[Symbol.asyncIterator]() {
							const chunks = mockState.chunks ?? [
								{
									choices: [{ delta: {}, finish_reason: "stop" }],
									usage: {
										prompt_tokens: 1,
										completion_tokens: 1,
										prompt_tokens_details: { cached_tokens: 0 },
										completion_tokens_details: { reasoning_tokens: 0 },
									},
								},
							];
							for (const chunk of chunks) {
								yield chunk;
							}
						},
					};
					const promise = Promise.resolve(stream) as Promise<typeof stream> & {
						withResponse: () => Promise<{
							data: typeof stream;
							response: { status: number; headers: Headers };
						}>;
					};
					promise.withResponse = async () => ({
						data: stream,
						response: { status: 200, headers: new Headers() },
					});
					return promise;
				},
			},
		};
	}

	return { default: FakeOpenAI };
});

describe("openai-completions tool_choice", () => {
	beforeEach(() => {
		mockState.lastParams = undefined;
		mockState.chunks = undefined;
	});

	it("forwards toolChoice from simple options to payload", async () => {
		const { compat: _compat, ...baseModel } = getModel("openai", "gpt-4o-mini")!;
		const model = { ...baseModel, api: "openai-completions" } as const;
		const tools: Tool[] = [
			{
				name: "ping",
				description: "Ping tool",
				parameters: Type.Object({
					ok: Type.Boolean(),
				}),
			},
		];
		let payload: unknown;

		await streamSimple(
			model,
			{
				messages: [
					{
						role: "user",
						content: "Call ping with ok=true",
						timestamp: Date.now(),
					},
				],
				tools,
			},
			{
				apiKey: "test",
				toolChoice: "required",
				onPayload: (params: unknown) => {
					payload = params;
				},
			} as unknown as Parameters<typeof streamSimple>[2],
		).result();

		const params = (payload ?? mockState.lastParams) as {
			tool_choice?: string;
			tools?: unknown[];
			messages?: unknown[];
		};
		expect(params.tool_choice).toBeUndefined();
		expect("tools" in (params as object)).toBe(false);
		expect(JSON.stringify(params.messages)).toContain("Enabled tool names: ping");
	});

	it("omits strict when compat disables strict mode", async () => {
		const { compat: _compat, ...baseModel } = getModel("openai", "gpt-4o-mini")!;
		const model = {
			...baseModel,
			api: "openai-completions",
			compat: { supportsStrictMode: false },
		} as const;
		const tools: Tool[] = [
			{
				name: "ping",
				description: "Ping tool",
				parameters: Type.Object({
					ok: Type.Boolean(),
				}),
			},
		];
		let payload: unknown;

		await streamSimple(
			model,
			{
				messages: [
					{
						role: "user",
						content: "Call ping with ok=true",
						timestamp: Date.now(),
					},
				],
				tools,
			},
			{
				apiKey: "test",
				onPayload: (params: unknown) => {
					payload = params;
				},
			} as unknown as Parameters<typeof streamSimple>[2],
		).result();

		const params = (payload ?? mockState.lastParams) as { tools?: unknown[]; messages?: unknown[] };
		expect("tools" in (params as object)).toBe(false);
		expect(JSON.stringify(params.messages)).toContain("Enabled tool names: ping");
		expect(JSON.stringify(params.messages)).toContain("ok");
	});

	it("maps groq qwen3 reasoning levels to default reasoning_effort", async () => {
		const model = getModel("groq", "qwen/qwen3-32b")!;
		let payload: unknown;

		await streamSimple(
			model,
			{
				messages: [
					{
						role: "user",
						content: "Hi",
						timestamp: Date.now(),
					},
				],
			},
			{
				apiKey: "test",
				reasoning: "medium",
				onPayload: (params: unknown) => {
					payload = params;
				},
			},
		).result();

		const params = (payload ?? mockState.lastParams) as { reasoning_effort?: string };
		expect(params.reasoning_effort).toBe("default");
	});

	it("keeps normal reasoning_effort for groq models without compat mapping", async () => {
		const model = getModel("groq", "openai/gpt-oss-20b")!;
		let payload: unknown;

		await streamSimple(
			model,
			{
				messages: [
					{
						role: "user",
						content: "Hi",
						timestamp: Date.now(),
					},
				],
			},
			{
				apiKey: "test",
				reasoning: "medium",
				onPayload: (params: unknown) => {
					payload = params;
				},
			},
		).result();

		const params = (payload ?? mockState.lastParams) as { reasoning_effort?: string };
		expect(params.reasoning_effort).toBe("medium");
	});

	it("does not emit native z.ai tool_stream even for z.ai models with tools", async () => {
		const tools: Tool[] = [
			{
				name: "read",
				description: "Read files",
				parameters: Type.Object({
					path: Type.String(),
				}),
			},
		];
		let payload: unknown;
		const model = getModel("zai", "glm-4.7")!;

		await streamSimple(
			model,
			{
				messages: [{ role: "user", content: "Read package.json", timestamp: Date.now() }],
				tools,
			},
			{
				apiKey: "test",
				onPayload: (params) => {
					payload = params;
				},
			},
		).result();

		const params = (payload ?? mockState.lastParams) as {
			tool_stream?: boolean;
			tools?: unknown[];
			messages?: unknown[];
		};
		expect(params.tool_stream).toBeUndefined();
		expect(params.tools).toBeUndefined();
		expect(JSON.stringify(params.messages)).toContain("Provider-native JSON/function tools are disabled");
	});

	it("omits tool_stream when no tools are provided", async () => {
		const model = getModel("zai", "glm-5.1")!;
		let payload: unknown;

		await streamSimple(
			model,
			{
				messages: [
					{
						role: "user",
						content: "Hi",
						timestamp: Date.now(),
					},
				],
			},
			{
				apiKey: "test",
				onPayload: (params: unknown) => {
					payload = params;
				},
			},
		).result();

		const params = (payload ?? mockState.lastParams) as { tool_stream?: boolean };
		expect(params.tool_stream).toBeUndefined();
	});

	it("maps non-standard provider finish_reason values to stopReason error", async () => {
		mockState.chunks = [
			{
				choices: [{ delta: { content: "partial" }, finish_reason: null }],
			},
			{
				choices: [{ delta: {}, finish_reason: "network_error" }],
				usage: {
					prompt_tokens: 1,
					completion_tokens: 1,
					prompt_tokens_details: { cached_tokens: 0 },
					completion_tokens_details: { reasoning_tokens: 0 },
				},
			},
		];

		const model = getModel("zai", "glm-5.1")!;
		const response = await streamSimple(
			model,
			{
				messages: [
					{
						role: "user",
						content: "Hi",
						timestamp: Date.now(),
					},
				],
			},
			{ apiKey: "test" },
		).result();

		expect(response.stopReason).toBe("error");
		expect(response.errorMessage).toBe("Provider finish_reason: network_error");
	});

	it("ignores null stream chunks from openai-compatible providers", async () => {
		mockState.chunks = [
			null,
			{
				id: "chatcmpl-test",
				choices: [{ delta: { content: "OK" }, finish_reason: null }],
			},
			{
				id: "chatcmpl-test",
				choices: [{ delta: {}, finish_reason: "stop" }],
				usage: {
					prompt_tokens: 3,
					completion_tokens: 1,
					prompt_tokens_details: { cached_tokens: 0 },
					completion_tokens_details: { reasoning_tokens: 0 },
				},
			},
		];

		const { compat: _compat, ...baseModel } = getModel("openai", "gpt-4o-mini")!;
		const model = { ...baseModel, api: "openai-completions" } as const;
		const response = await streamSimple(
			model,
			{
				messages: [
					{
						role: "user",
						content: "Reply with exactly OK",
						timestamp: Date.now(),
					},
				],
			},
			{ apiKey: "test" },
		).result();

		expect(response.stopReason).toBe("stop");
		expect(response.errorMessage).toBeUndefined();
		expect(response.responseId).toBe("chatcmpl-test");
		expect(response.usage.totalTokens).toBe(4);
		expect(response.content).toEqual([{ type: "text", text: "OK" }]);
	});

	it("errors when a stream ends after only null finish_reason chunks", async () => {
		mockState.chunks = [
			{
				id: "chatcmpl-truncated",
				choices: [{ delta: { content: "partial answer" }, finish_reason: null }],
			},
			{
				id: "chatcmpl-truncated",
				choices: [{ delta: { content: "partial answer" }, finish_reason: null }],
			},
		];

		const { compat: _compat, ...baseModel } = getModel("openai", "gpt-4o-mini")!;
		const model = { ...baseModel, api: "openai-completions" } as const;
		const response = await streamSimple(
			model,
			{
				messages: [
					{
						role: "user",
						content: "Reply with a longer sentence",
						timestamp: Date.now(),
					},
				],
			},
			{ apiKey: "test" },
		).result();

		expect(response.stopReason).toBe("error");
		expect(response.errorMessage).toBe("Stream ended without finish_reason");
	});

	it("parses streamed text tool blocks into tool calls", async () => {
		mockState.chunks = [
			{
				id: "chatcmpl-text-tool",
				choices: [
					{
						delta: { content: '<read path="README' },
						finish_reason: null,
					},
				],
			},
			{
				id: "chatcmpl-text-tool",
				choices: [
					{
						delta: { content: '.md"></read>' },
						finish_reason: "stop",
					},
				],
				usage: {
					prompt_tokens: 10,
					completion_tokens: 5,
					prompt_tokens_details: { cached_tokens: 0 },
					completion_tokens_details: { reasoning_tokens: 0 },
				},
			},
		];

		const { compat: _compat, ...baseModel } = getModel("openai", "gpt-4o-mini")!;
		const model = { ...baseModel, api: "openai-completions" } as const;
		const tool: Tool = {
			name: "read",
			description: "Read a file",
			parameters: Type.Object({
				path: Type.String(),
			}),
		};
		const s = streamSimple(
			model,
			{
				messages: [
					{
						role: "user",
						content: "Read README.md",
						timestamp: Date.now(),
					},
				],
				tools: [tool],
			},
			{ apiKey: "test" },
		);

		const toolCallContentIndexes: number[] = [];
		for await (const event of s) {
			if (event.type === "toolcall_start" || event.type === "toolcall_delta" || event.type === "toolcall_end") {
				toolCallContentIndexes.push(event.contentIndex);
			}
		}

		const response = await s.result();
		expect(response.stopReason).toBe("toolUse");
		expect(toolCallContentIndexes).toEqual([0, 0, 0]);
		expect(response.content).toHaveLength(1);
		const toolCall = response.content[0];
		expect(toolCall.type).toBe("toolCall");
		if (toolCall.type !== "toolCall") {
			throw new Error("Expected toolCall content");
		}
		expect(toolCall.name).toBe("read");
		expect(toolCall.arguments).toEqual({ path: "README.md" });
		expect(toolCall).not.toHaveProperty("streamIndex");
		expect(toolCall).not.toHaveProperty("partialArgs");
	});

	it("accumulates mixed content, reasoning, and parallel text tool blocks independently", async () => {
		mockState.chunks = [
			{
				id: "chatcmpl-mixed-text-tools",
				choices: [
					{
						delta: {
							content: 'answer 1 <read path="README',
							reasoning_content: "think 1",
						},
						finish_reason: null,
					},
				],
			},
			{
				id: "chatcmpl-mixed-text-tools",
				choices: [
					{
						delta: {
							content: '.md"></read> <grep pattern="TODO" path="src"></grep>',
						},
						finish_reason: null,
					},
				],
			},
			{
				id: "chatcmpl-mixed-text-tools",
				choices: [
					{
						delta: {
							content: '\n<ls path="packages/ai"></ls>\n' + '<write path="out.txt">\nok\n</write>',
							reasoning_content: " think 2",
						},
						finish_reason: "stop",
					},
				],
				usage: {
					prompt_tokens: 10,
					completion_tokens: 8,
					prompt_tokens_details: { cached_tokens: 0 },
					completion_tokens_details: { reasoning_tokens: 2 },
				},
			},
		];

		const { compat: _compat, ...baseModel } = getModel("openai", "gpt-4o-mini")!;
		const model = { ...baseModel, api: "openai-completions" } as const;
		const tools: Tool[] = [
			{
				name: "read",
				description: "Read a file",
				parameters: Type.Object({ path: Type.String() }),
			},
			{
				name: "grep",
				description: "Search a file",
				parameters: Type.Object({ pattern: Type.String(), path: Type.String() }),
			},
			{
				name: "ls",
				description: "List a directory",
				parameters: Type.Object({ path: Type.String() }),
			},
			{
				name: "write",
				description: "Write a file",
				parameters: Type.Object({ path: Type.String(), content: Type.String() }),
			},
		];
		const s = streamSimple(
			model,
			{
				messages: [
					{
						role: "user",
						content: "Think, answer, and use tools.",
						timestamp: Date.now(),
					},
				],
				tools,
			},
			{ apiKey: "test" },
		);

		const eventTypes: string[] = [];
		const toolEventsByContentIndex = new Map<number, string[]>();
		for await (const event of s) {
			eventTypes.push(event.type);
			if (event.type === "toolcall_start" || event.type === "toolcall_delta" || event.type === "toolcall_end") {
				const events = toolEventsByContentIndex.get(event.contentIndex) ?? [];
				events.push(event.type);
				toolEventsByContentIndex.set(event.contentIndex, events);
			}
		}

		const response = await s.result();
		expect(response.stopReason).toBe("toolUse");
		expect(eventTypes.filter((type) => type === "text_start")).toHaveLength(1);
		expect(eventTypes.filter((type) => type === "text_delta")).toHaveLength(3);
		expect(eventTypes.filter((type) => type === "text_end")).toHaveLength(1);
		expect(eventTypes.filter((type) => type === "thinking_start")).toHaveLength(1);
		expect(eventTypes.filter((type) => type === "thinking_delta")).toHaveLength(2);
		expect(eventTypes.filter((type) => type === "thinking_end")).toHaveLength(1);
		expect(eventTypes.filter((type) => type === "toolcall_start")).toHaveLength(4);
		expect(eventTypes.filter((type) => type === "toolcall_delta")).toHaveLength(4);
		expect(eventTypes.filter((type) => type === "toolcall_end")).toHaveLength(4);
		expect(toolEventsByContentIndex.get(2)).toEqual(["toolcall_start", "toolcall_delta", "toolcall_end"]);
		expect(toolEventsByContentIndex.get(3)).toEqual(["toolcall_start", "toolcall_delta", "toolcall_end"]);
		expect(toolEventsByContentIndex.get(4)).toEqual(["toolcall_start", "toolcall_delta", "toolcall_end"]);
		expect(toolEventsByContentIndex.get(5)).toEqual(["toolcall_start", "toolcall_delta", "toolcall_end"]);

		expect(response.content).toHaveLength(6);
		expect(response.content[0]).toMatchObject({ type: "text" });
		if (response.content[0].type !== "text") throw new Error("Expected text content");
		expect(response.content[0].text.trim()).toBe("answer 1");
		expect(response.content[1]).toMatchObject({ type: "thinking", thinking: "think 1 think 2" });
		expect(response.content[2]).toMatchObject({ type: "toolCall", name: "read", arguments: { path: "README.md" } });
		expect(response.content[3]).toMatchObject({
			type: "toolCall",
			name: "grep",
			arguments: { pattern: "TODO", path: "src" },
		});
		expect(response.content[4]).toMatchObject({ type: "toolCall", name: "ls", arguments: { path: "packages/ai" } });
		expect(response.content[5]).toMatchObject({
			type: "toolCall",
			name: "write",
			arguments: { path: "out.txt", content: "ok" },
		});
	});

	it("uses system messages for non-OpenAI/Anthropic OpenRouter reasoning model instructions", async () => {
		const model = getModel("openrouter", "deepseek/deepseek-v4-pro")!;
		let payload: unknown;

		await streamSimple(
			model,
			{
				systemPrompt: "Follow instructions.",
				messages: [{ role: "user", content: "Hi", timestamp: Date.now() }],
			},
			{
				apiKey: "test",
				onPayload: (params: unknown) => {
					payload = params;
				},
			},
		).result();

		const params = payload as { messages?: Array<{ role?: string }> };
		expect(params.messages?.[0]?.role).toBe("system");
	});

	it("keeps developer messages for OpenAI and Anthropic OpenRouter reasoning model instructions", async () => {
		for (const model of [
			getModel("openrouter", "openai/gpt-5.2-codex"),
			getModel("openrouter", "anthropic/claude-sonnet-4.5"),
		]) {
			expect(model).toBeDefined();
			let payload: unknown;

			await streamSimple(
				model!,
				{
					systemPrompt: "Follow instructions.",
					messages: [{ role: "user", content: "Hi", timestamp: Date.now() }],
				},
				{
					apiKey: "test",
					onPayload: (params: unknown) => {
						payload = params;
					},
				},
			).result();

			const params = payload as { messages?: Array<{ role?: string }> };
			expect(params.messages?.[0]?.role).toBe("developer");
		}
	});

	it("keeps developer messages for OpenAI reasoning model instructions", async () => {
		const { compat: _compat, ...baseModel } = getModel("openai", "gpt-5.5")!;
		const model = { ...baseModel, api: "openai-completions" } as const;
		let payload: unknown;

		await streamSimple(
			model,
			{
				systemPrompt: "Follow instructions.",
				messages: [{ role: "user", content: "Hi", timestamp: Date.now() }],
			},
			{
				apiKey: "test",
				onPayload: (params: unknown) => {
					payload = params;
				},
			},
		).result();

		const params = payload as { messages?: Array<{ role?: string }> };
		expect(params.messages?.[0]?.role).toBe("developer");
	});

	it("stores OpenRouter Kimi K2.6 reasoning replay compat in built-in metadata", () => {
		const model = getModel("openrouter", "moonshotai/kimi-k2.6")!;
		expect(model.compat?.supportsDeveloperRole).toBe(false);
		expect(model.compat?.requiresReasoningContentOnAssistantMessages).toBe(true);
	});

	it("stores Xiaomi MiMo reasoning replay compat in built-in metadata", () => {
		const providers = ["xiaomi", "xiaomi-token-plan-cn", "xiaomi-token-plan-ams", "xiaomi-token-plan-sgp"] as const;

		for (const provider of providers) {
			const model = getModel(provider, "mimo-v2.5-pro")!;
			expect(model.compat?.requiresReasoningContentOnAssistantMessages).toBe(true);
			expect(model.compat?.thinkingFormat).toBe("deepseek");
			expect(model.compat?.maxTokensField).toBeUndefined();
			expect(model.compat?.supportsDeveloperRole).toBeUndefined();
		}
	});

	it("replays Xiaomi MiMo assistant tool calls with empty reasoning_content when thinking is missing", async () => {
		const model = getModel("xiaomi", "mimo-v2.5-pro")!;
		const assistantMessage: AssistantMessage = {
			role: "assistant",
			api: "openai-completions",
			provider: "xiaomi",
			model: "mimo-v2.5-pro",
			content: [{ type: "toolCall", id: "call_1", name: "read", arguments: { path: "README.md" } }],
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "toolUse",
			timestamp: Date.now(),
		};
		const toolResult: ToolResultMessage = {
			role: "toolResult",
			toolCallId: "call_1",
			toolName: "read",
			content: [{ type: "text", text: "contents" }],
			isError: false,
			timestamp: Date.now(),
		};
		let payload: unknown;

		await streamSimple(
			model,
			{
				messages: [
					{ role: "user", content: "Read README.md", timestamp: Date.now() },
					assistantMessage,
					toolResult,
				],
			},
			{
				apiKey: "test",
				reasoning: "high",
				onPayload: (params: unknown) => {
					payload = params;
				},
			},
		).result();

		const params = (payload ?? mockState.lastParams) as {
			messages?: Array<Record<string, unknown>>;
			thinking?: { type?: string };
			reasoning_effort?: string;
		};
		const replayedAssistant = params.messages?.find((message) => message.role === "assistant");
		expect(replayedAssistant).toMatchObject({ role: "assistant", reasoning_content: "" });
		expect(params.thinking).toEqual({ type: "enabled" });
		expect(params.reasoning_effort).toBe("high");
	});

	it("normalizes OpenCode Go reasoning deltas to reasoning_content for replay", async () => {
		mockState.chunks = [
			{
				id: "chatcmpl-opencode-go-reasoning",
				choices: [{ delta: { reasoning: "think" }, finish_reason: "stop" }],
			},
		];

		const { compat: _compat, ...baseModel } = getModel("opencode-go", "kimi-k2.6")!;
		const model = { ...baseModel, api: "openai-completions" } as const;
		const response = await streamSimple(
			model,
			{
				messages: [{ role: "user", content: "Use reasoning.", timestamp: Date.now() }],
			},
			{ apiKey: "test" },
		).result();

		expect(response.content).toEqual([
			{
				type: "thinking",
				thinking: "think",
				thinkingSignature: "reasoning_content",
			},
		]);
	});

	it("keeps non-OpenCode Go reasoning deltas on the original reasoning field", async () => {
		mockState.chunks = [
			{
				id: "chatcmpl-reasoning",
				choices: [{ delta: { reasoning: "think" }, finish_reason: "stop" }],
			},
		];

		const { compat: _compat, ...baseModel } = getModel("openai", "gpt-4o-mini")!;
		const model = { ...baseModel, api: "openai-completions" } as const;
		const response = await streamSimple(
			model,
			{
				messages: [{ role: "user", content: "Use reasoning.", timestamp: Date.now() }],
			},
			{ apiKey: "test" },
		).result();

		expect(response.content).toEqual([
			{
				type: "thinking",
				thinking: "think",
				thinkingSignature: "reasoning",
			},
		]);
	});

	it("replays OpenCode Go reasoning thinking blocks as reasoning_content", () => {
		const { compat: _compat, ...baseModel } = getModel("opencode-go", "kimi-k2.6")!;
		const model = { ...baseModel, api: "openai-completions" } as Model<"openai-completions">;
		const messages = convertMessages(
			model,
			{
				messages: [
					{
						role: "assistant",
						api: "openai-completions",
						provider: "opencode-go",
						model: "kimi-k2.6",
						content: [
							{ type: "thinking", thinking: "think", thinkingSignature: "reasoning" },
							{ type: "toolCall", id: "call_1", name: "read", arguments: { path: "README.md" } },
						],
						usage: {
							input: 0,
							output: 0,
							cacheRead: 0,
							cacheWrite: 0,
							totalTokens: 0,
							cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
						},
						stopReason: "stop",
						timestamp: Date.now(),
					},
				],
			},
			{
				...model.compat,
				supportsStore: false,
				supportsDeveloperRole: false,
				supportsReasoningEffort: true,
				supportsUsageInStreaming: true,
				maxTokensField: "max_completion_tokens",
				requiresToolResultName: false,
				requiresAssistantAfterToolResult: false,
				requiresThinkingAsText: false,
				requiresReasoningContentOnAssistantMessages: false,
				thinkingFormat: "openai",
				openRouterRouting: {},
				vercelGatewayRouting: {},
				zaiToolStream: false,
				supportsStrictMode: true,
				sendSessionAffinityHeaders: false,
				supportsLongCacheRetention: true,
			},
		);

		expect(messages[0]).toMatchObject({ role: "assistant", reasoning_content: "think" });
		expect(messages[0]).not.toHaveProperty("reasoning");
	});

	it("sends thinking disabled for OpenCode Go Kimi K2.6 when thinking is off", async () => {
		const model = getModel("opencode-go", "kimi-k2.6")!;
		let payload: unknown;

		await streamSimple(
			model,
			{
				messages: [{ role: "user", content: "Hi", timestamp: Date.now() }],
			},
			{
				apiKey: "test",
				onPayload: (params: unknown) => {
					payload = params;
				},
			},
		).result();

		const params = (payload ?? mockState.lastParams) as { thinking?: unknown; reasoning_effort?: string };
		expect(params.thinking).toEqual({ type: "disabled" });
		expect(params.reasoning_effort).toBeUndefined();
	});

	it("sends thinking enabled for OpenCode Go Kimi K2.6 when thinking is enabled", async () => {
		const model = getModel("opencode-go", "kimi-k2.6")!;
		let payload: unknown;

		await streamSimple(
			model,
			{
				messages: [{ role: "user", content: "Hi", timestamp: Date.now() }],
			},
			{
				apiKey: "test",
				reasoning: "high",
				onPayload: (params: unknown) => {
					payload = params;
				},
			},
		).result();

		const params = (payload ?? mockState.lastParams) as { thinking?: unknown; reasoning_effort?: string };
		expect(params.thinking).toEqual({ type: "enabled" });
		expect(params.reasoning_effort).toBeUndefined();
	});

	it("omits disabled thinking for Moonshot Kimi K2.7 Code models", async () => {
		const cases = [getModel("moonshotai", "kimi-k2.7-code"), getModel("moonshotai-cn", "kimi-k2.7-code")];

		for (const model of cases) {
			expect(model).toBeDefined();
			let payload: unknown;

			await streamSimple(
				model!,
				{
					messages: [{ role: "user", content: "Hi", timestamp: Date.now() }],
				},
				{
					apiKey: "test",
					onPayload: (params: unknown) => {
						payload = params;
					},
				},
			).result();

			const params = (payload ?? mockState.lastParams) as { thinking?: unknown; reasoning_effort?: string };
			expect(params.thinking).toBeUndefined();
			expect(params.reasoning_effort).toBeUndefined();
		}
	});

	it("keeps disabled thinking for Moonshot Kimi K2.6 when thinking is off", async () => {
		const model = getModel("moonshotai-cn", "kimi-k2.6")!;
		let payload: unknown;

		await streamSimple(
			model,
			{
				messages: [{ role: "user", content: "Hi", timestamp: Date.now() }],
			},
			{
				apiKey: "test",
				onPayload: (params: unknown) => {
					payload = params;
				},
			},
		).result();

		const params = (payload ?? mockState.lastParams) as { thinking?: unknown; reasoning_effort?: string };
		expect(params.thinking).toEqual({ type: "disabled" });
		expect(params.reasoning_effort).toBeUndefined();
	});

	it("sends max_tokens for OpenCode completions models", async () => {
		const cases = [getModel("opencode-go", "kimi-k2.6")!, getModel("opencode", "grok-build-0.1")!] as const;

		for (const model of cases) {
			let payload: unknown;
			expect(model.compat?.maxTokensField).toBe("max_tokens");

			await streamSimple(
				model,
				{
					messages: [{ role: "user", content: "Hi", timestamp: Date.now() }],
				},
				{
					apiKey: "test",
					maxTokens: 123,
					onPayload: (params: unknown) => {
						payload = params;
					},
				},
			).result();

			const params = (payload ?? mockState.lastParams) as { max_tokens?: number; max_completion_tokens?: number };
			expect(params.max_tokens).toBe(123);
			expect(params.max_completion_tokens).toBeUndefined();
		}
	});

	it("omits reasoning effort for OpenCode Grok Build", async () => {
		const model = getModel("opencode", "grok-build-0.1")!;
		let payload: unknown;

		await streamSimple(
			model,
			{
				messages: [{ role: "user", content: "Hi", timestamp: Date.now() }],
			},
			{
				apiKey: "test",
				reasoning: "high",
				onPayload: (params: unknown) => {
					payload = params;
				},
			},
		).result();

		const params = (payload ?? mockState.lastParams) as { reasoning_effort?: string };
		expect(params.reasoning_effort).toBeUndefined();
	});

	it("does not double-count reasoning tokens in completion usage", async () => {
		mockState.chunks = [
			{
				id: "chatcmpl-reasoning-usage",
				choices: [{ delta: {}, finish_reason: "stop" }],
				usage: {
					prompt_tokens: 10,
					completion_tokens: 33,
					prompt_tokens_details: { cached_tokens: 0 },
					completion_tokens_details: { reasoning_tokens: 21 },
				},
			},
		];

		const { compat: _compat, ...baseModel } = getModel("openai", "gpt-4o-mini")!;
		const model = { ...baseModel, api: "openai-completions" } as const;
		const response = await streamSimple(
			model,
			{
				messages: [
					{
						role: "user",
						content: "Use reasoning.",
						timestamp: Date.now(),
					},
				],
			},
			{ apiKey: "test" },
		).result();

		expect(response.usage.input).toBe(10);
		expect(response.usage.output).toBe(33);
		expect(response.usage.totalTokens).toBe(43);
	});

	it("preserves prompt_tokens_details cache read/write fields from chunk usage", async () => {
		mockState.chunks = [
			{
				id: "chatcmpl-cache-write",
				choices: [{ delta: { content: "OK" }, finish_reason: null }],
			},
			{
				id: "chatcmpl-cache-write",
				choices: [{ delta: {}, finish_reason: "stop" }],
				usage: {
					prompt_tokens: 100,
					completion_tokens: 5,
					prompt_tokens_details: { cached_tokens: 50, cache_write_tokens: 30 },
					completion_tokens_details: { reasoning_tokens: 0 },
				},
			},
		];

		const { compat: _compat, ...baseModel } = getModel("openai", "gpt-4o-mini")!;
		const model = { ...baseModel, api: "openai-completions" } as const;
		const response = await streamSimple(
			model,
			{
				messages: [
					{
						role: "user",
						content: "Reply with exactly OK",
						timestamp: Date.now(),
					},
				],
			},
			{ apiKey: "test" },
		).result();

		// cached_tokens is documented as cache reads; cache_write_tokens is separate.
		expect(response.usage.input).toBe(20);
		expect(response.usage.cacheRead).toBe(50);
		expect(response.usage.cacheWrite).toBe(30);
		expect(response.usage.totalTokens).toBe(105);
	});

	it("preserves prompt_tokens_details cache read/write fields from choice usage fallback", async () => {
		mockState.chunks = [
			{
				id: "chatcmpl-cache-write-choice",
				choices: [{ delta: { content: "OK" }, finish_reason: null }],
			},
			{
				id: "chatcmpl-cache-write-choice",
				choices: [
					{
						delta: {},
						finish_reason: "stop",
						usage: {
							prompt_tokens: 100,
							completion_tokens: 5,
							prompt_tokens_details: { cached_tokens: 50, cache_write_tokens: 30 },
							completion_tokens_details: { reasoning_tokens: 0 },
						},
					},
				],
			},
		];

		const { compat: _compat, ...baseModel } = getModel("openai", "gpt-4o-mini")!;
		const model = { ...baseModel, api: "openai-completions" } as const;
		const response = await streamSimple(
			model,
			{
				messages: [
					{
						role: "user",
						content: "Reply with exactly OK",
						timestamp: Date.now(),
					},
				],
			},
			{ apiKey: "test" },
		).result();

		// cached_tokens is documented as cache reads; cache_write_tokens is separate.
		expect(response.usage.input).toBe(20);
		expect(response.usage.cacheRead).toBe(50);
		expect(response.usage.cacheWrite).toBe(30);
		expect(response.usage.totalTokens).toBe(105);
	});

	it("uses OpenRouter reasoning object instead of reasoning_effort", async () => {
		const model = getModel("openrouter", "deepseek/deepseek-r1")!;
		let payload: unknown;

		await streamSimple(
			model,
			{
				messages: [
					{
						role: "user",
						content: "Hi",
						timestamp: Date.now(),
					},
				],
			},
			{
				apiKey: "test",
				reasoning: "high",
				onPayload: (params: unknown) => {
					payload = params;
				},
			},
		).result();

		const params = (payload ?? mockState.lastParams) as {
			reasoning?: { effort?: string };
			reasoning_effort?: string;
		};
		expect(params.reasoning).toEqual({ effort: "high" });
		expect(params.reasoning_effort).toBeUndefined();
	});

	it("uses Ant Ling compatibility metadata", async () => {
		const model = getModel("ant-ling", "Ring-2.6-1T")!;
		let payload: unknown;

		expect(model.compat).toMatchObject({
			supportsStore: false,
			supportsDeveloperRole: false,
			supportsReasoningEffort: false,
			maxTokensField: "max_tokens",
			thinkingFormat: "ant-ling",
			supportsLongCacheRetention: false,
		});
		expect(model.compat?.supportsStrictMode).toBeUndefined();
		expect(model.compat?.requiresReasoningContentOnAssistantMessages).toBeUndefined();

		await streamSimple(
			model,
			{
				systemPrompt: "Follow instructions.",
				messages: [{ role: "user", content: "Hi", timestamp: Date.now() }],
			},
			{
				apiKey: "test",
				maxTokens: 123,
				reasoning: "high",
				cacheRetention: "long",
				sessionId: "ant-ling-session",
				onPayload: (params: unknown) => {
					payload = params;
				},
			},
		).result();

		const params = (payload ?? mockState.lastParams) as {
			max_tokens?: number;
			max_completion_tokens?: number;
			messages?: Array<{ role?: string }>;
			reasoning?: { effort?: string };
			reasoning_effort?: string;
			store?: boolean;
			prompt_cache_key?: string;
			prompt_cache_retention?: string;
		};
		expect(params.max_tokens).toBe(123);
		expect(params.max_completion_tokens).toBeUndefined();
		expect(params.messages?.[0]?.role).toBe("system");
		expect(params.reasoning).toEqual({ effort: "high" });
		expect(params.reasoning_effort).toBeUndefined();
		expect(params.store).toBeUndefined();
		expect(params.prompt_cache_key).toBeUndefined();
		expect(params.prompt_cache_retention).toBeUndefined();
	});

	it("omits Ant Ling reasoning for unmapped direct reasoning efforts and non-reasoning models", async () => {
		const ring = getModel("ant-ling", "Ring-2.6-1T")!;
		let payload: unknown;

		await stream(
			ring,
			{
				messages: [{ role: "user", content: "Hi", timestamp: Date.now() }],
			},
			{
				apiKey: "test",
				reasoningEffort: "medium",
				onPayload: (params: unknown) => {
					payload = params;
				},
			},
		).result();

		expect((payload ?? mockState.lastParams) as { reasoning?: unknown }).not.toHaveProperty("reasoning");

		const ling = getModel("ant-ling", "Ling-2.6-flash")!;
		await streamSimple(
			ling,
			{
				messages: [{ role: "user", content: "Hi", timestamp: Date.now() }],
			},
			{
				apiKey: "test",
				reasoning: "high",
				onPayload: (params: unknown) => {
					payload = params;
				},
			},
		).result();

		expect((payload ?? mockState.lastParams) as { reasoning?: unknown }).not.toHaveProperty("reasoning");
	});
});
