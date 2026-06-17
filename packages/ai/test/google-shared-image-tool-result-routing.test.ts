import { describe, expect, it } from "vitest";
import { convertMessages } from "../src/providers/google-shared.ts";
import type { Context, Model } from "../src/types.ts";

function makeModel<TApi extends "google-generative-ai">(
	api: TApi,
	provider: Model<TApi>["provider"],
	id: string,
): Model<TApi> {
	return {
		id,
		name: id,
		api,
		provider,
		baseUrl: "https://example.com",
		reasoning: true,
		input: ["text", "image"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 128000,
		maxTokens: 8192,
	};
}

function makeContext(model: { api: string; provider: string; id: string }): Context {
	const now = Date.now();
	return {
		messages: [
			{ role: "user", content: "read the files", timestamp: now },
			{
				role: "assistant",
				content: [
					{ type: "toolCall", id: "call_a", name: "read", arguments: { path: "a.txt" } },
					{ type: "toolCall", id: "call_img", name: "read", arguments: { path: "image.png" } },
					{ type: "toolCall", id: "call_b", name: "read", arguments: { path: "b.txt" } },
				],
				api: model.api,
				provider: model.provider,
				model: model.id,
				usage: {
					input: 0,
					output: 0,
					cacheRead: 0,
					cacheWrite: 0,
					totalTokens: 0,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
				},
				stopReason: "toolUse",
				timestamp: now,
			},
			{
				role: "toolResult",
				toolCallId: "call_a",
				toolName: "read",
				content: [{ type: "text", text: "alpha text" }],
				isError: false,
				timestamp: now,
			},
			{
				role: "toolResult",
				toolCallId: "call_img",
				toolName: "read",
				content: [{ type: "image", data: "abc", mimeType: "image/png" }],
				isError: false,
				timestamp: now,
			},
			{
				role: "toolResult",
				toolCallId: "call_b",
				toolName: "read",
				content: [{ type: "text", text: "beta text" }],
				isError: false,
				timestamp: now,
			},
		],
	};
}

describe("google-shared image tool result routing", () => {
	it("serializes tool results as text-plus-image user content for Gemini 2.x Google API models", () => {
		const model = makeModel("google-generative-ai", "google", "gemini-2.5-flash");
		const contents = convertMessages(model, makeContext(model));

		expect(contents).toHaveLength(3);
		expect(contents[2].role).toBe("user");
		expect(
			contents[2].parts?.filter((part) => typeof part.text === "string" && part.text.includes("<tool_results>")),
		).toHaveLength(3);
		expect(contents[2].parts?.filter((part) => part.inlineData)).toHaveLength(1);
		expect(JSON.stringify(contents[2])).not.toContain("functionResponse");
	});

	it("serializes tool results as text-plus-image user content for Gemini 3 Google API models", () => {
		const model = makeModel("google-generative-ai", "google", "gemini-3-pro-preview");
		const contents = convertMessages(model, makeContext(model));

		expect(contents).toHaveLength(3);
		expect(contents[2].role).toBe("user");
		expect(
			contents[2].parts?.filter((part) => typeof part.text === "string" && part.text.includes("<tool_results>")),
		).toHaveLength(3);
		expect(contents[2].parts?.filter((part) => part.inlineData)).toHaveLength(1);
		expect(JSON.stringify(contents[2])).not.toContain("functionResponse");
	});
});
