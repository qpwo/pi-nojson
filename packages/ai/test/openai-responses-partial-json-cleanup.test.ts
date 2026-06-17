import type { ResponseStreamEvent } from "openai/resources/responses/responses.js";
import { describe, expect, it, vi } from "vitest";
import { processResponsesStream } from "../src/providers/openai-responses-shared.ts";
import type { AssistantMessage, Model } from "../src/types.ts";
import { AssistantMessageEventStream } from "../src/utils/event-stream.ts";
import { attachTextToolCalls } from "../src/utils/text-tools.ts";

function createOutput(model: Model<"openai-responses">): AssistantMessage {
	return {
		role: "assistant",
		content: [],
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
		stopReason: "stop",
		timestamp: Date.now(),
	};
}

async function* createTextToolEvents(): AsyncIterable<ResponseStreamEvent> {
	yield {
		type: "response.output_item.added",
		item: {
			type: "message",
			id: "msg_test",
			role: "assistant",
			content: [],
			status: "in_progress",
		},
	} as ResponseStreamEvent;
	yield {
		type: "response.content_part.added",
		item_id: "msg_test",
		output_index: 0,
		content_index: 0,
		part: { type: "output_text", text: "", annotations: [] },
	} as ResponseStreamEvent;
	yield {
		type: "response.output_text.delta",
		item_id: "msg_test",
		output_index: 0,
		content_index: 0,
		delta: "<replace_text>\npath=README.md\ncontent=updated\n</replace_text>",
	} as ResponseStreamEvent;
	yield {
		type: "response.output_item.done",
		item: {
			type: "message",
			id: "msg_test",
			role: "assistant",
			content: [
				{
					type: "output_text",
					text: "<replace_text>\npath=README.md\ncontent=updated\n</replace_text>",
					annotations: [],
				},
			],
			status: "completed",
		},
	} as ResponseStreamEvent;
}

describe("openai responses text tool parsing", () => {
	it("parses text tool blocks from streamed output without native function_call items", async () => {
		const model: Model<"openai-responses"> = {
			id: "gpt-5-mini",
			name: "GPT-5 Mini",
			api: "openai-responses",
			provider: "openai",
			baseUrl: "https://api.openai.com/v1",
			reasoning: true,
			input: ["text"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 400000,
			maxTokens: 128000,
		};
		const output = createOutput(model);
		const stream = new AssistantMessageEventStream();
		const pushSpy = vi.spyOn(stream, "push");

		await processResponsesStream(createTextToolEvents(), output, stream, model);
		attachTextToolCalls({ tools: [{ name: "replace_text", description: "Replace text", parameters: {} }] }, output);

		expect(output.content).toHaveLength(1);
		const persistedToolCall = output.content[0];
		expect(persistedToolCall?.type).toBe("toolCall");
		if (!persistedToolCall || persistedToolCall.type !== "toolCall") {
			throw new Error("Expected toolCall block");
		}
		expect(persistedToolCall.arguments).toEqual({ path: "README.md", content: "updated" });
		expect(JSON.stringify(pushSpy.mock.calls)).not.toContain("function_call");
	});
});
