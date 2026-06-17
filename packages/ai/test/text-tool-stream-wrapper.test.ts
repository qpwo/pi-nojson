import { describe, expect, it } from "vitest";
import type { AssistantMessage, AssistantMessageEvent, Context } from "../src/types.ts";
import { AssistantMessageEventStream } from "../src/utils/event-stream.ts";
import { wrapTextToolStream } from "../src/utils/text-tools.ts";

async function collect(stream: AssistantMessageEventStream): Promise<AssistantMessageEvent[]> {
	const events: AssistantMessageEvent[] = [];
	for await (const event of stream) {
		events.push(event);
	}
	return events;
}

function messageWithText(text: string): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text }],
		api: "faux",
		provider: "faux",
		model: "faux",
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "stop",
		timestamp: 1,
	};
}

describe("text tool stream wrapper", () => {
	it("turns final plain-text tool blocks into tool-call events and message blocks", async () => {
		const raw = new AssistantMessageEventStream();
		const context: Context = {
			messages: [{ role: "user", content: "call lookup", timestamp: 1 }],
			tools: [
				{
					name: "lookup",
					description: "lookup",
					parameters: { type: "object", properties: { value: { type: "number" } } },
				},
			],
		};
		const message = messageWithText('<lookup>\n{"value":7}\n</lookup>');
		const eventsPromise = collect(wrapTextToolStream(raw, context));

		raw.push({ type: "start", partial: message });
		raw.push({ type: "done", reason: "stop", message });

		const events = await eventsPromise;
		expect(events.map((event) => event.type)).toEqual([
			"start",
			"toolcall_start",
			"toolcall_delta",
			"toolcall_end",
			"done",
		]);
		const done = events.at(-1);
		expect(done?.type).toBe("done");
		if (done?.type !== "done") throw new Error("expected done");
		expect(done.reason).toBe("toolUse");
		expect(done.message.stopReason).toBe("toolUse");
		expect(done.message.content).toEqual([
			{
				type: "toolCall",
				id: expect.any(String),
				name: "lookup",
				arguments: { value: 7 },
				textToolRaw: '<lookup>\n{"value":7}\n</lookup>',
			},
		]);
	});
});
