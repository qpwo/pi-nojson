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

function tag(name: string, body = "", attrs = ""): string {
	const lt = String.fromCharCode(60);
	const gt = String.fromCharCode(62);
	return `${lt}${name}${attrs}${gt}${body}${lt}/${name}${gt}`;
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
		const lookupText = tag("lookup", '\n{"value":7}\n');
		const message = messageWithText(lookupText);
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
				textToolRaw: lookupText,
			},
		]);
	});

	it("buffers streamed text tool tags instead of leaking raw tags as text events", async () => {
		const raw = new AssistantMessageEventStream();
		const context: Context = {
			messages: [{ role: "user", content: "inspect", timestamp: 1 }],
			tools: [
				{
					name: "bash",
					description: "bash",
					parameters: { type: "object", properties: { command: { type: "string" } } },
				},
				{
					name: "read",
					description: "read",
					parameters: { type: "object", properties: { path: { type: "string" } } },
				},
			],
		};
		const text =
			tag("bash", "\npwd && ls -la\n", ' timeout="20"') +
			tag("read", "", ' path="package.json" offset="1" limit="200"');
		const message = messageWithText(text);
		const eventsPromise = collect(wrapTextToolStream(raw, context));

		raw.push({ type: "start", partial: messageWithText("") });
		raw.push({ type: "text_start", contentIndex: 0, partial: messageWithText("") });
		raw.push({ type: "text_delta", contentIndex: 0, delta: text, partial: message });
		raw.push({ type: "text_end", contentIndex: 0, content: text, partial: message });
		raw.push({ type: "done", reason: "stop", message });

		const events = await eventsPromise;
		expect(events.map((event) => event.type)).toEqual([
			"start",
			"toolcall_start",
			"toolcall_delta",
			"toolcall_end",
			"toolcall_start",
			"toolcall_delta",
			"toolcall_end",
			"done",
		]);
		for (const event of events) {
			if (event.type === "text_delta") expect(event.delta).not.toContain(String.fromCharCode(60));
			if (event.type === "text_end") expect(event.content).not.toContain(String.fromCharCode(60));
		}

		const done = events.at(-1);
		expect(done?.type).toBe("done");
		if (done?.type !== "done") throw new Error("expected done");
		expect(done.message.content).toMatchObject([
			{ type: "toolCall", name: "bash", arguments: { command: "pwd && ls -la", timeout: 20 } },
			{ type: "toolCall", name: "read", arguments: { path: "package.json", offset: 1, limit: 200 } },
		]);
	});
});
