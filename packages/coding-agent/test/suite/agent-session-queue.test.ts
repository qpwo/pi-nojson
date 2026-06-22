import type { AgentTool } from "@earendil-works/pi-agent-core";
import { fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { afterEach, describe, expect, it } from "vitest";
import { createHarness, getAssistantTexts, getMessageText, getUserTexts, type Harness } from "./harness.ts";

async function createWaitingHarness(
	options: {
		tools?: AgentTool[];
		extensionFactories?: Harness["session"]["extensionRunner"] extends never
			? never
			: Array<(pi: ExtensionAPI) => void>;
	} = {},
): Promise<{
	harness: Harness;
	releaseToolExecution: () => void;
	promptPromise: Promise<void>;
	waitForToolStart: Promise<void>;
}> {
	let releaseToolExecution: (() => void) | undefined;
	const toolRelease = new Promise<void>((resolve) => {
		releaseToolExecution = resolve;
	});
	const waitTool: AgentTool = {
		name: "wait",
		label: "Wait",
		description: "Wait for release",
		parameters: Type.Object({}),
		execute: async () => {
			await toolRelease;
			return {
				content: [{ type: "text", text: "released" }],
				details: {},
			};
		},
	};
	const harness = await createHarness({
		tools: [waitTool, ...(options.tools ?? [])],
		extensionFactories: options.extensionFactories,
	});

	const waitForToolStart = new Promise<void>((resolve) => {
		const unsubscribe = harness.session.subscribe((event) => {
			if (event.type === "tool_execution_start" && event.toolName === "wait") {
				unsubscribe();
				resolve();
			}
		});
	});

	return {
		harness,
		releaseToolExecution: () => releaseToolExecution?.(),
		promptPromise: harness.session.prompt("start"),
		waitForToolStart,
	};
}

describe("AgentSession queue characterization", () => {
	const harnesses: Harness[] = [];

	afterEach(() => {
		while (harnesses.length > 0) {
			harnesses.pop()?.cleanup();
		}
	});

	it("dispatches extension commands immediately when prompted while idle", async () => {
		const commandRuns: string[] = [];
		const harness = await createHarness({
			extensionFactories: [
				(pi) => {
					pi.registerCommand("testcmd", {
						description: "Test command",
						handler: async (args) => {
							commandRuns.push(args);
						},
					});
				},
			],
		});
		harnesses.push(harness);

		await harness.session.prompt("/testcmd hello world");

		expect(commandRuns).toEqual(["hello world"]);
		expect(harness.getPendingResponseCount()).toBe(0);
		expect(harness.session.messages).toEqual([]);
	});

	it("delivers extension-origin steering messages before the next LLM call", async () => {
		let extensionApi: ExtensionAPI | undefined;
		const waiting = await createWaitingHarness({
			extensionFactories: [
				(pi) => {
					extensionApi = pi;
				},
			],
		});
		const { harness, waitForToolStart, promptPromise, releaseToolExecution } = waiting;
		harnesses.push(harness);

		harness.setResponses([
			fauxAssistantMessage(fauxToolCall("wait", {}), { stopReason: "toolUse" }),
			(context) => {
				const sawSteer = context.messages.some(
					(message) => message.role === "user" && getMessageText(message) === "steer now",
				);
				return fauxAssistantMessage(sawSteer ? "saw steer" : "missing steer");
			},
		]);

		await waitForToolStart;
		await new Promise((resolve) => setTimeout(resolve, 0));

		extensionApi?.sendUserMessage("steer now", { deliverAs: "steer" });
		releaseToolExecution();
		await promptPromise;

		expect(getUserTexts(harness)).toEqual(["start", "steer now"]);
		expect(getAssistantTexts(harness)).toContain("saw steer");
	});

	it("delivers follow-up messages only after the current run finishes", async () => {
		const waiting = await createWaitingHarness();
		const { harness, waitForToolStart, promptPromise, releaseToolExecution } = waiting;
		harnesses.push(harness);
		const assistantSeenBeforeFollowUp: string[] = [];

		harness.setResponses([
			fauxAssistantMessage(fauxToolCall("wait", {}), { stopReason: "toolUse" }),
			(context) => {
				assistantSeenBeforeFollowUp.push(
					...context.messages
						.filter((message) => message.role === "assistant")
						.map((message) =>
							message.content
								.filter((part): part is { type: "text"; text: string } => part.type === "text")
								.map((part) => part.text)
								.join("\n"),
						),
				);
				return fauxAssistantMessage("follow-up response");
			},
		]);

		await waitForToolStart;
		await harness.session.followUp("after current run");
		releaseToolExecution();
		await promptPromise;

		expect(getUserTexts(harness)).toEqual(["start", "after current run"]);
		expect(assistantSeenBeforeFollowUp).toContain("");
		expect(getAssistantTexts(harness)).toContain("follow-up response");
	});

	it("delivers multiple steering messages in order in one-at-a-time mode", async () => {
		const waiting = await createWaitingHarness();
		const { harness, waitForToolStart, promptPromise, releaseToolExecution } = waiting;
		harnesses.push(harness);

		harness.setResponses([
			fauxAssistantMessage(fauxToolCall("wait", {}), { stopReason: "toolUse" }),
			fauxAssistantMessage("handled steer 1"),
			fauxAssistantMessage("handled steer 2"),
		]);

		await waitForToolStart;
		await harness.session.steer("steer 1");
		await harness.session.steer("steer 2");
		releaseToolExecution();
		await promptPromise;

		expect(getUserTexts(harness)).toEqual(["start", "steer 1", "steer 2"]);
		expect(getAssistantTexts(harness)).toEqual(["", "handled steer 1", "handled steer 2"]);
	});

	it("delivers multiple follow-up messages in order in one-at-a-time mode", async () => {
		const waiting = await createWaitingHarness();
		const { harness, waitForToolStart, promptPromise, releaseToolExecution } = waiting;
		harnesses.push(harness);

		harness.setResponses([
			fauxAssistantMessage(fauxToolCall("wait", {}), { stopReason: "toolUse" }),
			fauxAssistantMessage("original turn complete"),
			fauxAssistantMessage("handled follow-up 1"),
			fauxAssistantMessage("handled follow-up 2"),
		]);

		await waitForToolStart;
		await harness.session.followUp("follow-up 1");
		await harness.session.followUp("follow-up 2");
		releaseToolExecution();
		await promptPromise;

		expect(getUserTexts(harness)).toEqual(["start", "follow-up 1", "follow-up 2"]);
		expect(getAssistantTexts(harness)).toEqual([
			"",
			"original turn complete",
			"handled follow-up 1",
			"handled follow-up 2",
		]);
	});

	it("delivers all steering messages in one batch in all mode", async () => {
		const waiting = await createWaitingHarness();
		const { harness, waitForToolStart, promptPromise, releaseToolExecution } = waiting;
		harnesses.push(harness);
		harness.session.setSteeringMode("all");
		let batchedUserMessages: string[] = [];

		harness.setResponses([
			fauxAssistantMessage(fauxToolCall("wait", {}), { stopReason: "toolUse" }),
			(context) => {
				batchedUserMessages = context.messages
					.filter((message) => message.role === "user")
					.map((message) => getMessageText(message));
				return fauxAssistantMessage("batched steer response");
			},
		]);

		await waitForToolStart;
		await harness.session.steer("steer 1");
		await harness.session.steer("steer 2");
		releaseToolExecution();
		await promptPromise;

		expect(batchedUserMessages).toEqual(["start", "steer 1", "steer 2"]);
		expect(getAssistantTexts(harness)).toEqual(["", "batched steer response"]);
	});

	it("delivers all follow-up messages in one batch in all mode", async () => {
		const waiting = await createWaitingHarness();
		const { harness, waitForToolStart, promptPromise, releaseToolExecution } = waiting;
		harnesses.push(harness);
		harness.session.setFollowUpMode("all");
		let batchedUserMessages: string[] = [];

		harness.setResponses([
			fauxAssistantMessage(fauxToolCall("wait", {}), { stopReason: "toolUse" }),
			fauxAssistantMessage("original turn complete"),
			(context) => {
				batchedUserMessages = context.messages
					.filter((message) => message.role === "user")
					.map((message) => getMessageText(message));
				return fauxAssistantMessage("batched follow-up response");
			},
		]);

		await waitForToolStart;
		await harness.session.followUp("follow-up 1");
		await harness.session.followUp("follow-up 2");
		releaseToolExecution();
		await promptPromise;

		expect(batchedUserMessages).toEqual(["start", "follow-up 1", "follow-up 2"]);
		expect(getAssistantTexts(harness)).toEqual(["", "original turn complete", "batched follow-up response"]);
	});

	it("queues custom messages with deliverAs steer while streaming", async () => {
		const waiting = await createWaitingHarness();
		const { harness, waitForToolStart, promptPromise, releaseToolExecution } = waiting;
		harnesses.push(harness);
		let sawCustomMessage = false;

		harness.setResponses([
			fauxAssistantMessage(fauxToolCall("wait", {}), { stopReason: "toolUse" }),
			(context) => {
				sawCustomMessage = context.messages.some(
					(message) =>
						message.role === "user" &&
						typeof message.content !== "string" &&
						message.content.some((part) => part.type === "text" && part.text === "steer custom"),
				);
				return fauxAssistantMessage("done");
			},
		]);

		await waitForToolStart;
		await harness.session.sendCustomMessage(
			{ customType: "queue-test", content: "steer custom", display: true, details: { value: 1 } },
			{ deliverAs: "steer" },
		);
		releaseToolExecution();
		await promptPromise;

		expect(sawCustomMessage).toBe(true);
		expect(
			harness.session.messages.some((message) => message.role === "custom" && message.customType === "queue-test"),
		).toBe(true);
	});

	it("queues custom messages with deliverAs followUp while streaming", async () => {
		const waiting = await createWaitingHarness();
		const { harness, waitForToolStart, promptPromise, releaseToolExecution } = waiting;
		harnesses.push(harness);
		let sawCustomMessage = false;

		harness.setResponses([
			fauxAssistantMessage(fauxToolCall("wait", {}), { stopReason: "toolUse" }),
			fauxAssistantMessage("original turn complete"),
			(context) => {
				sawCustomMessage = context.messages.some(
					(message) =>
						message.role === "user" &&
						typeof message.content !== "string" &&
						message.content.some((part) => part.type === "text" && part.text === "follow-up custom"),
				);
				return fauxAssistantMessage("done");
			},
		]);

		await waitForToolStart;
		await harness.session.sendCustomMessage(
			{ customType: "queue-test", content: "follow-up custom", display: true, details: { value: 1 } },
			{ deliverAs: "followUp" },
		);
		releaseToolExecution();
		await promptPromise;

		expect(sawCustomMessage).toBe(true);
		expect(
			harness.session.messages.some((message) => message.role === "custom" && message.customType === "queue-test"),
		).toBe(true);
	});

	it("injects nextTurn custom messages into the next prompt", async () => {
		const harness = await createHarness();
		harnesses.push(harness);
		let sawCustomMessage = false;

		await harness.session.sendCustomMessage(
			{ customType: "next-turn", content: "carry this", display: true, details: {} },
			{ deliverAs: "nextTurn" },
		);

		harness.setResponses([
			(context) => {
				sawCustomMessage = context.messages.some(
					(message) =>
						message.role === "user" &&
						typeof message.content !== "string" &&
						message.content.some((part) => part.type === "text" && part.text === "carry this"),
				);
				return fauxAssistantMessage("done");
			},
		]);

		await harness.session.prompt("normal prompt");

		expect(sawCustomMessage).toBe(true);
		expect(harness.session.messages.map((message) => message.role)).toEqual(["user", "custom", "assistant"]);
	});

	it("updates pendingMessageCount and removes queued text before message_start is emitted", async () => {
		const waiting = await createWaitingHarness();
		const { harness, waitForToolStart, promptPromise, releaseToolExecution } = waiting;
		harnesses.push(harness);
		const countsAtQueuedMessageStart: number[] = [];

		harness.setResponses([
			fauxAssistantMessage(fauxToolCall("wait", {}), { stopReason: "toolUse" }),
			fauxAssistantMessage("done"),
		]);

		harness.session.subscribe((event) => {
			if (
				event.type === "message_start" &&
				event.message.role === "user" &&
				getMessageText(event.message) === "queued"
			) {
				countsAtQueuedMessageStart.push(harness.session.pendingMessageCount);
			}
		});

		await waitForToolStart;
		await harness.session.steer("queued");
		expect(harness.session.pendingMessageCount).toBe(1);
		releaseToolExecution();
		await promptPromise;

		expect(countsAtQueuedMessageStart).toEqual([0]);
		expect(harness.session.pendingMessageCount).toBe(0);
	});

	it("throws when queueing an extension command with steer", async () => {
		const harness = await createHarness({
			extensionFactories: [
				(pi) => {
					pi.registerCommand("testcmd", {
						description: "Test command",
						handler: async () => {},
					});
				},
			],
		});
		harnesses.push(harness);

		await expect(harness.session.steer("/testcmd queued")).rejects.toThrow(
			'Extension command "/testcmd" cannot be queued. Use prompt() or execute the command when not streaming.',
		);
	});

	it("throws when queueing an extension command with followUp", async () => {
		const harness = await createHarness({
			extensionFactories: [
				(pi) => {
					pi.registerCommand("testcmd", {
						description: "Test command",
						handler: async () => {},
					});
				},
			],
		});
		harnesses.push(harness);

		await expect(harness.session.followUp("/testcmd queued")).rejects.toThrow(
			'Extension command "/testcmd" cannot be queued. Use prompt() or execute the command when not streaming.',
		);
	});

	it("delivers follow-ups queued during agent_end", async () => {
		let sent = false;
		const harness = await createHarness({
			extensionFactories: [
				(pi: ExtensionAPI) => {
					pi.on("agent_end", async () => {
						if (sent) return;
						sent = true;
						pi.sendUserMessage("conflict report", { deliverAs: "followUp" });
					});
				},
			],
		});
		harnesses.push(harness);

		harness.setResponses([fauxAssistantMessage("reply"), fauxAssistantMessage("follow-up reply")]);

		await harness.session.prompt("hello");
		await harness.session.agent.waitForIdle();

		expect(getUserTexts(harness)).toEqual(["hello", "conflict report"]);
	});
	it("auto follow-up sends what at stop up to the configured limit", async () => {
		const harness = await createHarness({
			autoFollowUpOnStop: { text: "what", maxConsecutive: 3 },
		});
		harnesses.push(harness);

		harness.setResponses([
			fauxAssistantMessage("idle 0"),
			fauxAssistantMessage("idle 1"),
			fauxAssistantMessage("idle 2"),
			fauxAssistantMessage("idle 3"),
		]);

		await harness.session.prompt("start");

		expect(getUserTexts(harness)).toEqual(["start", "what", "what", "what"]);
		expect(getAssistantTexts(harness)).toEqual(["idle 0", "idle 1", "idle 2", "idle 3"]);
	});
	it("auto follow-up can start from an empty transcript", async () => {
		const harness = await createHarness({
			autoFollowUpOnStop: { text: "what", maxConsecutive: 3 },
		});
		harnesses.push(harness);

		harness.setResponses([
			fauxAssistantMessage("empty 0"),
			fauxAssistantMessage("empty 1"),
			fauxAssistantMessage("empty 2"),
		]);

		await harness.session.runAutoFollowUpOnStop();

		expect(getUserTexts(harness)).toEqual(["what", "what", "what"]);
		expect(getAssistantTexts(harness)).toEqual(["empty 0", "empty 1", "empty 2"]);
	});

	it("auto follow-up sends what after an unretryable assistant error", async () => {
		const harness = await createHarness({
			autoFollowUpOnStop: { text: "what", maxConsecutive: 1 },
		});
		harnesses.push(harness);

		harness.setResponses([
			{ ...fauxAssistantMessage("failed"), stopReason: "error", errorMessage: "fatal validation failed" },
			fauxAssistantMessage("recovered"),
		]);

		await harness.session.prompt("start");

		expect(getUserTexts(harness)).toEqual(["start", "what"]);
		expect(getAssistantTexts(harness)).toEqual(["failed", "recovered"]);
	});

	it("auto follow-up waits for explicit agent_end follow-ups", async () => {
		let sent = false;
		const harness = await createHarness({
			autoFollowUpOnStop: { text: "what", maxConsecutive: 1 },
			extensionFactories: [
				(pi: ExtensionAPI) => {
					pi.on("agent_end", async () => {
						if (sent) {
							return;
						}
						sent = true;
						pi.sendUserMessage("conflict report", { deliverAs: "followUp" });
					});
				},
			],
		});
		harnesses.push(harness);

		harness.setResponses([
			fauxAssistantMessage("reply"),
			fauxAssistantMessage("follow-up reply"),
			fauxAssistantMessage("auto reply"),
		]);

		await harness.session.prompt("hello");
		await harness.session.agent.waitForIdle();

		expect(getUserTexts(harness)).toEqual(["hello", "conflict report", "what"]);
		expect(getAssistantTexts(harness)).toEqual(["reply", "follow-up reply", "auto reply"]);
	});

	it("auto follow-up stops after the since-tool-call budget is exhausted", async () => {
		const harness = await createHarness({
			autoFollowUpOnStop: { text: "what", maxSinceToolCall: 3, maxSinceRealUserInput: 6 },
		});
		harnesses.push(harness);

		harness.setResponses([
			fauxAssistantMessage("idle 0"),
			fauxAssistantMessage("idle 1"),
			fauxAssistantMessage("idle 2"),
			fauxAssistantMessage("idle 3"),
			fauxAssistantMessage("should not run"),
		]);

		await harness.session.prompt("start");

		expect(getUserTexts(harness)).toEqual(["start", "what", "what", "what"]);
		expect(getAssistantTexts(harness)).toEqual(["idle 0", "idle 1", "idle 2", "idle 3"]);
		expect(harness.getPendingResponseCount()).toBe(1);
	});

	it("auto follow-up tool calls reset only the since-tool-call budget", async () => {
		const toolCalls: string[] = [];
		const pingTool: AgentTool = {
			name: "ping",
			label: "Ping",
			description: "Record a ping",
			parameters: Type.Object({}),
			execute: async () => {
				toolCalls.push("ping");
				return {
					content: [{ type: "text", text: "pong" }],
					details: {},
				};
			},
		};
		const harness = await createHarness({
			tools: [pingTool],
			autoFollowUpOnStop: { text: "what", maxSinceToolCall: 3, maxSinceRealUserInput: 6 },
		});
		harnesses.push(harness);

		harness.setResponses([
			fauxAssistantMessage("idle 0"),
			fauxAssistantMessage("idle 1"),
			fauxAssistantMessage(fauxToolCall("ping", {}), { stopReason: "toolUse" }),
			fauxAssistantMessage("after tool"),
			fauxAssistantMessage("after reset 1"),
			fauxAssistantMessage("after reset 2"),
			fauxAssistantMessage(fauxToolCall("ping", {}), { stopReason: "toolUse" }),
			fauxAssistantMessage("after second tool"),
			fauxAssistantMessage("after sixth what"),
			fauxAssistantMessage("should not run"),
		]);

		await harness.session.prompt("start");

		expect(toolCalls).toEqual(["ping", "ping"]);
		expect(getUserTexts(harness)).toEqual(["start", "what", "what", "what", "what", "what", "what"]);
		expect(getAssistantTexts(harness)).toEqual([
			"idle 0",
			"idle 1",
			"",
			"after tool",
			"after reset 1",
			"after reset 2",
			"",
			"after second tool",
			"after sixth what",
		]);
		expect(harness.getPendingResponseCount()).toBe(1);
	});
	it("auto follow-up real-user-input budget resets on real user input", async () => {
		const harness = await createHarness({
			autoFollowUpOnStop: { text: "what", maxSinceToolCall: 10, maxSinceRealUserInput: 1 },
		});
		harnesses.push(harness);

		harness.setResponses([
			fauxAssistantMessage("first reply"),
			fauxAssistantMessage("first what reply"),
			fauxAssistantMessage("second reply"),
			fauxAssistantMessage("second what reply"),
		]);

		await harness.session.prompt("start");
		await harness.session.prompt("next");

		expect(getUserTexts(harness)).toEqual(["start", "what", "next", "what"]);
		expect(getAssistantTexts(harness)).toEqual([
			"first reply",
			"first what reply",
			"second reply",
			"second what reply",
		]);
	});
});
