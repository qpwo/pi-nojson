import { describe, expect, it, vi } from "vitest";
import { InteractiveMode } from "../src/modes/interactive/interactive-mode.ts";

type SubmitContext = {
	defaultEditor: { onSubmit?: (text: string) => void };
	editor: {
		addToHistory?: (text: string) => void;
		setText: (text: string) => void;
	};
	session: {
		isCompacting: boolean;
		isStreaming: boolean;
		isBashRunning: boolean;
		prompt: (text: string, options?: unknown) => Promise<void>;
	};
	flushPendingBashComponents: () => void;
	onInputCallback?: (text: string) => void;
	pendingUserInputs: string[];
};

type InputContext = {
	onInputCallback?: (text: string) => void;
	pendingUserInputs: string[];
};

type StartupPromptContext = {
	options: {
		initialMessage?: string;
		initialImages?: unknown[];
		initialMessages?: string[];
	};
	session: {
		prompt: ReturnType<typeof vi.fn<(text: string, options?: unknown) => Promise<void>>>;
		runAutoFollowUpOnStop: ReturnType<typeof vi.fn<() => Promise<boolean>>>;
	};
	showError: ReturnType<typeof vi.fn<(message: string) => void>>;
	pendingUserInputs: string[];
};

type InteractiveModePrivate = {
	setupEditorSubmitHandler(this: SubmitContext): void;
	getUserInput(this: InputContext): Promise<string>;
	processStartupPrompts(this: StartupPromptContext): Promise<void>;
};

const interactiveModePrototype = InteractiveMode.prototype as unknown as InteractiveModePrivate;

function createSubmitContext(): SubmitContext {
	return {
		defaultEditor: {},
		editor: {
			addToHistory: vi.fn(),
			setText: vi.fn(),
		},
		session: {
			isCompacting: false,
			isStreaming: false,
			isBashRunning: false,
			prompt: vi.fn(async () => {}),
		},
		flushPendingBashComponents: vi.fn(),
		pendingUserInputs: [],
	};
}

function createStartupPromptContext(options: StartupPromptContext["options"] = {}): StartupPromptContext {
	return {
		options,
		session: {
			prompt: vi.fn(async () => {}),
			runAutoFollowUpOnStop: vi.fn(async () => true),
		},
		showError: vi.fn(),
		pendingUserInputs: [],
	};
}

describe("InteractiveMode startup input", () => {
	it("queues a normal prompt submitted before the input callback is installed", async () => {
		const context = createSubmitContext();
		interactiveModePrototype.setupEditorSubmitHandler.call(context);

		await context.defaultEditor.onSubmit?.(" early prompt ");

		expect(context.pendingUserInputs).toEqual(["early prompt"]);
		expect(context.flushPendingBashComponents).toHaveBeenCalledTimes(1);
		expect(context.editor.addToHistory).toHaveBeenCalledWith("early prompt");
	});

	it("returns queued startup input before installing a new input callback", async () => {
		const context: InputContext = {
			pendingUserInputs: ["queued prompt"],
		};

		await expect(interactiveModePrototype.getUserInput.call(context)).resolves.toBe("queued prompt");
		expect(context.onInputCallback).toBeUndefined();
		expect(context.pendingUserInputs).toEqual([]);
	});

	it("starts auto follow-up when no startup prompt exists", async () => {
		const context = createStartupPromptContext();

		await interactiveModePrototype.processStartupPrompts.call(context);

		expect(context.session.prompt).not.toHaveBeenCalled();
		expect(context.session.runAutoFollowUpOnStop).toHaveBeenCalledTimes(1);
		expect(context.showError).not.toHaveBeenCalled();
	});

	it("does not start auto follow-up after configured startup prompts", async () => {
		const context = createStartupPromptContext({
			initialMessage: "hello",
			initialImages: [{ type: "image", data: "abc" }],
			initialMessages: ["again"],
		});

		await interactiveModePrototype.processStartupPrompts.call(context);

		expect(context.session.prompt).toHaveBeenCalledTimes(2);
		expect(context.session.prompt).toHaveBeenNthCalledWith(1, "hello", { images: [{ type: "image", data: "abc" }] });
		expect(context.session.prompt).toHaveBeenNthCalledWith(2, "again");
		expect(context.session.runAutoFollowUpOnStop).not.toHaveBeenCalled();
		expect(context.showError).not.toHaveBeenCalled();
	});

	it("does not start auto follow-up before queued startup input", async () => {
		const context = createStartupPromptContext();
		context.pendingUserInputs.push("typed early");

		await interactiveModePrototype.processStartupPrompts.call(context);

		expect(context.session.prompt).not.toHaveBeenCalled();
		expect(context.session.runAutoFollowUpOnStop).not.toHaveBeenCalled();
		expect(context.pendingUserInputs).toEqual(["typed early"]);
		expect(context.showError).not.toHaveBeenCalled();
	});
});
