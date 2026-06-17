import type {
	AssistantMessage,
	AssistantMessageEvent,
	Context,
	ImageContent,
	TextContent,
	Tool,
	ToolCall,
	ToolResultMessage,
} from "../types.ts";
import { AssistantMessageEventStream } from "./event-stream.ts";

type TextToolCall = ToolCall & { textToolRaw?: string };
type ParsedToolBlock = {
	name: string;
	attrs: Record<string, unknown>;
	body: string;
	raw: string;
	start: number;
	end: number;
	parseError?: string;
};

type ToolLiteralRead = { ok: true; value: unknown; index: number } | { ok: false; index: number };

type TextToolSpec = {
	name: string;
	description: string;
	parameters?: unknown;
};

const QUOTE_TOOL_NAME = "quote";
const ALL_TOOLS = ["bash", "read", "write", "edit", "grep", "find", "ls", "stop"];

const BUILT_IN_TOOL_EXAMPLES: Record<string, string> = {
	bash: `<bash timeout="20">
fd -e js
</bash>`,
	read: `<read path="singlefilepi.js" offset="1" limit="80"></read>`,
	write: '<write path="foo.txt">\nfull file content\n</write>',
	edit: `<edit path="foo.txt">
<old>
exact old text
</old>
<new>
replacement text
</new>
</edit>`,
	grep: `<grep pattern="needle" path="." literal="true" context="2" limit="50"></grep>`,
	find: `<find pattern="*.js" path="." limit="1000"></find>`,
	ls: `<ls path="." limit="500"></ls>`,
	stop: "<stop></stop>",
};

export function textToolProtocol(selectedTools: (Tool | string)[]): string {
	if (!selectedTools.length) return "(none)";

	const tools = selectedTools.map(normalizeTextToolSpec);
	const names = tools.map(textToolName);
	const customTools = tools.filter(isCustomTextToolSpec);
	const customExample = customToolExample(customTools[0]?.name);
	const builtInExamples = builtInToolExamples(new Set(names));

	return `Provider-native JSON/function tools are disabled. Use only plain text tool blocks for tools.

Enabled tool names: ${names.join(", ")}

Custom tool definitions:
${customTools.length ? customTools.map(formatTextToolSpec).join("\n\n") : "(none)"}

Rules:
- When using tools, output only complete tool blocks and no final answer yet.
- Do not wrap tool blocks in markdown fences.
- For custom tools, put key=value argument lines inside the tool tag. One compact object literal is also accepted for compatibility.
- Use numbers as numbers, booleans as true/false, and arrays/objects as compact literal values when needed.
- After receiving <tool_results>, continue normally or issue more tool blocks.
- Do not invent tool results.
- Never use <tool_code> or print(tool_name(...)); use the exact tag-style tool blocks below.
- Tool blocks are executable; do not mention these exact tag forms unless you intend to run them.
- <quote>...</quote> is always available as inert display text, not an executable tool.
- To mention literal tool syntax without running it, put the entire example inside <quote>...</quote>.
- Never write executable-looking tool tags in prose, markdown, or code spans outside <quote>.

Custom tool call example:
${customExample}

Built-in tool examples (ONLY use enabled tool names; NEVER use key=value lines for built-ins):
${builtInExamples || "(none)"}`;
}

export function wrapTextToolStream(
	eventStream: AssistantMessageEventStream,
	context: Context,
): AssistantMessageEventStream {
	const wrappedStream = new AssistantMessageEventStream();
	const bufferProviderText = !!context.tools?.length;
	const bufferedEvents: AssistantMessageEvent[] = [];
	let sawNativeToolEvent = false;
	let closed = false;

	void (async function forwardTextToolStream() {
		try {
			for await (const event of eventStream) {
				if (isToolEvent(event)) sawNativeToolEvent = true;

				if (event.type === "error") {
					wrappedStream.push(event);
					wrappedStream.end();
					closed = true;
					return;
				}

				if (event.type !== "done") {
					if (bufferProviderText) bufferedEvents.push(event);
					else wrappedStream.push(event);
					continue;
				}

				attachTextToolCalls(context, event.message);
				const unwrappedQuoteText = unwrapAssistantQuoteBlocks(event.message);
				const doneEvent = {
					...event,
					reason: event.message.stopReason === "toolUse" ? "toolUse" : event.reason,
				};

				if (
					bufferProviderText &&
					!sawNativeToolEvent &&
					(event.message.stopReason === "toolUse" || unwrappedQuoteText)
				) {
					pushSyntheticAssistantContentEvents(wrappedStream, event.message);
				} else {
					for (const bufferedEvent of bufferedEvents) {
						wrappedStream.push(bufferedEvent);
					}
					if (!sawNativeToolEvent) pushSyntheticTextToolEvents(wrappedStream, event.message);
				}

				wrappedStream.push(doneEvent);
				wrappedStream.end();
				closed = true;
				return;
			}

			if (!closed) wrappedStream.end();
		} catch {
			if (!closed) wrappedStream.end();
		}
	})();

	return wrappedStream;
}

export function contextWithTextToolProtocol(context: Context): Context {
	if (!context.tools?.length) return context;
	const protocol = textToolProtocol(context.tools);
	return {
		...context,
		systemPrompt: context.systemPrompt ? `${context.systemPrompt}\n\n${protocol}` : protocol,
		tools: undefined,
	};
}

export function attachTextToolCalls(context: { tools?: Tool[] }, assistant: AssistantMessage): AssistantMessage {
	if (!context.tools?.length) return assistant;
	if (assistant.stopReason === "error" || assistant.stopReason === "aborted") return assistant;
	if ((assistant.content || []).some(isToolCall)) return assistant;

	const allowed = new Set(context.tools.map(contextToolName));
	const calls = toolCallsFromAssistantText(getText(assistant.content), allowed);
	if (!calls.length) return assistant;

	stripAssistantTextToolBlocks(assistant, allowed);
	assistant.content.push(...calls);
	assistant.stopReason = "toolUse";
	return assistant;
}

export function toolCallsFromAssistantText(text: string, allowed?: Set<string>): ToolCall[] {
	const calls = toolCallsFromTextBlocks(parseToolBlocks(text, allowed), allowed);
	if (calls.length) return calls;
	return toolCallsFromToolCode(text, allowed);
}

export function assistantTextForProvider(content: (TextContent | ImageContent | ToolCall)[] | undefined): string {
	const chunks: string[] = [];
	for (const block of content || []) {
		if (block.type === "text" && block.text) chunks.push(block.text);
		if (block.type === "toolCall") chunks.push(textToolCallForProvider(block));
	}
	return chunks.join("\n");
}

export function textToolCallForProvider(block: ToolCall): string {
	const textBlock = block as TextToolCall;
	const args = objectArgs(block.arguments);
	if (textBlock.textToolRaw) return textBlock.textToolRaw;
	if (block.name === "bash") return `<bash${textToolAttrs(args, ["timeout"])}>\n${stringArg(args.command)}\n</bash>`;
	if (block.name === "read") return `<read${textToolAttrs(args, ["path", "offset", "limit"])}></read>`;
	if (block.name === "write") return `<write${textToolAttrs(args, ["path"])}>\n${stringArg(args.content)}\n</write>`;
	if (block.name === "edit") return textEditToolCallForProvider(args);
	if (block.name === "grep") {
		return `<grep${textToolAttrs(args, ["pattern", "path", "glob", "ignoreCase", "literal", "context", "limit"])}></grep>`;
	}
	if (block.name === "find") return `<find${textToolAttrs(args, ["pattern", "path", "limit"])}></find>`;
	if (block.name === "ls") return `<ls${textToolAttrs(args, ["path", "limit"])}></ls>`;
	if (block.name === "stop") return "<stop></stop>";
	return `<${block.name}>\n${formatKeyValueArgs(args)}\n</${block.name}>`;
}

export function formatToolResultText(message: ToolResultMessage): string {
	return (
		'<tool_results>\n<tool_result id="' +
		escapeToolAttr(message.toolCallId || "") +
		'" name="' +
		escapeToolAttr(message.toolName || "") +
		'" is_error="' +
		(message.isError ? "true" : "false") +
		'">\n' +
		safeToolResultText(getText(message.content)) +
		"\n</tool_result>\n</tool_results>"
	);
}

function normalizeTextToolSpec(tool: Tool | string): TextToolSpec {
	if (typeof tool === "string") return { name: tool, description: "" };
	return {
		name: tool.name,
		description: tool.description || "",
		parameters: tool.parameters,
	};
}

function textToolName(tool: TextToolSpec): string {
	return tool.name;
}

function isCustomTextToolSpec(tool: TextToolSpec): boolean {
	return !ALL_TOOLS.includes(tool.name);
}

function builtInToolExamples(enabled: Set<string>): string {
	const examples: string[] = [];
	for (const name of ALL_TOOLS) {
		const example = BUILT_IN_TOOL_EXAMPLES[name];
		if (enabled.has(name) && example) examples.push(example);
	}
	return examples.join("\n\n");
}

function customToolExample(name: string | undefined): string {
	if (!name) return "(none)";
	return `<${name}>
example_argument=replace with real schema fields
</${name}>`;
}

function formatTextToolSpec(tool: TextToolSpec): string {
	let out = `<${tool.name}>\n`;
	if (tool.description) out += `description: ${tool.description}\n`;
	if (tool.parameters) out += `parameters schema: ${JSON.stringify(tool.parameters)}\n`;
	out += `call format:\n<${tool.name}>\nkey=value\n</${tool.name}>`;
	return out;
}

function isToolEvent(event: AssistantMessageEvent): boolean {
	return event.type === "toolcall_start" || event.type === "toolcall_delta" || event.type === "toolcall_end";
}

function unwrapAssistantQuoteBlocks(message: AssistantMessage): boolean {
	let changed = false;
	for (const block of message.content) {
		if (block.type !== "text") continue;
		const text = unwrapQuoteBlocks(block.text);
		if (text === block.text) continue;
		block.text = text;
		changed = true;
	}
	return changed;
}

function unwrapQuoteBlocks(text: string): string {
	let out = "";
	let index = 0;
	while (index < text.length) {
		const start = findNextQuoteBlockStart(text, index);
		if (start === -1) {
			out += text.slice(index);
			break;
		}

		const block = readQuoteBlockAt(text, start);
		if (!block) {
			out += text.slice(index);
			break;
		}

		out += text.slice(index, start);
		out += unwrapQuoteBlocks(text.slice(block.bodyStart, block.closeStart));
		index = block.end;
	}
	return out;
}

function pushSyntheticAssistantContentEvents(stream: AssistantMessageEventStream, message: AssistantMessage): void {
	const content: AssistantMessage["content"] = [];
	stream.push({ type: "start", partial: assistantWithContent(message, content) });
	for (const block of message.content) {
		const contentIndex = content.length;
		content.push(block);
		const partial = assistantWithContent(message, content);
		if (block.type === "text") {
			stream.push({ type: "text_start", contentIndex, partial });
			if (block.text) stream.push({ type: "text_delta", contentIndex, delta: block.text, partial });
			stream.push({ type: "text_end", contentIndex, content: block.text, partial });
		} else if (block.type === "thinking") {
			stream.push({ type: "thinking_start", contentIndex, partial });
			if (block.thinking) stream.push({ type: "thinking_delta", contentIndex, delta: block.thinking, partial });
			stream.push({ type: "thinking_end", contentIndex, content: block.thinking, partial });
		} else if (block.type === "toolCall") {
			stream.push({ type: "toolcall_start", contentIndex, partial });
			stream.push({
				type: "toolcall_delta",
				contentIndex,
				delta: JSON.stringify(block.arguments ?? {}),
				partial,
			});
			stream.push({ type: "toolcall_end", contentIndex, toolCall: block, partial });
		}
	}
}

function assistantWithContent(message: AssistantMessage, content: AssistantMessage["content"]): AssistantMessage {
	return { ...message, content: content.slice() };
}

function pushSyntheticTextToolEvents(stream: AssistantMessageEventStream, message: AssistantMessage): void {
	for (let contentIndex = 0; contentIndex < message.content.length; contentIndex++) {
		const block = message.content[contentIndex];
		if (!block || block.type !== "toolCall") continue;
		stream.push({ type: "toolcall_start", contentIndex, partial: message });
		stream.push({
			type: "toolcall_delta",
			contentIndex,
			delta: JSON.stringify(block.arguments ?? {}),
			partial: message,
		});
		stream.push({ type: "toolcall_end", contentIndex, toolCall: block, partial: message });
	}
}

function contextToolName(tool: Tool): string {
	return tool.name;
}

function isToolCall(block: { type: string }): block is ToolCall {
	return block.type === "toolCall";
}

function getText(content: (TextContent | ImageContent)[] | AssistantMessage["content"] | string | undefined): string {
	if (typeof content === "string") return content;
	const chunks: string[] = [];
	for (const block of content || []) {
		if (block.type === "text") chunks.push(block.text);
	}
	return chunks.join("");
}

function toolCallsFromTextBlocks(blocks: ParsedToolBlock[], allowed?: Set<string>): ToolCall[] {
	const calls: ToolCall[] = [];
	for (const block of blocks) {
		if (allowed && !allowed.has(block.name)) continue;
		try {
			calls.push(parseTextToolBlock(block));
		} catch (error) {
			calls.push(textToolParseErrorCall(block, error));
		}
	}
	return calls;
}

function stripAssistantTextToolBlocks(assistant: AssistantMessage, allowed?: Set<string>): void {
	assistant.content = assistant.content
		.map(function stripTextBlock(block) {
			if (block.type !== "text") return block;
			return { ...block, text: maybeStripToolBlocks(block.text || "", allowed) };
		})
		.filter(function keepBlock(block) {
			return block.type !== "text" || block.text.trim();
		});
}

function parseToolBlocks(text: string, allowed?: Set<string>): ParsedToolBlock[] {
	text = stripKnownStopSequences(text);
	const out: ParsedToolBlock[] = [];
	let index = 0;
	while (index < text.length) {
		const start = findNextTextToolStart(text, index, allowed);
		if (start === -1) break;

		const nameAtStart = knownTextToolNameAt(text, start, allowed);
		const tagEnd = findTagEnd(text, start);
		if (tagEnd === -1) {
			out.push({
				name: nameAtStart,
				attrs: {},
				body: "",
				raw: text.slice(start),
				start,
				end: text.length,
				parseError: `Malformed <${nameAtStart}> tool block: missing >`,
			});
			break;
		}

		let header = text.slice(start + 1, tagEnd).trim();
		const selfClosing = header.endsWith("/");
		if (selfClosing) header = header.slice(0, -1).trim();

		const parsed = parseToolHeader(header);
		if (!isKnownTextToolName(parsed.name, allowed)) {
			index = tagEnd + 1;
			continue;
		}

		const attrs = parseAttrs(parsed.attrText);
		let body = "";
		let end = tagEnd + 1;
		if (!selfClosing) {
			const close = `</${parsed.name}>`;
			const closeStart = text.indexOf(close, tagEnd + 1);
			if (closeStart === -1) {
				out.push({
					name: parsed.name,
					attrs,
					body: text.slice(tagEnd + 1),
					raw: text.slice(start),
					start,
					end: text.length,
					parseError: `Malformed <${parsed.name}> tool block: missing ${close}`,
				});
				break;
			}

			body = text.slice(tagEnd + 1, closeStart);
			end = closeStart + close.length;
		}

		out.push({ name: parsed.name, attrs, body, raw: text.slice(start, end), start, end });
		index = end;
	}
	return out;
}

function findNextTextToolStart(text: string, from: number, allowed?: Set<string>): number {
	const tools = allowed ? Array.from(allowed) : ALL_TOOLS;
	let index = from;
	while (index < text.length) {
		const inertEnd = inertTextEndAt(text, index);
		if (inertEnd !== index) {
			index = Math.max(index + 1, inertEnd);
			continue;
		}

		if (text[index] !== "<") {
			index++;
			continue;
		}

		for (const name of tools) {
			if (text.startsWith(`<${name}`, index) && isToolNameBoundary(text[index + name.length + 1])) return index;
		}
		index++;
	}
	return -1;
}

function knownTextToolNameAt(text: string, index: number, allowed?: Set<string>): string {
	const tools = allowed ? Array.from(allowed) : ALL_TOOLS;
	for (const name of tools) {
		if (text.startsWith(`<${name}`, index) && isToolNameBoundary(text[index + name.length + 1])) return name;
	}
	return "";
}

function findNextLiteralOutsideInertText(text: string, literal: string, from: number): number {
	let index = from;
	while (index < text.length) {
		const inertEnd = inertTextEndAt(text, index);
		if (inertEnd !== index) {
			index = Math.max(index + 1, inertEnd);
			continue;
		}
		if (text.startsWith(literal, index)) return index;
		index++;
	}
	return -1;
}

function inertTextEndAt(text: string, index: number): number {
	if (isQuoteBlockStart(text, index)) return findQuoteBlockEnd(text, index);
	if (text[index] === "`") return findBacktickQuotedTextEnd(text, index);
	return index;
}

function findBacktickQuotedTextEnd(text: string, start: number): number {
	let ticks = 0;
	while (text[start + ticks] === "`") ticks++;
	if (!ticks) return start;

	let index = start + ticks;
	while (index < text.length) {
		let found = 0;
		while (found < ticks && text[index + found] === "`") found++;
		if (found === ticks) return index + ticks;
		index++;
	}
	return text.length;
}

function findNextQuoteBlockStart(text: string, from: number): number {
	for (
		let index = text.indexOf(`<${QUOTE_TOOL_NAME}`, from);
		index !== -1;
		index = text.indexOf(`<${QUOTE_TOOL_NAME}`, index + 1)
	) {
		if (isQuoteBlockStart(text, index)) return index;
	}
	return -1;
}

function isQuoteBlockStart(text: string, index: number): boolean {
	return text.startsWith(`<${QUOTE_TOOL_NAME}`, index) && isToolNameBoundary(text[index + QUOTE_TOOL_NAME.length + 1]);
}

function findQuoteBlockEnd(text: string, start: number): number {
	return readQuoteBlockAt(text, start)?.end ?? text.length;
}

function readQuoteBlockAt(
	text: string,
	start: number,
): { bodyStart: number; closeStart: number; end: number } | undefined {
	const tagEnd = findTagEnd(text, start);
	if (tagEnd === -1) return undefined;

	let header = text.slice(start + 1, tagEnd).trim();
	const selfClosing = header.endsWith("/");
	if (selfClosing) return { bodyStart: tagEnd + 1, closeStart: tagEnd + 1, end: tagEnd + 1 };
	if (selfClosing) header = header.slice(0, -1).trim();

	const parsed = parseToolHeader(header);
	if (parsed.name !== QUOTE_TOOL_NAME) return undefined;

	let depth = 1;
	let index = tagEnd + 1;
	while (index < text.length) {
		const nextOpen = findNextQuoteBlockStart(text, index);
		const nextClose = text.indexOf(`</${QUOTE_TOOL_NAME}>`, index);
		if (nextClose === -1) return undefined;

		if (nextOpen !== -1 && nextOpen < nextClose) {
			const openEnd = findTagEnd(text, nextOpen);
			if (openEnd === -1) return undefined;
			const openHeader = text.slice(nextOpen + 1, openEnd).trim();
			if (!openHeader.endsWith("/")) depth++;
			index = openEnd + 1;
			continue;
		}

		depth--;
		const end = nextClose + QUOTE_TOOL_NAME.length + 3;
		if (depth === 0) return { bodyStart: tagEnd + 1, closeStart: nextClose, end };
		index = end;
	}
	return undefined;
}

function isToolNameBoundary(ch: string | undefined): boolean {
	return !ch || ch === ">" || ch === "/" || isSpace(ch);
}

function findTagEnd(text: string, start: number): number {
	let quote = "";
	for (let index = start; index < text.length; index++) {
		const ch = text[index];
		if (quote) {
			if (ch === quote) quote = "";
		} else if (ch === "'" || ch === '"') {
			quote = ch;
		} else if (ch === ">") {
			return index;
		}
	}
	return -1;
}

function parseToolHeader(header: string): { name: string; attrText: string } {
	let index = 0;
	while (index < header.length && !isSpace(header[index])) index++;
	return { name: header.slice(0, index), attrText: header.slice(index).trim() };
}

function parseAttrs(text: string): Record<string, unknown> {
	const attrs: Record<string, unknown> = {};
	let index = 0;
	while (index < text.length) {
		index = skipSpaces(text, index);
		if (index >= text.length) break;

		const keyStart = index;
		while (index < text.length && text[index] !== "=" && !isSpace(text[index])) index++;
		const key = text.slice(keyStart, index);
		if (!key) break;

		index = skipSpaces(text, index);
		let value = "true";
		if (text[index] === "=") {
			index++;
			index = skipSpaces(text, index);
			const read = readAttrValue(text, index);
			value = read.value;
			index = read.index;
		}

		attrs[key] = parseLooseToolCodeValue(value);
	}
	return attrs;
}

function readAttrValue(text: string, index: number): { value: string; index: number } {
	if (text[index] === "'" || text[index] === '"') {
		const quote = text[index];
		const start = index + 1;
		let cursor = start;
		while (cursor < text.length && text[cursor] !== quote) cursor++;
		return { value: text.slice(start, cursor), index: cursor < text.length ? cursor + 1 : cursor };
	}

	let cursor = index;
	while (cursor < text.length && !isSpace(text[cursor])) cursor++;
	return { value: text.slice(index, cursor), index: cursor };
}

function skipSpaces(text: string, index: number): number {
	while (index < text.length && isSpace(text[index])) index++;
	return index;
}

function isSpace(ch: string | undefined): boolean {
	return ch === " " || ch === "\n" || ch === "\r" || ch === "\t";
}

function isKnownTextToolName(name: string, allowed?: Set<string>): boolean {
	return allowed ? allowed.has(name) : ALL_TOOLS.includes(name);
}

function parseTextToolBlock(block: ParsedToolBlock): ToolCall {
	if (block.parseError) throw new Error(block.parseError);
	if (block.name === "bash")
		return textToolCall(block, { command: cleanCommandToolBody(block.body), timeout: block.attrs.timeout });
	if (block.name === "read")
		return textToolCall(block, cleanToolArgs(pickAttrs(block.attrs, ["path", "offset", "limit"])));
	if (block.name === "write") {
		return textToolCall(block, cleanToolArgs({ path: block.attrs.path, content: cleanToolBody(block.body) }));
	}
	if (block.name === "edit") return parseTextEditToolBlock(block);
	if (block.name === "grep") {
		return textToolCall(
			block,
			cleanToolArgs(
				pickAttrs(block.attrs, ["pattern", "path", "glob", "ignoreCase", "literal", "context", "limit"]),
			),
		);
	}
	if (block.name === "find")
		return textToolCall(block, cleanToolArgs(pickAttrs(block.attrs, ["pattern", "path", "limit"])));
	if (block.name === "ls") return textToolCall(block, cleanToolArgs(pickAttrs(block.attrs, ["path", "limit"])));
	if (block.name === "stop") return textToolCall(block, {});

	let args = coerceCustomToolArgs(block.attrs);
	if (block.body) args = { ...args, ...parseCustomToolBody(block.body) };
	return textToolCall(block, args);
}

function coerceCustomToolArgs(attrs: Record<string, unknown>): Record<string, unknown> {
	const out: Record<string, unknown> = {};
	for (const key of Object.keys(attrs)) {
		out[key] = typeof attrs[key] === "string" ? parseLooseToolCodeValue(attrs[key]) : attrs[key];
	}
	return out;
}

function parseCustomToolBody(body: string): Record<string, unknown> {
	const text = cleanToolBody(body).trim();
	if (!text) return {};

	const objectLiteral = parseObjectLiteral(text);
	if (objectLiteral) return objectLiteral;

	const tagged = parseTaggedArgumentBody(text);
	if (Object.keys(tagged).length) return tagged;

	const lines = parseLineDelimitedToolArgs(text);
	if (Object.keys(lines).length) return lines;

	const arithmetic = parseArithmeticToolArgs(text);
	if (arithmetic) return arithmetic;

	return { _body: text };
}

/** Parse one whole compact object literal with the local handwritten scanner. */
function parseObjectLiteral(text: string): Record<string, unknown> | undefined {
	const parsed = parseToolLiteralWhole(text);
	return isRecord(parsed) ? parsed : undefined;
}

/** Parse one whole compact value literal without a generic parser. */
function parseToolLiteralWhole(text: string): unknown | undefined {
	const parsed = parseToolLiteral(text, 0);
	if (!parsed.ok) return undefined;
	if (skipSpaces(text, parsed.index) !== text.length) return undefined;
	return parsed.value;
}

/** Parse a compact object, array, string, atom, bool, null, or number. */
function parseToolLiteral(text: string, index: number): ToolLiteralRead {
	index = skipSpaces(text, index);
	const ch = text[index];
	if (ch === "{") return parseLiteralObjectAt(text, index);
	if (ch === "[") return parseLiteralArrayAt(text, index);
	if (ch === "'" || ch === '"') return parseLiteralStringAt(text, index);
	return parseLiteralAtom(text, index);
}

/** Parse a compact object literal. */
function parseLiteralObjectAt(text: string, index: number): ToolLiteralRead {
	const out: Record<string, unknown> = {};
	index++;
	for (;;) {
		index = skipSpaces(text, index);
		if (text[index] === "}") return { ok: true, value: out, index: index + 1 };
		const key = parseLiteralKey(text, index);
		if (!key.ok) return { ok: false, index };
		index = skipSpaces(text, key.index);
		if (text[index] !== ":") return { ok: false, index };
		const value = parseToolLiteral(text, index + 1);
		if (!value.ok) return { ok: false, index };
		out[String(key.value)] = value.value;
		index = skipSpaces(text, value.index);
		if (text[index] === ",") {
			index++;
			continue;
		}
		if (text[index] === "}") return { ok: true, value: out, index: index + 1 };
		return { ok: false, index };
	}
}

/** Parse a compact array literal. */
function parseLiteralArrayAt(text: string, index: number): ToolLiteralRead {
	const out: unknown[] = [];
	index++;
	for (;;) {
		index = skipSpaces(text, index);
		if (text[index] === "]") return { ok: true, value: out, index: index + 1 };
		const value = parseToolLiteral(text, index);
		if (!value.ok) return { ok: false, index };
		out.push(value.value);
		index = skipSpaces(text, value.index);
		if (text[index] === ",") {
			index++;
			continue;
		}
		if (text[index] === "]") return { ok: true, value: out, index: index + 1 };
		return { ok: false, index };
	}
}

/** Parse one object key. */
function parseLiteralKey(text: string, index: number): ToolLiteralRead {
	index = skipSpaces(text, index);
	if (text[index] === "'" || text[index] === '"') return parseLiteralStringAt(text, index);
	if (!isIdentifierStart(text[index])) return { ok: false, index };
	const start = index;
	index++;
	while (index < text.length && isIdentifierChar(text[index])) index++;
	return { ok: true, value: text.slice(start, index), index };
}

/** Parse one quoted string literal. */
function parseLiteralStringAt(text: string, index: number): ToolLiteralRead {
	const quote = text[index];
	let out = "";
	for (index++; index < text.length; index++) {
		const ch = text[index];
		if (ch === quote) return { ok: true, value: out, index: index + 1 };
		if (ch !== "\\") {
			out += ch;
			continue;
		}
		index++;
		if (index >= text.length) return { ok: false, index };
		out += escapedLiteralChar(text, index, quote);
		if (text[index] === "u" && isHexString(text.slice(index + 1, index + 5))) index += 4;
	}
	return { ok: false, index };
}

/** Return one decoded escape character. */
function escapedLiteralChar(text: string, index: number, quote: string): string {
	const ch = text[index];
	if (ch === "n") return "\n";
	if (ch === "t") return "\t";
	if (ch === "r") return "\r";
	if (ch === "b") return "\b";
	if (ch === "f") return "\f";
	if (ch === quote) return quote;
	if (ch === "\\" || ch === "/") return ch;
	if (ch === "u" && isHexString(text.slice(index + 1, index + 5))) {
		return String.fromCharCode(Number.parseInt(text.slice(index + 1, index + 5), 16));
	}
	return ch;
}

/** Parse one bare atom. */
function parseLiteralAtom(text: string, index: number): ToolLiteralRead {
	const start = index;
	while (index < text.length && !isLiteralTerminator(text[index])) index++;
	const raw = text.slice(start, index).trim();
	if (!raw) return { ok: false, index };
	if (raw === "true" || raw === "True") return { ok: true, value: true, index };
	if (raw === "false" || raw === "False") return { ok: true, value: false, index };
	if (raw === "null" || raw === "None") return { ok: true, value: null, index };
	if (isNumberLiteral(raw)) return { ok: true, value: Number(raw), index };
	return { ok: true, value: raw, index };
}

/** True when a character ends a literal atom. */
function isLiteralTerminator(ch: string | undefined): boolean {
	return !ch || ch === "," || ch === "}" || ch === "]" || isSpace(ch);
}

/** True for a simple finite number literal. */
function isNumberLiteral(text: string): boolean {
	let index = 0;
	if (text[index] === "-") index++;
	let digits = 0;
	while (isDigit(text[index])) {
		index++;
		digits++;
	}
	if (text[index] === ".") {
		index++;
		while (isDigit(text[index])) {
			index++;
			digits++;
		}
	}
	if (!digits) return false;
	if (text[index] === "e" || text[index] === "E") {
		index++;
		if (text[index] === "+" || text[index] === "-") index++;
		let expDigits = 0;
		while (isDigit(text[index])) {
			index++;
			expDigits++;
		}
		if (!expDigits) return false;
	}
	return index === text.length && Number.isFinite(Number(text));
}

/** True for ascii digit. */
function isDigit(ch: string | undefined): boolean {
	return !!ch && ch >= "0" && ch <= "9";
}

/** True when every char is a hex digit and at least one exists. */
function isHexString(text: string): boolean {
	if (!text) return false;
	for (let index = 0; index < text.length; index++) {
		if (!isHexDigit(text[index])) return false;
	}
	return true;
}

/** True for ascii hex digit. */
function isHexDigit(ch: string | undefined): boolean {
	return !!ch && ((ch >= "0" && ch <= "9") || (ch >= "a" && ch <= "f") || (ch >= "A" && ch <= "F"));
}

function parseTaggedArgumentBody(text: string): Record<string, unknown> {
	const out: Record<string, unknown> = {};
	let index = 0;
	while (index < text.length) {
		index = skipSpaces(text, index);
		if (index >= text.length) break;
		if (text[index] !== "<") return {};

		const tagEnd = findTagEnd(text, index);
		if (tagEnd === -1) return {};
		const header = parseToolHeader(text.slice(index + 1, tagEnd).trim());
		if (!isIdentifier(header.name)) return {};
		const close = `</${header.name}>`;
		const closeStart = text.indexOf(close, tagEnd + 1);
		if (closeStart === -1) return {};

		out[header.name] = parseLooseToolCodeValue(cleanToolBody(text.slice(tagEnd + 1, closeStart)).trim());
		index = closeStart + close.length;
	}
	return out;
}

function parseLineDelimitedToolArgs(text: string): Record<string, unknown> {
	const out: Record<string, unknown> = {};
	const lines = text.split("\n").map(trimCarriageLine).filter(Boolean);

	for (const line of lines) {
		const sep = firstLineArgumentSeparator(line);
		if (sep === -1) return {};
		const key = line.slice(0, sep).trim();
		if (!isIdentifier(key)) return {};
		out[key] = parseLooseToolCodeValue(line.slice(sep + 1).trim());
	}

	return out;
}

function trimCarriageLine(line: string): string {
	return line.replace("\r", "").trim();
}

function firstLineArgumentSeparator(line: string): number {
	for (let index = 0; index < line.length; index++) {
		if (line[index] === "=" || line[index] === ":") return index;
	}
	return -1;
}

function parseArithmeticToolArgs(text: string): Record<string, unknown> | undefined {
	const trimmed = text.trim();
	const opIndex = findArithmeticOperator(trimmed);
	if (opIndex === -1) return undefined;

	const a = Number(trimmed.slice(0, opIndex).trim());
	const b = Number(trimmed.slice(opIndex + 1).trim());
	if (!Number.isFinite(a) || !Number.isFinite(b)) return undefined;

	const op = trimmed[opIndex];
	return {
		a,
		b,
		operation: op === "+" ? "add" : op === "-" ? "subtract" : op === "/" ? "divide" : "multiply",
	};
}

function findArithmeticOperator(text: string): number {
	for (let index = 1; index < text.length; index++) {
		const ch = text[index];
		if (ch === "+" || ch === "*" || ch === "/" || ch === "x" || ch === "X") return index;
		if (ch === "-" && !isExponentSign(text, index)) return index;
	}
	return -1;
}

function isExponentSign(text: string, index: number): boolean {
	return (text[index - 1] === "e" || text[index - 1] === "E") && index + 1 < text.length;
}

function parseTextEditToolBlock(block: ParsedToolBlock): ToolCall {
	const oldText = extractTagBody(block.body, "old");
	const newText = extractTagBody(block.body, "new");
	if (oldText === undefined || newText === undefined) throw new Error("edit text tool needs <old> and <new> blocks");
	return textToolCall(block, {
		path: block.attrs.path,
		edits: [{ oldText: cleanToolBody(oldText), newText: cleanToolBody(newText) }],
	});
}

function pickAttrs(attrs: Record<string, unknown>, keys: string[]): Record<string, unknown> {
	const out: Record<string, unknown> = {};
	for (const key of keys) {
		if (attrs[key] !== undefined) out[key] = attrs[key];
	}
	return out;
}

function textToolCall(block: ParsedToolBlock, args: Record<string, unknown>): ToolCall {
	return {
		type: "toolCall",
		id: textToolCallId(block),
		name: block.name,
		arguments: dropUndefined(args),
		textToolRaw: block.raw,
	} as TextToolCall;
}

function textToolParseErrorCall(block: ParsedToolBlock, error: unknown): ToolCall {
	return {
		type: "toolCall",
		id: textToolCallId(block),
		name: block.name,
		arguments: { __toolParseError: errorText(error) },
		textToolRaw: block.raw,
	} as TextToolCall;
}

function textToolCallId(block: ParsedToolBlock): string {
	const id = block.attrs.id;
	return typeof id === "string" && id ? id : randomId(block.name || "tool");
}

function extractTagBody(text: string, tag: string): string | undefined {
	const open = `<${tag}>`;
	const close = `</${tag}>`;
	const start = text.indexOf(open);
	if (start === -1) return undefined;
	const bodyStart = start + open.length;
	const end = text.indexOf(close, bodyStart);
	if (end === -1) return undefined;
	return text.slice(bodyStart, end);
}

function cleanToolBody(text: string): string {
	if (text.startsWith("\r\n")) text = text.slice(2);
	else if (text.startsWith("\n")) text = text.slice(1);
	if (text.endsWith("\r\n")) text = text.slice(0, -2);
	else if (text.endsWith("\n")) text = text.slice(0, -1);
	return text;
}

function cleanCommandToolBody(text: string): string {
	text = stripKnownStopSequences(cleanToolBody(text).trim());
	text = stripSimpleTagBody(text, "command") ?? text;
	text = stripSimpleTagBody(text, "bash") ?? text;
	text = stripSimpleTagBody(text, "command") ?? text;
	return cleanToolBody(text);
}

function stripSimpleTagBody(text: string, name: string): string | undefined {
	const open = `<${name}`;
	const close = `</${name}>`;
	if (!text.trimStart().startsWith(open) || !text.trimEnd().endsWith(close)) return undefined;
	const start = text.indexOf(">");
	const end = text.lastIndexOf(close);
	if (start === -1 || end < start) return undefined;
	return text.slice(start + 1, end);
}

function cleanToolArgs(args: Record<string, unknown>): Record<string, unknown> {
	for (const key of ["path", "pattern", "glob"]) {
		if (typeof args[key] === "string") args[key] = stripKnownStopSequences(args[key]);
	}
	return args;
}

function stripKnownStopSequences(text: unknown): string {
	return String(text || "")
		.split("]<]minimax[>[")
		.join("")
		.trim();
}

function maybeStripToolBlocks(text: string, allowed?: Set<string>): string {
	let blocks = parseToolBlocks(text, allowed);
	if (!blocks.length) blocks = parseToolCodeBlocks(text);
	if (!blocks.length) return text;

	let out = "";
	let index = 0;
	for (const block of blocks) {
		out += text.slice(index, block.start);
		index = block.end;
	}
	return out + text.slice(index);
}

function toolCallsFromToolCode(text: string, allowed?: Set<string>): ToolCall[] {
	const calls: ToolCall[] = [];
	for (const block of parseToolCodeBlocks(text)) {
		try {
			calls.push(parseToolCodeCall(block.body, allowed, block.raw));
		} catch (error) {
			calls.push(textToolParseErrorCall({ ...block, name: "tool_code" }, error));
		}
	}
	return calls;
}

function parseToolCodeBlocks(text: string): ParsedToolBlock[] {
	const out: ParsedToolBlock[] = [];
	const open = "<tool_code>";
	const close = "</tool_code>";
	let index = 0;
	while (index < text.length) {
		const start = findNextLiteralOutsideInertText(text, open, index);
		if (start === -1) break;
		const closeStart = findNextLiteralOutsideInertText(text, close, start + open.length);
		if (closeStart === -1) break;
		const end = closeStart + close.length;
		out.push({
			name: "tool_code",
			attrs: {},
			body: text.slice(start + open.length, closeStart),
			raw: text.slice(start, end),
			start,
			end,
		});
		index = end;
	}
	return out;
}

function parseToolCodeCall(text: string, allowed: Set<string> | undefined, raw: string): ToolCall {
	let code = unwrapToolCodePrint(String(text || "").trim());
	if (!code) throw new Error("Empty <tool_code> body");
	if (code.endsWith(";")) code = code.slice(0, -1).trim();

	let index = skipToolCodeSpaces(code, 0);
	const start = index;
	if (!isIdentifierStart(code[index])) throw new Error("No tool name in <tool_code>");
	index++;
	while (index < code.length && isIdentifierChar(code[index])) index++;
	const name = code.slice(start, index);
	if (allowed && !allowed.has(name)) throw new Error(`Unknown tool in <tool_code>: ${name}`);

	index = skipToolCodeSpaces(code, index);
	if (code[index] !== "(") throw new Error("Expected ( after tool name in <tool_code>");
	const end = findToolCodeCallEnd(code, index);
	if (end === -1) throw new Error("Malformed call in <tool_code>");

	return {
		type: "toolCall",
		id: randomId(name),
		name,
		arguments: parseToolCodeArguments(code.slice(index + 1, end), name),
		textToolRaw: raw,
	} as TextToolCall;
}

function unwrapToolCodePrint(text: string): string {
	if (!text.startsWith("print")) return text;
	const index = skipToolCodeSpaces(text, 5);
	if (text[index] !== "(") return text;
	const end = findToolCodeCallEnd(text, index);
	if (end === -1) return text;
	if (text.slice(end + 1).trim()) return text;
	return text.slice(index + 1, end).trim();
}

function parseToolCodeArguments(text: string, name: string): Record<string, unknown> {
	const pieces = splitTopLevelToolCodeArgs(text.trim());
	if (!pieces.length || !pieces[0]) return {};

	if (pieces.length === 1 && topLevelToolCodeEqualsIndex(pieces[0]) === -1) {
		const only = parseLooseToolCodeValue(pieces[0]);
		if (name === "bash" && typeof only === "string") return { command: only };
		if (name === "read" && typeof only === "string") return { path: only };
		if (isRecord(only)) return only;
		return { _args: [only] };
	}

	const args: Record<string, unknown> = {};
	const positional: unknown[] = [];
	for (const piece of pieces) {
		const eq = topLevelToolCodeEqualsIndex(piece);
		if (eq === -1) {
			positional.push(parseLooseToolCodeValue(piece));
			continue;
		}
		const key = piece.slice(0, eq).trim();
		if (key) args[key] = parseLooseToolCodeValue(piece.slice(eq + 1));
	}

	if (positional.length) args._args = positional;
	return args;
}

function splitTopLevelToolCodeArgs(text: string): string[] {
	if (!text) return [];

	const parts: string[] = [];
	let start = 0;
	let depth = 0;
	let quote = "";
	let escaped = false;

	for (let index = 0; index < text.length; index++) {
		const ch = text[index];
		if (quote) {
			if (escaped) escaped = false;
			else if (ch === "\\") escaped = true;
			else if (ch === quote) quote = "";
			continue;
		}

		if (ch === "'" || ch === '"') {
			quote = ch;
			continue;
		}
		if (ch === "(" || ch === "[" || ch === "{") {
			depth++;
			continue;
		}
		if (ch === ")" || ch === "]" || ch === "}") {
			if (depth > 0) depth--;
			continue;
		}
		if (ch === "," && depth === 0) {
			parts.push(text.slice(start, index).trim());
			start = index + 1;
		}
	}

	const tail = text.slice(start).trim();
	if (tail) parts.push(tail);
	return parts;
}

function topLevelToolCodeEqualsIndex(text: string): number {
	let depth = 0;
	let quote = "";
	let escaped = false;

	for (let index = 0; index < text.length; index++) {
		const ch = text[index];
		if (quote) {
			if (escaped) escaped = false;
			else if (ch === "\\") escaped = true;
			else if (ch === quote) quote = "";
			continue;
		}

		if (ch === "'" || ch === '"') {
			quote = ch;
			continue;
		}
		if (ch === "(" || ch === "[" || ch === "{") {
			depth++;
			continue;
		}
		if (ch === ")" || ch === "]" || ch === "}") {
			if (depth > 0) depth--;
			continue;
		}
		if (ch === "=" && depth === 0) return index;
	}

	return -1;
}

function parseLooseToolCodeValue(text: unknown): unknown {
	const value = String(text || "").trim();
	if (!value) return "";
	if (isQuoted(value)) return unquoteToolCodeString(value);
	if (value === "true" || value === "True") return true;
	if (value === "false" || value === "False") return false;
	if (value === "null" || value === "None") return null;

	const numberValue = Number(value);
	if (!Number.isNaN(numberValue) && value !== "") return numberValue;

	if (value[0] === "{" || value[0] === "[") {
		const parsed = parseToolLiteralWhole(value);
		if (parsed !== undefined) return parsed;
	}

	return value;
}

function isQuoted(value: string): boolean {
	if (value.length < 2) return false;
	return (
		(value[0] === '"' && value[value.length - 1] === '"') || (value[0] === "'" && value[value.length - 1] === "'")
	);
}

function unquoteToolCodeString(text: string): string {
	const quote = text[0];
	let out = "";
	for (let index = 1; index < text.length - 1; index++) {
		const ch = text[index];
		if (ch === "\\" && index + 1 < text.length - 1) {
			index++;
			const next = text[index];
			if (next === "n") out += "\n";
			else if (next === "t") out += "\t";
			else if (next === quote) out += quote;
			else out += next;
			continue;
		}
		out += ch;
	}
	return out;
}

function findToolCodeCallEnd(text: string, openIndex: number): number {
	let depth = 0;
	let quote = "";
	let escaped = false;

	for (let index = openIndex; index < text.length; index++) {
		const ch = text[index];
		if (quote) {
			if (escaped) escaped = false;
			else if (ch === "\\") escaped = true;
			else if (ch === quote) quote = "";
			continue;
		}

		if (ch === "'" || ch === '"') {
			quote = ch;
			continue;
		}
		if (ch === "(") {
			depth++;
			continue;
		}
		if (ch === ")") {
			depth--;
			if (depth === 0) return index;
		}
	}

	return -1;
}

function skipToolCodeSpaces(text: string, index: number): number {
	while (index < text.length && isSpace(text[index])) index++;
	return index;
}

function isIdentifier(text: string): boolean {
	if (!isIdentifierStart(text[0])) return false;
	for (let index = 1; index < text.length; index++) {
		if (!isIdentifierChar(text[index])) return false;
	}
	return true;
}

function isIdentifierStart(ch: string | undefined): boolean {
	return !!ch && ((ch >= "a" && ch <= "z") || (ch >= "A" && ch <= "Z") || ch === "_");
}

function isIdentifierChar(ch: string | undefined): boolean {
	return isIdentifierStart(ch) || (!!ch && ch >= "0" && ch <= "9") || ch === "-";
}

function textEditToolCallForProvider(args: Record<string, unknown>): string {
	const edits = Array.isArray(args.edits) ? args.edits : [];
	const edit = isRecord(edits[0]) ? edits[0] : {};
	return (
		"<edit" +
		textToolAttrs(args, ["path"]) +
		">\n<old>\n" +
		stringArg(edit.oldText) +
		"\n</old>\n<new>\n" +
		stringArg(edit.newText) +
		"\n</new>\n</edit>"
	);
}

function textToolAttrs(args: Record<string, unknown>, keys: string[]): string {
	const parts: string[] = [];
	for (const key of keys) {
		if (args[key] !== undefined) parts.push(`${key}="${escapeToolAttr(args[key])}"`);
	}
	return parts.length ? ` ${parts.join(" ")}` : "";
}

function formatKeyValueArgs(args: Record<string, unknown>): string {
	const lines: string[] = [];
	for (const key of Object.keys(args)) {
		lines.push(`${key}=${formatValue(args[key])}`);
	}
	return lines.join("\n");
}

function formatValue(value: unknown): string {
	if (value === undefined) return "";
	if (typeof value === "string") return value;
	if (typeof value === "number" || typeof value === "boolean") return String(value);
	return JSON.stringify(value);
}

function escapeToolAttr(value: unknown): string {
	return String(value)
		.split("\n")
		.join(" ")
		.split("\r")
		.join(" ")
		.split('"')
		.join("'")
		.split("<")
		.join("(")
		.split(">")
		.join(")");
}

function safeToolResultText(text: string): string {
	return String(text || "")
		.split("</tool_result>")
		.join("</ tool_result>")
		.split("</tool_results>")
		.join("</ tool_results>");
}

function dropUndefined(obj: Record<string, unknown>): Record<string, unknown> {
	const out: Record<string, unknown> = {};
	for (const key of Object.keys(obj)) {
		if (obj[key] !== undefined) out[key] = obj[key];
	}
	return out;
}

function errorText(error: unknown): string {
	return error instanceof Error ? error.message : String(error);
}

function randomId(prefix: string): string {
	return `${prefix}_${Math.random().toString(36).slice(2, 9)}`;
}

function objectArgs(value: unknown): Record<string, unknown> {
	return isRecord(value) ? value : {};
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return !!value && typeof value === "object" && !Array.isArray(value);
}

function stringArg(value: unknown): string {
	return typeof value === "string" ? value : value === undefined || value === null ? "" : String(value);
}
