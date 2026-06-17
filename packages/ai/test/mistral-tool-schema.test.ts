import { Type } from "typebox";
import { describe, expect, it } from "vitest";
import { getModel } from "../src/models.ts";
import { complete } from "../src/stream.ts";
import type { Context, Model } from "../src/types.ts";

interface MistralToolPayload {
	tools?: Array<{
		type: "function";
		function: {
			name: string;
			parameters: Record<string, unknown>;
		};
	}>;
}

describe("Mistral tool schema serialization", () => {
	it("strips TypeBox symbol keys before the SDK validates tool schemas", async () => {
		const model: Model<"mistral-conversations"> = {
			...getModel("mistral", "devstral-medium-latest"),
			baseUrl: "http://127.0.0.1:9",
		};
		const parameters = Type.Object({
			nested: Type.Object({
				value: Type.String(),
			}),
		});
		const context: Context = {
			messages: [{ role: "user", content: "Hi", timestamp: Date.now() }],
			tools: [
				{
					name: "inspect_schema",
					description: "Inspect the schema",
					parameters,
				},
			],
		};
		let capturedPayload: MistralToolPayload | undefined;

		const response = await complete(model, context, {
			apiKey: "fake-key",
			onPayload: (payload) => {
				capturedPayload = payload as MistralToolPayload;
				return payload;
			},
		});

		expect(capturedPayload?.tools).toBeUndefined();
		expect(JSON.stringify(capturedPayload)).toContain("inspect_schema");
		expect(JSON.stringify(capturedPayload)).toContain("nested");
		expect(response.stopReason).toBe("error");
		expect(response.errorMessage).not.toContain("Input validation failed");
	});
});
