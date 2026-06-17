import { Type } from "typebox";
import { describe, expect, it } from "vitest";
import { textToolProtocol } from "../src/utils/text-tools.ts";

describe("google text-tool protocol", () => {
	it("serializes tool schemas into plain text instead of native function declarations", () => {
		const protocol = textToolProtocol([
			{
				name: "lookup",
				description: "Look up a value",
				parameters: Type.Object({
					value: Type.String(),
					nested: Type.Object({
						count: Type.Number(),
					}),
				}),
			},
		]);

		expect(protocol).toContain("Provider-native JSON/function tools are disabled");
		expect(protocol).toContain("Enabled tool names: lookup");
		expect(protocol).toContain("<lookup>");
		expect(protocol).toContain("parameters schema");
		expect(protocol).toContain("nested");
		expect(protocol).toContain("count");
		expect(protocol).toContain("");
		expect(protocol).toContain("");
	});
});
