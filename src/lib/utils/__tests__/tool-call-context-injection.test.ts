import { describe, expect, it } from 'vitest';

import {
	getToolCallDetailsFromContent,
	updateAllToolCallContextInjectionStatesInContent,
	updateToolCallContextInjectionStateInContent
} from '../tool-call-context-injection';

describe('getToolCallDetailsFromContent', () => {
	it('returns an empty array when there are no tool calls', () => {
		expect(getToolCallDetailsFromContent('plain content')).toEqual([]);
	});

	it('parses done and context injection state from tool call details', () => {
		const content =
			'hello<details type="tool_calls" id="tool-1" done="true"></details><details type="tool_calls" id="tool-2" done="false" context_injection_disabled="true"></details><details type="reasoning" id="skip"></details>';

		expect(getToolCallDetailsFromContent(content)).toEqual([
			{ id: 'tool-1', done: true, contextInjectionDisabled: false },
			{ id: 'tool-2', done: false, contextInjectionDisabled: true }
		]);
	});
});

describe('updateToolCallContextInjectionStateInContent', () => {
	it('updates only the matching tool call', () => {
		const content =
			'<details type="tool_calls" id="tool-1" done="true"></details><details type="tool_calls" id="tool-2" done="true" context_injection_disabled="false"></details>';

		expect(updateToolCallContextInjectionStateInContent(content, 'tool-2', true)).toBe(
			'<details type="tool_calls" id="tool-1" done="true"></details><details type="tool_calls" id="tool-2" done="true" context_injection_disabled="true"></details>'
		);
	});
});

describe('updateAllToolCallContextInjectionStatesInContent', () => {
	it('sets all tool calls to disabled and preserves non-tool details', () => {
		const content =
			'before<details type="tool_calls" id="tool-1" done="true"></details><details type="reasoning" id="reason-1"></details><details type="tool_calls" id="tool-2" done="true" context_injection_disabled="false"></details>after';

		expect(updateAllToolCallContextInjectionStatesInContent(content, true)).toBe(
			'before<details type="tool_calls" id="tool-1" done="true" context_injection_disabled="true"></details><details type="reasoning" id="reason-1"></details><details type="tool_calls" id="tool-2" done="true" context_injection_disabled="true"></details>after'
		);
	});

	it('sets all tool calls back to enabled', () => {
		const content =
			'<details type="tool_calls" id="tool-1" done="true" context_injection_disabled="true"></details><details type="tool_calls" id="tool-2" done="true"></details>';

		expect(updateAllToolCallContextInjectionStatesInContent(content, false)).toBe(
			'<details type="tool_calls" id="tool-1" done="true" context_injection_disabled="false"></details><details type="tool_calls" id="tool-2" done="true" context_injection_disabled="false"></details>'
		);
	});
});
