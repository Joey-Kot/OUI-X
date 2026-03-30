import { describe, expect, it } from 'vitest';

import { extractTaskContentFromCompletion } from '$lib/apis';

describe('extractTaskContentFromCompletion', () => {
	it('extracts from chat completions payload', () => {
		const payload = {
			choices: [
				{
					message: {
						role: 'assistant',
						content: 'chat text'
					}
				}
			]
		};

		expect(extractTaskContentFromCompletion(payload)).toBe('chat text');
	});

	it('extracts from responses payload', () => {
		const payload = {
			object: 'response',
			output: [
				{
					type: 'message',
					content: [
						{ type: 'output_text', text: 'responses ' },
						{ type: 'output_text', text: 'text' }
					]
				}
			]
		};

		expect(extractTaskContentFromCompletion(payload)).toBe('responses text');
	});
});

