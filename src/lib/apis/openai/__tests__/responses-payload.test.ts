import { describe, expect, it } from 'vitest';

import { sanitizeResponsesRequestBody } from '../responses-payload';

describe('sanitizeResponsesRequestBody', () => {
	it('removes null response params and keeps non-parameter metadata', () => {
		const result = sanitizeResponsesRequestBody({
			model: 'gpt-5-mini',
			temperature: null,
			top_p: null,
			stop: null,
			verbosity: null,
			max_output_tokens: null,
			chat_id: 'chat-1',
			id: 'msg-1',
			background_tasks: { follow_up_generation: true },
			reasoning: { effort: null, summary: null }
		});

		expect(result).toEqual({
			model: 'gpt-5-mini',
			chat_id: 'chat-1',
			id: 'msg-1',
			background_tasks: { follow_up_generation: true }
		});
	});

	it('keeps explicitly configured response params', () => {
		const result = sanitizeResponsesRequestBody({
			model: 'gpt-5-mini',
			temperature: 0.9,
			top_p: 0.8,
			stop: ['DONE'],
			verbosity: 'high',
			max_output_tokens: 512,
			reasoning: { effort: 'high', summary: 'detailed' }
		});

		expect(result).toEqual({
			model: 'gpt-5-mini',
			temperature: 0.9,
			top_p: 0.8,
			stop: ['DONE'],
			verbosity: 'high',
			max_output_tokens: 512,
			reasoning: { effort: 'high', summary: 'detailed' }
		});
	});
});
