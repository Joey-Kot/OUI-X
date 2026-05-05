import { describe, expect, it } from 'vitest';

import {
	normalizeContextTruncation,
	trimMessagesAfterContextTruncation
} from '../context-truncation';

const messages = [
	{ id: 'user-1', role: 'user' },
	{ id: 'assistant-1', role: 'assistant' },
	{ id: 'user-2', role: 'user' },
	{ id: 'assistant-2', role: 'assistant' }
];

describe('trimMessagesAfterContextTruncation', () => {
	it('returns the full message chain when truncation is disabled', () => {
		expect(
			trimMessagesAfterContextTruncation(messages, {
				enabled: false,
				cutoffMessageId: null,
				updatedAt: null
			})
		).toBe(messages);
	});

	it('returns only messages after the cutoff when the cutoff is in the chain', () => {
		expect(
			trimMessagesAfterContextTruncation(messages, {
				enabled: true,
				cutoffMessageId: 'assistant-1',
				updatedAt: 1
			})
		).toEqual([
			{ id: 'user-2', role: 'user' },
			{ id: 'assistant-2', role: 'assistant' }
		]);
	});

	it('returns the full current branch when the cutoff is not in the chain', () => {
		expect(
			trimMessagesAfterContextTruncation(messages, {
				enabled: true,
				cutoffMessageId: 'other-branch-message',
				updatedAt: 1
			})
		).toBe(messages);
	});
});

describe('normalizeContextTruncation', () => {
	it('clears truncation when the cutoff message no longer exists', () => {
		expect(
			normalizeContextTruncation(
				{
					enabled: true,
					cutoffMessageId: 'deleted-message',
					updatedAt: 1
				},
				{
					messages: {
						'user-1': messages[0]
					}
				}
			)
		).toEqual({
			enabled: false,
			cutoffMessageId: null,
			updatedAt: null
		});
	});
});
