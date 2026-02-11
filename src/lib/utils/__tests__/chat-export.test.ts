import { describe, expect, it } from 'vitest';

import { buildChatMarkdownFileName, serializeChatToMarkdown } from '../chat-export';

describe('serializeChatToMarkdown', () => {
	it('serializes a chat history into markdown role sections', () => {
		const chat = {
			chat: {
				title: 'Demo Chat',
				history: {
					currentId: 'assistant-1',
					messages: {
						'user-1': {
							id: 'user-1',
							parentId: null,
							childrenIds: ['assistant-1'],
							role: 'user',
							content: 'Hello'
						},
						'assistant-1': {
							id: 'assistant-1',
							parentId: 'user-1',
							childrenIds: [],
							role: 'assistant',
							content: 'Hi there'
						}
					}
				}
			}
		};

		expect(serializeChatToMarkdown(chat)).toBe('### USER\nHello\n\n### ASSISTANT\nHi there');
	});

	it('returns empty string when history is unavailable', () => {
		expect(serializeChatToMarkdown({ chat: {} })).toBe('');
	});
});

describe('buildChatMarkdownFileName', () => {
	it('builds deterministic file name from title and timestamp', () => {
		const chat = {
			chat: {
				title: 'My Chat Title'
			}
		};

		const fixedDate = new Date(2026, 1, 11, 9, 5);
		expect(buildChatMarkdownFileName(chat, fixedDate)).toBe('chat-my-chat-title-20260211-0905.md');
	});
});
