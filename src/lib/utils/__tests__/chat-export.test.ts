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

	it('removes reasoning details when includeThinkingContent is false', () => {
		const chat = {
			chat: {
				history: {
					currentId: 'assistant-1',
					messages: {
						'assistant-1': {
							id: 'assistant-1',
							parentId: null,
							childrenIds: [],
							role: 'assistant',
							content: 'before<details type="reasoning">secret</details>after'
						}
					}
				}
			}
		};

		expect(serializeChatToMarkdown(chat, { includeThinkingContent: false })).toBe(
			'### ASSISTANT\nbeforeafter'
		);
	});

	it('removes tool call details when includeToolCallingContent is false', () => {
		const chat = {
			chat: {
				history: {
					currentId: 'assistant-1',
					messages: {
						'assistant-1': {
							id: 'assistant-1',
							parentId: null,
							childrenIds: [],
							role: 'assistant',
							content: 'before<details type="tool_calls">tool</details>after'
						}
					}
				}
			}
		};

		expect(serializeChatToMarkdown(chat, { includeToolCallingContent: false })).toBe(
			'### ASSISTANT\nbeforeafter'
		);
	});

	it('removes both details types when both options are false', () => {
		const chat = {
			chat: {
				history: {
					currentId: 'assistant-1',
					messages: {
						'assistant-1': {
							id: 'assistant-1',
							parentId: null,
							childrenIds: [],
							role: 'assistant',
							content:
								'start<details type="reasoning">secret</details><details type="tool_calls">tool</details>end'
						}
					}
				}
			}
		};

		expect(
			serializeChatToMarkdown(chat, {
				includeThinkingContent: false,
				includeToolCallingContent: false
			})
		).toBe('### ASSISTANT\nstartend');
	});

	it('keeps non-details content unchanged when filters are disabled', () => {
		const chat = {
			chat: {
				history: {
					currentId: 'assistant-1',
					messages: {
						'assistant-1': {
							id: 'assistant-1',
							parentId: null,
							childrenIds: [],
							role: 'assistant',
							content: 'plain content'
						}
					}
				}
			}
		};

		expect(
			serializeChatToMarkdown(chat, {
				includeThinkingContent: false,
				includeToolCallingContent: false
			})
		).toBe('### ASSISTANT\nplain content');
	});

	it('removes citations by default when sources exist', () => {
		const chat = {
			chat: {
				history: {
					currentId: 'assistant-1',
					messages: {
						'assistant-1': {
							id: 'assistant-1',
							parentId: null,
							childrenIds: [],
							role: 'assistant',
							content: 'Result [1] and [2]',
							sources: [{ document: ['a', 'b'] }]
						}
					}
				}
			}
		};

		expect(serializeChatToMarkdown(chat)).toBe('### ASSISTANT\nResult and');
	});

	it('preserves citations when excludeCitations is false', () => {
		const chat = {
			chat: {
				history: {
					currentId: 'assistant-1',
					messages: {
						'assistant-1': {
							id: 'assistant-1',
							parentId: null,
							childrenIds: [],
							role: 'assistant',
							content: 'Result [1] and [2]',
							sources: [{ document: ['a', 'b'] }]
						}
					}
				}
			}
		};

		expect(serializeChatToMarkdown(chat, { excludeCitations: false })).toBe(
			'### ASSISTANT\nResult [1] and [2]'
		);
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
		expect(buildChatMarkdownFileName(chat, fixedDate)).toBe('chat-20260211-0905-My Chat Title.md');
	});

	it('preserves non-latin title characters', () => {
		const chat = {
			chat: {
				title: '我的会话'
			}
		};

		const fixedDate = new Date(2026, 1, 11, 9, 5);
		expect(buildChatMarkdownFileName(chat, fixedDate)).toBe('chat-20260211-0905-我的会话.md');
	});

	it('sanitizes invalid filename characters', () => {
		const chat = {
			chat: {
				title: 'A/B:C'
			}
		};

		const fixedDate = new Date(2026, 1, 11, 9, 5);
		expect(buildChatMarkdownFileName(chat, fixedDate)).toBe('chat-20260211-0905-A-B-C.md');
	});

	it('falls back to chat when title becomes empty', () => {
		const chat = {
			chat: {
				title: '...'
			}
		};

		const fixedDate = new Date(2026, 1, 11, 9, 5);
		expect(buildChatMarkdownFileName(chat, fixedDate)).toBe('chat-20260211-0905-chat.md');
	});
});
