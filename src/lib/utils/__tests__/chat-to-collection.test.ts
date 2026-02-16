import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { addChatToCollection } from '../chat-to-collection';

const uploadFileMock = vi.fn();
const addFileToKnowledgeByIdMock = vi.fn();
const deleteFileByIdMock = vi.fn();

vi.mock('$lib/apis/files', () => ({
	uploadFile: (...args: unknown[]) => uploadFileMock(...args),
	deleteFileById: (...args: unknown[]) => deleteFileByIdMock(...args)
}));

vi.mock('$lib/apis/knowledge', () => ({
	addFileToKnowledgeById: (...args: unknown[]) => addFileToKnowledgeByIdMock(...args)
}));

describe('addChatToCollection', () => {
	const token = 'token';
	const knowledgeId = 'knowledge-1';
	const chat = {
		chat: {
			title: 'Collection Chat',
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
						content: 'Hi [1]',
						sources: [{ document: ['doc-1'] }]
					}
				}
			}
		}
	};

	beforeEach(() => {
		vi.useFakeTimers();
		vi.setSystemTime(new Date(2026, 1, 11, 9, 5));
		uploadFileMock.mockReset();
		addFileToKnowledgeByIdMock.mockReset();
		deleteFileByIdMock.mockReset();
	});

	afterEach(() => {
		vi.useRealTimers();
	});

	it('uploads markdown and attaches file to knowledge base', async () => {
		uploadFileMock.mockResolvedValue({ id: 'file-1' });
		addFileToKnowledgeByIdMock.mockResolvedValue({ status: true });

		const result = await addChatToCollection({ token, chat, knowledgeId });

		expect(uploadFileMock).toHaveBeenCalledTimes(1);
		expect((uploadFileMock.mock.calls[0]?.[1] as File)?.name).toBe(
			'chat-20260211-0905-Collection Chat.md'
		);
		await expect(((uploadFileMock.mock.calls[0]?.[1] as File).text())).resolves.toContain(
			'### ASSISTANT\nHi'
		);
		await expect(((uploadFileMock.mock.calls[0]?.[1] as File).text())).resolves.not.toContain('[1]');
		expect(addFileToKnowledgeByIdMock).toHaveBeenCalledWith(token, knowledgeId, 'file-1');
		expect(deleteFileByIdMock).not.toHaveBeenCalled();
		expect(result).toEqual({ fileId: 'file-1', knowledgeId });
	});

	it('filters reasoning and tool-calling details when options are disabled', async () => {
		uploadFileMock.mockResolvedValue({ id: 'file-1' });
		addFileToKnowledgeByIdMock.mockResolvedValue({ status: true });

		const detailedChat = {
			chat: {
				title: 'Collection Chat',
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
							content:
								'A<details type="reasoning">think</details><details type="tool_calls">tool</details>B'
						}
					}
				}
			}
		};

		await addChatToCollection({
			token,
			chat: detailedChat,
			knowledgeId,
			includeThinkingContent: false,
			includeToolCallingContent: false
		});

		const uploadedFile = uploadFileMock.mock.calls[0]?.[1] as File;
		await expect(uploadedFile.text()).resolves.toContain('### ASSISTANT\nAB');
	});

	it('throws when upload returns no file id', async () => {
		uploadFileMock.mockResolvedValue(null);

		await expect(addChatToCollection({ token, chat, knowledgeId })).rejects.toThrow(
			'Failed to upload file.'
		);
		expect(addFileToKnowledgeByIdMock).not.toHaveBeenCalled();
	});

	it('cleans up uploaded file when knowledge add fails', async () => {
		uploadFileMock.mockResolvedValue({ id: 'file-1' });
		addFileToKnowledgeByIdMock.mockRejectedValue(new Error('add failed'));
		deleteFileByIdMock.mockResolvedValue({ status: true });

		await expect(addChatToCollection({ token, chat, knowledgeId })).rejects.toThrow('add failed');
		expect(deleteFileByIdMock).toHaveBeenCalledWith(token, 'file-1');
	});
});
