import { createMessagesList } from '$lib/utils';
import { getCitationSourceCount, stripCitationTokens } from '$lib/utils/message-export';

export type SerializableChat = {
	chat?: {
		history?: {
			currentId?: string;
			messages?: Record<string, unknown>;
		};
		title?: string;
	};
};

type SerializeChatToMarkdownOptions = {
	includeThinkingContent?: boolean;
	includeToolCallingContent?: boolean;
	excludeCitations?: boolean;
};

const stripTrailingDots = (value: string): string => {
	let result = value;
	while (result.endsWith('.')) {
		result = result.slice(0, -1);
	}
	return result;
};

const sanitizeFileTitle = (title: string): string => {
	const sanitized = stripTrailingDots(
		title
			// Replace Windows/Unix-invalid filename characters and control chars.
			.replace(/[\x5C/:*?\x22<>\x7C\x00-\x1F]/g, '-')
			.trim()
	).trim();

	return sanitized || 'chat';
};

const formatTimestamp = (date: Date): string => {
	const yyyy = date.getFullYear();
	const mm = String(date.getMonth() + 1).padStart(2, '0');
	const dd = String(date.getDate()).padStart(2, '0');
	const hh = String(date.getHours()).padStart(2, '0');
	const min = String(date.getMinutes()).padStart(2, '0');
	return String(yyyy) + mm + dd + '-' + hh + min;
};

const removeDetailsByType = (content: string, type: 'reasoning' | 'tool_calls'): string => {
	return content.replace(new RegExp(`<details\\s+type="${type}"[^>]*>[\\s\\S]*?<\\/details>`, 'gis'), '');
};

export const serializeChatToMarkdown = (
	chat: SerializableChat,
	{
		includeThinkingContent = true,
		includeToolCallingContent = true,
		excludeCitations = true
	}: SerializeChatToMarkdownOptions = {}
): string => {
	const history = chat?.chat?.history;
	if (!history?.messages || !history.currentId) {
		return '';
	}

	const messages = createMessagesList(history, history.currentId);
	return messages
		.reduce((acc, message) => {
			const messageWithSources = message as { sources?: Array<unknown> };
			const role = String(message?.role ?? '').toUpperCase();
			let content = String(message?.content ?? '');

			if (!includeThinkingContent) {
				content = removeDetailsByType(content, 'reasoning');
			}

			if (!includeToolCallingContent) {
				content = removeDetailsByType(content, 'tool_calls');
			}

			if (excludeCitations) {
				content = stripCitationTokens(content, getCitationSourceCount(messageWithSources.sources));
			}

			return acc + '### ' + role + '\n' + content + '\n\n';
		}, '')
		.trim();
};

export const buildChatMarkdownFileName = (chat: SerializableChat, date: Date = new Date()): string => {
	const title = chat?.chat?.title?.trim() || 'chat';
	const sanitizedTitle = sanitizeFileTitle(title);
	return 'chat-' + formatTimestamp(date) + '-' + sanitizedTitle + '.md';
};
