export type ContextTruncationState = {
	enabled: boolean;
	cutoffMessageId: string | null;
	updatedAt: number | null;
};

type MessageLike = {
	id?: string | null;
};

type HistoryLike = {
	messages?: Record<string, unknown>;
};

export const createDisabledContextTruncation = (): ContextTruncationState => ({
	enabled: false,
	cutoffMessageId: null,
	updatedAt: null
});

export const normalizeContextTruncation = (
	value: unknown,
	history?: HistoryLike | null
): ContextTruncationState => {
	if (!value || typeof value !== 'object' || Array.isArray(value)) {
		return createDisabledContextTruncation();
	}

	const rawValue = value as Partial<ContextTruncationState>;
	const cutoffMessageId =
		typeof rawValue.cutoffMessageId === 'string' && rawValue.cutoffMessageId.trim()
			? rawValue.cutoffMessageId
			: null;

	if (!rawValue.enabled || !cutoffMessageId) {
		return createDisabledContextTruncation();
	}

	if (history?.messages && !history.messages[cutoffMessageId]) {
		return createDisabledContextTruncation();
	}

	return {
		enabled: true,
		cutoffMessageId,
		updatedAt: typeof rawValue.updatedAt === 'number' ? rawValue.updatedAt : null
	};
};

export const getContextTruncationIndex = <T extends MessageLike>(
	messages: T[] = [],
	contextTruncation: ContextTruncationState | null | undefined
) => {
	if (!contextTruncation?.enabled || !contextTruncation.cutoffMessageId) {
		return -1;
	}

	return messages.findIndex((message) => message?.id === contextTruncation.cutoffMessageId);
};

export const isContextTruncationInMessages = <T extends MessageLike>(
	messages: T[] = [],
	contextTruncation: ContextTruncationState | null | undefined
) => getContextTruncationIndex(messages, contextTruncation) !== -1;

export const trimMessagesAfterContextTruncation = <T extends MessageLike>(
	messages: T[] = [],
	contextTruncation: ContextTruncationState | null | undefined
) => {
	const cutoffIndex = getContextTruncationIndex(messages, contextTruncation);
	return cutoffIndex === -1 ? messages : messages.slice(cutoffIndex + 1);
};
