type ExportableMessage = {
	content?: string | null;
	sources?: Array<unknown> | null;
};

type BuildMessageExportTextOptions = {
	removeDetails?: boolean;
	excludeCitations?: boolean;
	includeWatermark?: boolean;
	watermark?: string;
};

const removeAllDetailsTags = (content: string): string => {
	return content.replace(/<details[^>]*>.*?<\/details>/gis, '');
};

const parseCitationGroup = (
	content: string,
	start: number
): { end: number; ids: number[] } | null => {
	let openLength = 0;
	let closeToken = '';

	if (content.slice(start, start + 2) === '[[') {
		openLength = 2;
		closeToken = ']]';
	} else if (content.slice(start, start + 2) === '【【') {
		openLength = 2;
		closeToken = '】】';
	}

	if (openLength === 0) {
		return null;
	}

	if (closeToken === ']]' && content[start + openLength] === '^') {
		return null;
	}

	const closeIndex = content.indexOf(closeToken, start + openLength);
	if (closeIndex === -1) {
		return null;
	}

	const raw = content.slice(start + openLength, closeIndex);
	if (!/^\d[\d,\s]*$/.test(raw)) {
		return null;
	}

	const ids = raw
		.split(',')
		.map((value) => Number.parseInt(value.trim(), 10))
		.filter((value) => Number.isFinite(value));

	if (ids.length === 0) {
		return null;
	}

	return {
		end: closeIndex + closeToken.length,
		ids
	};
};

const parseCitationSequence = (
	content: string,
	start: number
): { end: number; ids: number[] } | null => {
	let cursor = start;
	let ids: number[] = [];
	let groupCount = 0;

	while (cursor < content.length) {
		const group = parseCitationGroup(content, cursor);
		if (!group) {
			break;
		}

		ids = ids.concat(group.ids);
		cursor = group.end;
		groupCount += 1;
	}

	if (groupCount === 0) {
		return null;
	}

	return { end: cursor, ids };
};

const normalizeCitationWhitespace = (content: string): string => {
	return content
		.replace(/[ \t]{2,}/g, ' ')
		.replace(/[ \t]+([,.;:!?])/g, '$1')
		.replace(/[ \t]+\n/g, '\n')
		.replace(/\n{3,}/g, '\n\n');
};

export const getCitationSourceCount = (sources: ExportableMessage['sources']): number => {
	if (!Array.isArray(sources) || sources.length === 0) {
		return 0;
	}

	return sources.reduce((total, source) => {
		if (!source || typeof source !== 'object') {
			return total + 1;
		}

		const sourceObject = source as Record<string, unknown>;
		const documentList = sourceObject.document;
		if (Array.isArray(documentList)) {
			return total + documentList.length;
		}

		const metadataList = sourceObject.metadata;
		if (Array.isArray(metadataList)) {
			return total + metadataList.length;
		}

		return total + 1;
	}, 0);
};

export const stripCitationTokens = (content: string, sourceCount: number): string => {
	if (!content || sourceCount <= 0) {
		return content;
	}

	let cursor = 0;
	let output = '';

	while (cursor < content.length) {
		const sequence = parseCitationSequence(content, cursor);
		if (!sequence) {
			output += content[cursor];
			cursor += 1;
			continue;
		}

		const removable = sequence.ids.every((id) => id >= 1 && id <= sourceCount);
		if (removable) {
			cursor = sequence.end;
			continue;
		}

		output += content.slice(cursor, sequence.end);
		cursor = sequence.end;
	}

	return normalizeCitationWhitespace(output);
};

export const buildMessageExportText = (
	message: ExportableMessage,
	options: BuildMessageExportTextOptions = {}
): string => {
	const {
		removeDetails = true,
		excludeCitations = true,
		includeWatermark = false,
		watermark = ''
	} = options;

	let text = `${message?.content ?? ''}`;

	if (removeDetails) {
		text = removeAllDetailsTags(text);
	}

	if (excludeCitations) {
		text = stripCitationTokens(text, getCitationSourceCount(message?.sources));
	}

	if (includeWatermark && watermark.trim() !== '') {
		text = `${text}\n\n${watermark}`;
	}

	return text;
};
