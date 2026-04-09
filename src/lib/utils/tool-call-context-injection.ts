export type ToolCallDetails = {
	id: string;
	done: boolean;
	contextInjectionDisabled: boolean;
};

const escapeRegExp = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

const DETAILS_TAG_REGEX = /<details\s+([^>]*)>/gis;
const TOOL_CALL_TYPE_REGEX = /\btype\s*=\s*"tool_calls"/i;

let lastToolCallDetailsContent = '';
let lastToolCallDetails: ToolCallDetails[] = [];

const getAttributeValue = (attributes: string, attributeName: string) => {
	const match = attributes.match(new RegExp(`\\b${escapeRegExp(attributeName)}="([^"]*)"`, 'i'));
	return match?.[1] ?? null;
};

const updateContextInjectionAttribute = (attributes: string, disabled: boolean) => {
	const normalizedValue = disabled ? 'true' : 'false';

	if (/\bcontext_injection_disabled="/i.test(attributes)) {
		return attributes.replace(
			/\bcontext_injection_disabled="(true|false)"/i,
			`context_injection_disabled="${normalizedValue}"`
		);
	}

	return `${attributes} context_injection_disabled="${normalizedValue}"`;
};

export const getToolCallDetailsFromContent = (content: string): ToolCallDetails[] => {
	if (!content) {
		lastToolCallDetailsContent = '';
		lastToolCallDetails = [];
		return [];
	}

	if (content === lastToolCallDetailsContent) {
		return lastToolCallDetails;
	}

	// Most responses do not contain tool call details; skip full regex scanning in that common case.
	if (!content.includes('<details') || !TOOL_CALL_TYPE_REGEX.test(content)) {
		lastToolCallDetailsContent = content;
		lastToolCallDetails = [];
		return [];
	}

	const toolCalls: ToolCallDetails[] = [];

	content.replace(DETAILS_TAG_REGEX, (match, attributes) => {
		if (!/\btype="tool_calls"/i.test(attributes)) {
			return match;
		}

		toolCalls.push({
			id: getAttributeValue(attributes, 'id') ?? '',
			done: (getAttributeValue(attributes, 'done') ?? '').toLowerCase() === 'true',
			contextInjectionDisabled:
				(getAttributeValue(attributes, 'context_injection_disabled') ?? '').toLowerCase() === 'true'
		});

		return match;
	});

	lastToolCallDetailsContent = content;
	lastToolCallDetails = toolCalls;
	return toolCalls;
};

export const updateToolCallContextInjectionStateInContent = (
	content: string,
	toolCallId: string,
	disabled: boolean
) => {
	if (
		!content ||
		!toolCallId ||
		!content.includes('<details') ||
		!TOOL_CALL_TYPE_REGEX.test(content)
	) {
		return content;
	}

	const toolCallIdPattern = escapeRegExp(toolCallId);

	return content.replace(DETAILS_TAG_REGEX, (match, attributes) => {
		if (!/\btype="tool_calls"/i.test(attributes)) {
			return match;
		}

		if (!new RegExp(`\\bid="${toolCallIdPattern}"`, 'i').test(attributes)) {
			return match;
		}

		return `<details ${updateContextInjectionAttribute(attributes, disabled)}>`;
	});
};

export const updateAllToolCallContextInjectionStatesInContent = (
	content: string,
	disabled: boolean
) => {
	if (!content || !content.includes('<details') || !TOOL_CALL_TYPE_REGEX.test(content)) {
		return content;
	}

	return content.replace(DETAILS_TAG_REGEX, (match, attributes) => {
		if (!/\btype="tool_calls"/i.test(attributes)) {
			return match;
		}

		return `<details ${updateContextInjectionAttribute(attributes, disabled)}>`;
	});
};
