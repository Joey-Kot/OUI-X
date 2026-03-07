export type ToolCallDetails = {
	id: string;
	done: boolean;
	contextInjectionDisabled: boolean;
};

const escapeRegExp = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

const DETAILS_TAG_REGEX = /<details\s+([^>]*)>/gis;

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

	return toolCalls;
};

export const updateToolCallContextInjectionStateInContent = (
	content: string,
	toolCallId: string,
	disabled: boolean
) => {
	if (!content || !toolCallId) {
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
	if (!content) {
		return content;
	}

	return content.replace(DETAILS_TAG_REGEX, (match, attributes) => {
		if (!/\btype="tool_calls"/i.test(attributes)) {
			return match;
		}

		return `<details ${updateContextInjectionAttribute(attributes, disabled)}>`;
	});
};
