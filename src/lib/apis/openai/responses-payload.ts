type ResponsesRequestBody = Record<string, unknown>;

const NULLABLE_RESPONSE_PARAM_KEYS = [
	'temperature',
	'top_p',
	'stop',
	'verbosity',
	'max_output_tokens'
] as const;

const cleanReasoningConfig = (reasoning: unknown): unknown => {
	if (!reasoning || typeof reasoning !== 'object' || Array.isArray(reasoning)) {
		return reasoning;
	}

	const cleaned = { ...(reasoning as Record<string, unknown>) };
	if (cleaned.effort == null) {
		delete cleaned.effort;
	}
	if (cleaned.summary == null) {
		delete cleaned.summary;
	}

	return Object.keys(cleaned).length > 0 ? cleaned : undefined;
};

export const sanitizeResponsesRequestBody = (body: ResponsesRequestBody): ResponsesRequestBody => {
	const cleaned = { ...body };

	for (const key of NULLABLE_RESPONSE_PARAM_KEYS) {
		if (cleaned[key] == null) {
			delete cleaned[key];
		}
	}

	const reasoning = cleanReasoningConfig(cleaned.reasoning);
	if (reasoning === undefined) {
		delete cleaned.reasoning;
	} else {
		cleaned.reasoning = reasoning;
	}

	Object.keys(cleaned).forEach((key) => cleaned[key] === undefined && delete cleaned[key]);
	return cleaned;
};
