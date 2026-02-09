export type ProviderType = 'openai' | 'azure_openai' | 'openai_responses';

export const getProviderTypeFromModel = (model: any = {}, directConnections: any = null): ProviderType => {
	const defaultProvider: ProviderType = 'openai_responses';

	if (!model?.direct) {
		return defaultProvider;
	}

	const urlIdx = Number(model?.urlIdx);
	if (!Number.isInteger(urlIdx) || urlIdx < 0) {
		return defaultProvider;
	}

	const apiConfig = directConnections?.OPENAI_API_CONFIGS?.[urlIdx] ?? {};
	const providerType = apiConfig?.provider_type;

	if (providerType === 'openai' || providerType === 'azure_openai' || providerType === 'openai_responses') {
		return providerType;
	}

	if (apiConfig?.azure) {
		return 'azure_openai';
	}

	return defaultProvider;
};

export const getEndpointTypeFromProvider = (
	providerType: ProviderType
): 'chat_completions' | 'responses' => {
	return providerType === 'openai_responses' ? 'responses' : 'chat_completions';
};
