export type ProviderType = 'openai' | 'openai_responses';

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

	if (providerType === 'openai' || providerType === 'openai_responses') {
		return providerType;
	}

	// Legacy migration: Azure OpenAI provider is removed; treat old configs as OpenAI.
	if (providerType === 'azure_openai' || apiConfig?.azure) {
		return 'openai';
	}

	return defaultProvider;
};

export const getEndpointTypeFromProvider = (
	providerType: ProviderType
): 'chat_completions' | 'responses' => {
	return providerType === 'openai_responses' ? 'responses' : 'chat_completions';
};
