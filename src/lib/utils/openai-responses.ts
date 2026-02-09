const normalizeResponsesPart = (role = 'user', part: any = null) => {
	if (!part || typeof part !== 'object') return null;

	if ('type' in part) {
		if (part.type === 'text') {
			return {
				type: role === 'assistant' ? 'output_text' : 'input_text',
				text: part.text ?? ''
			};
		}

		if (part.type === 'image_url') {
			const imageUrl = part.image_url;
			const url = typeof imageUrl === 'string' ? imageUrl : imageUrl?.url;
			if (!url) return null;
			return {
				type: 'input_image',
				image_url: url
			};
		}
	}

	if ('text' in part) {
		return {
			type: role === 'assistant' ? 'output_text' : 'input_text',
			text: part.text ?? ''
		};
	}

	if ('image_url' in part) {
		const imageUrl = part.image_url;
		const url = typeof imageUrl === 'string' ? imageUrl : imageUrl?.url;
		if (!url) return null;
		return {
			type: 'input_image',
			image_url: url
		};
	}

	return null;
};

const toResponsesInput = (messages: any[] = []) => {
	return (messages || [])
		.map((message) => {
			if (!message || typeof message !== 'object') return null;

			if (message.role === 'tool') {
				return {
					type: 'function_call_output',
					call_id: message.tool_call_id,
					output: message.content ?? ''
				};
			}

			if (message.role === 'assistant' && Array.isArray(message.tool_calls) && message.tool_calls.length) {
				const calls = message.tool_calls.map((toolCall) => ({
					type: 'function_call',
					call_id: toolCall.id,
					name: toolCall.function?.name,
					arguments:
						typeof toolCall.function?.arguments === 'string'
							? toolCall.function.arguments
							: JSON.stringify(toolCall.function?.arguments ?? {})
				}));

				if (message.content) {
					calls.push({
						role: 'assistant',
						content: [{ type: 'output_text', text: message.content }]
					});
				}

				return calls;
			}

			if (typeof message.content === 'string') {
				return { role: message.role, content: message.content };
			}

			if (Array.isArray(message.content)) {
				return {
					role: message.role,
					content: message.content
						.map((part) => normalizeResponsesPart(message.role, part))
						.filter(Boolean)
				};
			}

			return null;
		})
		.flat()
		.filter(Boolean);
};

const toResponsesTools = (tools: any[] = []) =>
	(tools || []).map((tool) => {
		if (tool?.type === 'function' && tool.function) {
			return {
				type: 'function',
				name: tool.function.name,
				description: tool.function.description,
				parameters: tool.function.parameters
			};
		}
		return tool;
	});

const normalizeReasoningSummary = (value: any, fallback = 'auto') => {
	if (typeof value !== 'string') return fallback;
	const normalized = value.trim().toLowerCase();
	return ['auto', 'concise', 'detailed'].includes(normalized) ? normalized : fallback;
};

const extractReasoningSummaryFromOutput = (items: any[] = []) =>
	(items ?? [])
		.filter((item) => item?.type === 'reasoning')
		.flatMap((item) => item?.summary ?? [])
		.map((summaryItem) => summaryItem?.text ?? '')
		.filter((text) => typeof text === 'string' && text.trim() !== '')
		.join('\n');

const extractReasoningSummaryFromRoot = (reasoning: any = null) => {
	const value = Array.isArray(reasoning?.summary)
		? reasoning.summary.map((summary: any) => summary?.text ?? '').join('\n')
		: reasoning?.summary ?? '';

	const normalized = typeof value === 'string' ? value.trim().toLowerCase() : '';
	if (['auto', 'concise', 'detailed'].includes(normalized)) {
		return '';
	}

	return typeof value === 'string' ? value : '';
};

export const toResponsesPayload = (formData: any = {}) => {
	const payload: any = {
		model: formData.model,
		input: toResponsesInput(formData.messages ?? []),
		stream: Boolean(formData.stream),
		metadata: formData.metadata
	};

	if (Array.isArray(formData.tools) && formData.tools.length) {
		payload.tools = toResponsesTools(formData.tools);
	}

	if (formData.tool_choice) {
		if (typeof formData.tool_choice === 'object' && formData.tool_choice?.function?.name) {
			payload.tool_choice = {
				type: 'function',
				name: formData.tool_choice.function.name
			};
		} else {
			payload.tool_choice = formData.tool_choice;
		}
	}

	if (formData.parallel_tool_calls !== undefined) {
		payload.parallel_tool_calls = formData.parallel_tool_calls;
	}

	const params = formData.params ?? {};
	const summaryCandidate = formData?.reasoning?.summary ?? params?.summary ?? formData?.summary ?? 'auto';
	const effortCandidate = formData?.reasoning?.effort ?? params?.reasoning_effort;
	payload.reasoning = {
		summary: normalizeReasoningSummary(summaryCandidate, 'auto'),
		...(effortCandidate ? { effort: effortCandidate } : {})
	};

	for (const key of ['temperature', 'top_p', 'stop', 'store', 'truncation', 'text', 'user']) {
		if (params[key] !== undefined) {
			payload[key] = params[key];
		}
	}

	const maxOutputTokens = params.max_completion_tokens ?? params.max_tokens;
	if (maxOutputTokens !== undefined) {
		payload.max_output_tokens = maxOutputTokens;
	}

	return payload;
};

export const responsesToChatCompletion = (response: any = {}) => {
	const output = response.output ?? [];
	const toolCalls = output
		.filter((item: any) => item?.type === 'function_call')
		.map((item: any) => ({
			id: item.call_id ?? item.id,
			type: 'function',
			function: {
				name: item.name,
				arguments: typeof item.arguments === 'string' ? item.arguments : JSON.stringify(item.arguments ?? {})
			}
		}));

	const text = output
		.filter((item: any) => item?.type === 'message')
		.flatMap((item: any) => item.content ?? [])
		.filter((part: any) => part?.type === 'output_text')
		.map((part: any) => part.text ?? '')
		.join('');

	const reasoningSummary =
		extractReasoningSummaryFromOutput(output) || extractReasoningSummaryFromRoot(response.reasoning);

	return {
		id: response.id,
		model: response.model,
		choices: [
			{
				index: 0,
				message: {
					role: 'assistant',
					content: text,
					...(toolCalls.length ? { tool_calls: toolCalls } : {}),
					...(reasoningSummary ? { reasoning_content: reasoningSummary } : {})
				}
			}
		],
		usage: response.usage
			? {
					prompt_tokens: response.usage.input_tokens ?? 0,
					completion_tokens: response.usage.output_tokens ?? 0,
					total_tokens: response.usage.total_tokens ?? 0
				}
			: undefined
	};
};

export const streamResponsesSSEToChatLines = async (
	responseBody: ReadableStream<Uint8Array>,
	emitLine: (line: string) => void
) => {
	const reader = responseBody.getReader();
	const decoder = new TextDecoder();
	let buffer = '';

	const state: any = {
		functionCalls: {},
		callIndexes: {},
		emittedCallIds: new Set(),
		textEmitted: false
	};

	const emitDelta = (delta: any) => {
		emitLine('data: ' + JSON.stringify({ choices: [{ delta }] }));
	};

	while (true) {
		const { done, value } = await reader.read();
		if (done) {
			break;
		}

		buffer += decoder.decode(value, { stream: true });

		while (buffer.includes('\n\n')) {
			const splitIdx = buffer.indexOf('\n\n');
			const eventBlock = buffer.slice(0, splitIdx);
			buffer = buffer.slice(splitIdx + 2);

			if (!eventBlock.trim()) continue;

			let eventType = null;
			const dataLines = [];
			for (const line of eventBlock.split('\n')) {
				if (line.startsWith('event:')) {
					eventType = line.slice('event:'.length).trim();
				} else if (line.startsWith('data:')) {
					dataLines.push(line.slice('data:'.length).trim());
				}
			}

			if (!dataLines.length) continue;
			const dataStr = dataLines.join('\n');
			if (dataStr === '[DONE]') {
				emitLine('data: [DONE]');
				return;
			}

			let payload = null;
			try {
				payload = JSON.parse(dataStr);
			} catch {
				continue;
			}

			const type = eventType ?? payload?.type;
			if (type === 'response.output_text.delta') {
				const delta = payload?.delta ?? '';
				if (delta) {
					state.textEmitted = true;
					emitDelta({ content: delta });
				}
			}

			if (['response.function_call_arguments.delta', 'response.function_call.delta'].includes(type)) {
				const callId = payload?.call_id ?? payload?.item_id;
				if (callId) {
					const callState = state.functionCalls[callId] ?? {
						name: payload?.name ?? '',
						arguments: ''
					};
					if (payload?.name) callState.name = payload.name;
					if (typeof payload?.delta === 'string') callState.arguments += payload.delta;
					state.functionCalls[callId] = callState;
				}
			}

			if (type === 'response.output_item.done') {
				const item = payload?.item ?? {};
				if (item?.type === 'function_call') {
					const callId = item.call_id ?? item.id;
					if (callId && !state.emittedCallIds.has(callId)) {
						const callState = state.functionCalls[callId] ?? {
							name: item.name ?? '',
							arguments: ''
						};
						if (item.name) callState.name = item.name;
						if (typeof item.arguments === 'string') callState.arguments = item.arguments;
						state.functionCalls[callId] = callState;

						const index = state.callIndexes[callId] ?? Object.keys(state.callIndexes).length;
						state.callIndexes[callId] = index;
						emitDelta({
							tool_calls: [
								{
									index,
									id: callId,
									type: 'function',
									function: {
										name: callState.name,
										arguments: callState.arguments || '{}'
									}
								}
							]
						});
						state.emittedCallIds.add(callId);
					}
				}
			}

			if (type === 'response.completed') {
				const response = payload?.response ?? {};
				const summary =
					extractReasoningSummaryFromOutput(response?.output ?? []) ||
					extractReasoningSummaryFromRoot(response?.reasoning ?? null);
				if (summary) {
					emitDelta({ reasoning_content: summary });
				}

				if (!state.textEmitted) {
					const text = (response?.output ?? [])
						.filter((item: any) => item?.type === 'message')
						.flatMap((item: any) => item?.content ?? [])
						.filter((part: any) => part?.type === 'output_text')
						.map((part: any) => part?.text ?? '')
						.join('');
					if (text) emitDelta({ content: text });
				}

				if (response?.usage) {
					emitLine(
						'data: ' +
							JSON.stringify({
								usage: {
									prompt_tokens: response.usage.input_tokens ?? 0,
									completion_tokens: response.usage.output_tokens ?? 0,
									total_tokens: response.usage.total_tokens ?? 0
								}
							})
					);
				}

				emitLine('data: [DONE]');
				return;
			}
		}
	}

	emitLine('data: [DONE]');
};
