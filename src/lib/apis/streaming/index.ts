import { EventSourceParserStream } from 'eventsource-parser/stream';
import type { ParsedEvent } from 'eventsource-parser';

type TextStreamUpdate = {
	done: boolean;
	value: string;
	reasoning?: string;
	toolCalls?: Array<Record<string, unknown>>;
	rawEvent?: Record<string, unknown>;
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	sources?: any;
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	selectedModelId?: any;
	error?: any;
	usage?: ResponseUsage;
};

type ResponseUsage = {
	/** Prompt/input token count, including images and tools if any */
	prompt_tokens: number;
	/** Generated/output token count */
	completion_tokens: number;
	/** Total token count; may come from provider or be derived as prompt + completion */
	total_tokens: number;
	/** Additional provider-specific usage fields, e.g. input_tokens_details/output_tokens_details */
	[other: string]: unknown;
};

// createOpenAITextStream takes a responseBody with a SSE response,
// and returns an async generator that emits delta updates with large deltas chunked into random sized chunks
export async function createOpenAITextStream(
	responseBody: ReadableStream<Uint8Array>,
	splitLargeDeltas: boolean
): Promise<AsyncGenerator<TextStreamUpdate>> {
	const eventStream = responseBody
		.pipeThrough(new TextDecoderStream())
		.pipeThrough(new EventSourceParserStream())
		.getReader();
	let iterator = openAIStreamToIterator(eventStream);
	if (splitLargeDeltas) {
		iterator = streamLargeDeltasAsRandomChunks(iterator);
	}
	return iterator;
}

async function* openAIStreamToIterator(
	reader: ReadableStreamDefaultReader<ParsedEvent>
): AsyncGenerator<TextStreamUpdate> {
	const responsesState: {
		textEmitted: boolean;
		functionCalls: Record<string, { name: string; arguments: string }>;
		emittedCallIds: Set<string>;
		callIndexes: Record<string, number>;
		reasoningSummaryStreamed: boolean;
		reasoningSummaryCurrentText: string;
		reasoningSummaryEmittedLength: number;
		reasoningSummaryPendingSeparator: boolean;
	} = {
		textEmitted: false,
		functionCalls: {},
		emittedCallIds: new Set(),
		callIndexes: {},
		reasoningSummaryStreamed: false,
		reasoningSummaryCurrentText: '',
		reasoningSummaryEmittedLength: 0,
		reasoningSummaryPendingSeparator: false
	};

	const normalizeResponsesUsage = (usage: any) => {
		if (!usage || typeof usage !== 'object' || Array.isArray(usage)) return null;
		const prompt_tokens = typeof usage.input_tokens === 'number' ? usage.input_tokens : 0;
		const completion_tokens = typeof usage.output_tokens === 'number' ? usage.output_tokens : 0;
		const total_tokens =
			typeof usage.total_tokens === 'number' ? usage.total_tokens : prompt_tokens + completion_tokens;
		return {
			...usage,
			prompt_tokens,
			completion_tokens,
			total_tokens
		};
	};

	const extractResponsesFallbackText = (response: any) => {
		const output = Array.isArray(response?.output) ? response.output : [];
		return output
			.filter((item: any) => item?.type === 'message')
			.flatMap((item: any) => item?.content ?? [])
			.filter((part: any) => part?.type === 'output_text')
			.map((part: any) => part?.text ?? '')
			.join('');
	};

	const extractResponsesReasoning = (response: any) => {
		const output = Array.isArray(response?.output) ? response.output : [];
		return output
			.filter((item: any) => item?.type === 'reasoning')
			.flatMap((item: any) => item?.summary ?? [])
			.map((summary: any) => summary?.text ?? '')
			.filter((text: any) => typeof text === 'string' && text.trim())
			.join('\n');
	};

	while (true) {
		const { value, done } = await reader.read();
		if (done) {
			yield { done: true, value: '' };
			break;
		}
		if (!value) {
			continue;
		}
		const eventName = (value as any).event || '';
		const data = value.data;
		if (data.startsWith('[DONE]')) {
			yield { done: true, value: '' };
			break;
		}

		try {
			const parsedData = JSON.parse(data);
			console.log(parsedData);

			if (parsedData.error) {
				yield { done: true, value: '', error: parsedData.error };
				break;
			}

			const responsesEventType = eventName || parsedData?.type;
			const isResponsesEvent =
				typeof responsesEventType === 'string' && responsesEventType.startsWith('response.');
			if (isResponsesEvent) {
				if (responsesEventType === 'response.output_text.delta') {
					const delta = parsedData?.delta ?? '';
					if (delta) {
						responsesState.textEmitted = true;
						yield { done: false, value: delta, rawEvent: parsedData };
					}
					continue;
				}

				if (responsesEventType === 'response.reasoning_summary_part.added') {
					responsesState.reasoningSummaryCurrentText = '';
					responsesState.reasoningSummaryEmittedLength = 0;
					continue;
				}

				if (responsesEventType === 'response.reasoning_summary_text.delta') {
					const delta = parsedData?.delta;
					if (typeof delta === 'string' && delta) {
						responsesState.reasoningSummaryCurrentText += delta;
						responsesState.reasoningSummaryEmittedLength += delta.length;

						let emitted = delta;
						if (
							responsesState.reasoningSummaryPendingSeparator &&
							responsesState.reasoningSummaryStreamed
						) {
							emitted = `\n${emitted}`;
							responsesState.reasoningSummaryPendingSeparator = false;
						}

						responsesState.reasoningSummaryStreamed = true;
						yield { done: false, value: '', reasoning: emitted, rawEvent: parsedData };
					}
					continue;
				}

				if (responsesEventType === 'response.reasoning_summary_text.done') {
					const finalText =
						typeof parsedData?.text === 'string'
							? parsedData.text
							: responsesState.reasoningSummaryCurrentText;

					let suffix = '';
					if (finalText) {
						if (responsesState.reasoningSummaryEmittedLength <= 0) {
							suffix = finalText;
						} else if (finalText.length > responsesState.reasoningSummaryEmittedLength) {
							suffix = finalText.slice(responsesState.reasoningSummaryEmittedLength);
						}
					}

					responsesState.reasoningSummaryCurrentText = finalText ?? '';
					responsesState.reasoningSummaryEmittedLength = finalText?.length ?? 0;

					if (suffix) {
						let emitted = suffix;
						if (
							responsesState.reasoningSummaryPendingSeparator &&
							responsesState.reasoningSummaryStreamed
						) {
							emitted = `\n${emitted}`;
							responsesState.reasoningSummaryPendingSeparator = false;
						}

						responsesState.reasoningSummaryStreamed = true;
						yield { done: false, value: '', reasoning: emitted, rawEvent: parsedData };
					}
					continue;
				}

				if (responsesEventType === 'response.reasoning_summary_part.done') {
					const hadCurrentPartOutput =
						responsesState.reasoningSummaryCurrentText.length > 0 ||
						responsesState.reasoningSummaryEmittedLength > 0;
					if (hadCurrentPartOutput && responsesState.reasoningSummaryStreamed) {
						responsesState.reasoningSummaryPendingSeparator = true;
					}

					responsesState.reasoningSummaryCurrentText = '';
					responsesState.reasoningSummaryEmittedLength = 0;
					continue;
				}

				if (
					responsesEventType === 'response.function_call_arguments.delta' ||
					responsesEventType === 'response.function_call.delta'
				) {
					const callId = parsedData?.call_id ?? parsedData?.item_id;
					if (callId) {
						const call = responsesState.functionCalls[callId] ?? {
							name: parsedData?.name ?? '',
							arguments: ''
						};
						if (parsedData?.name) call.name = parsedData.name;
						if (typeof parsedData?.delta === 'string') call.arguments += parsedData.delta;
						responsesState.functionCalls[callId] = call;
					}
					yield { done: false, value: '', rawEvent: parsedData };
					continue;
				}

				if (responsesEventType === 'response.output_item.done') {
					const item = parsedData?.item ?? {};
					if (item?.type === 'function_call') {
						const callId = item?.call_id ?? item?.id;
						if (callId && !responsesState.emittedCallIds.has(callId)) {
							const call = responsesState.functionCalls[callId] ?? {
								name: item?.name ?? '',
								arguments: typeof item?.arguments === 'string' ? item.arguments : '{}'
							};
							if (item?.name) call.name = item.name;
							if (typeof item?.arguments === 'string') call.arguments = item.arguments;
							responsesState.functionCalls[callId] = call;
							const index =
								responsesState.callIndexes[callId] ?? Object.keys(responsesState.callIndexes).length;
							responsesState.callIndexes[callId] = index;
							responsesState.emittedCallIds.add(callId);

							yield {
								done: false,
								value: '',
								toolCalls: [
									{
										index,
										id: callId,
										type: 'function',
										function: {
											name: call.name,
											arguments: call.arguments || '{}'
										}
									}
								],
								rawEvent: parsedData
							};
						}
					}
					continue;
				}

				if (responsesEventType === 'response.completed') {
					const response = parsedData?.response ?? {};
					if (!responsesState.reasoningSummaryStreamed) {
						const reasoning = extractResponsesReasoning(response);
						if (reasoning) {
							yield { done: false, value: '', reasoning, rawEvent: parsedData };
						}
					}

					if (!responsesState.textEmitted) {
						const fallbackText = extractResponsesFallbackText(response);
						if (fallbackText) {
							yield { done: false, value: fallbackText, rawEvent: parsedData };
						}
					}

					const usage = normalizeResponsesUsage(response?.usage);
					if (usage) {
						yield { done: false, value: '', usage, rawEvent: parsedData };
					}
					yield { done: true, value: '', rawEvent: parsedData };
					break;
				}

				const usage = normalizeResponsesUsage(parsedData?.usage);
				if (usage) {
					yield { done: false, value: '', usage, rawEvent: parsedData };
				} else {
					yield { done: false, value: '', rawEvent: parsedData };
				}
				continue;
			}

			if (parsedData.sources) {
				yield { done: false, value: '', sources: parsedData.sources };
				continue;
			}

			if (parsedData.selected_model_id) {
				yield { done: false, value: '', selectedModelId: parsedData.selected_model_id };
				continue;
			}

			if (parsedData.usage) {
				yield { done: false, value: '', usage: parsedData.usage };
				continue;
			}

			yield {
				done: false,
				value: parsedData.choices?.[0]?.delta?.content ?? ''
			};
		} catch (e) {
			console.error('Error extracting delta from SSE event:', e);
		}
	}
}

// streamLargeDeltasAsRandomChunks will chunk large deltas (length > 5) into random sized chunks between 1-3 characters
// This is to simulate a more fluid streaming, even though some providers may send large chunks of text at once
async function* streamLargeDeltasAsRandomChunks(
	iterator: AsyncGenerator<TextStreamUpdate>
): AsyncGenerator<TextStreamUpdate> {
	for await (const textStreamUpdate of iterator) {
		if (textStreamUpdate.done) {
			yield textStreamUpdate;
			return;
		}

		if (textStreamUpdate.error) {
			yield textStreamUpdate;
			continue;
		}
		if (textStreamUpdate.reasoning || textStreamUpdate.toolCalls || textStreamUpdate.rawEvent) {
			yield textStreamUpdate;
			continue;
		}
		if (textStreamUpdate.sources) {
			yield textStreamUpdate;
			continue;
		}
		if (textStreamUpdate.selectedModelId) {
			yield textStreamUpdate;
			continue;
		}
		if (textStreamUpdate.usage) {
			yield textStreamUpdate;
			continue;
		}

		let content = textStreamUpdate.value;
		if (content.length < 5) {
			yield { done: false, value: content };
			continue;
		}
		while (content != '') {
			const chunkSize = Math.min(Math.floor(Math.random() * 3) + 1, content.length);
			const chunk = content.slice(0, chunkSize);
			yield { done: false, value: chunk };
			// Do not sleep if the tab is hidden
			// Timers are throttled to 1s in hidden tabs
			if (document?.visibilityState !== 'hidden') {
				await sleep(5);
			}
			content = content.slice(chunkSize);
		}
	}
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
