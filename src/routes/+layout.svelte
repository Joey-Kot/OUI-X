<script>
	import { io } from 'socket.io-client';
	import { spring } from 'svelte/motion';
	import PyodideWorker from '$lib/workers/pyodide.worker?worker';
	import { Toaster, toast } from 'svelte-sonner';

	let loadingProgress = spring(0, {
		stiffness: 0.05
	});

	import { onMount, tick, setContext, onDestroy } from 'svelte';
	import {
		config,
		user,
		settings,
		theme,
		WEBUI_NAME,
		WEBUI_VERSION,
		WEBUI_DEPLOYMENT_ID,
		mobile,
		socket,
		chatId,
		chats,
		currentChatPage,
		tags,
		temporaryChatEnabled,
		isLastActiveTab,
		isApp,
		appInfo,
		playingNotificationSound,
		channels,
		channelId
	} from '$lib/stores';
	import { goto } from '$app/navigation';
	import { page } from '$app/stores';
	import { beforeNavigate } from '$app/navigation';
	import { updated } from '$app/state';

	import i18n, { initI18n, getLanguages, changeLanguage } from '$lib/i18n';

	import '../tailwind.css';
	import '../app.css';
	import 'tippy.js/dist/tippy.css';

	import { getBackendConfig, getVersion } from '$lib/apis';
	import { getSessionUser, userSignOut } from '$lib/apis/auths';
	import { getAllTags, getChatList } from '$lib/apis/chats';
	import { chatCompletion, responsesCompletion } from '$lib/apis/openai';
	import { getEndpointTypeFromProvider, getProviderTypeFromModel } from '$lib/utils/provider-routing';
	import {
		responsesToChatCompletion as responsesToChatCompletionUtil,
		streamResponsesSSEToChatLines as streamResponsesSSEToChatLinesUtil,
		toResponsesPayload as toResponsesPayloadUtil
	} from '$lib/utils/openai-responses';

	import { WEBUI_API_BASE_URL, WEBUI_BASE_URL, WEBUI_HOSTNAME } from '$lib/constants';
	import { bestMatchingLanguage } from '$lib/utils';
	import { setTextScale } from '$lib/utils/text-scale';

	import NotificationToast from '$lib/components/NotificationToast.svelte';
	import AppSidebar from '$lib/components/app/AppSidebar.svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import { getUserSettings } from '$lib/apis/users';
	import dayjs from 'dayjs';
	import { getChannels } from '$lib/apis/channels';

	const unregisterServiceWorkers = async () => {
		if ('serviceWorker' in navigator) {
			try {
				const registrations = await navigator.serviceWorker.getRegistrations();
				await Promise.all(registrations.map((r) => r.unregister()));
				return true;
			} catch (error) {
				console.error('Error unregistering service workers:', error);
				return false;
			}
		}
		return false;
	};

	// handle frontend updates (https://svelte.dev/docs/kit/configuration#version)
	beforeNavigate(async ({ willUnload, to }) => {
		if (updated.current && !willUnload && to?.url) {
			await unregisterServiceWorkers();
			location.href = to.url.href;
		}
	});

	setContext('i18n', i18n);

	const bc = new BroadcastChannel('active-tab-channel');

	let loaded = false;
	let tokenTimer = null;

	let showRefresh = false;

	let heartbeatInterval = null;

	const BREAKPOINT = 768;

	const setupSocket = async (enableWebsocket) => {
		const _socket = io(`${WEBUI_BASE_URL}` || undefined, {
			reconnection: true,
			reconnectionDelay: 1000,
			reconnectionDelayMax: 5000,
			randomizationFactor: 0.5,
			path: '/ws/socket.io',
			transports: enableWebsocket ? ['websocket'] : ['polling', 'websocket'],
			auth: { token: localStorage.token }
		});
		await socket.set(_socket);

		_socket.on('connect_error', (err) => {
			console.log('connect_error', err);
		});

		_socket.on('connect', async () => {
			console.log('connected', _socket.id);
			const res = await getVersion(localStorage.token);

			const deploymentId = res?.deployment_id ?? null;
			const version = res?.version ?? null;

			if (version !== null || deploymentId !== null) {
				if (
					($WEBUI_VERSION !== null && version !== $WEBUI_VERSION) ||
					($WEBUI_DEPLOYMENT_ID !== null && deploymentId !== $WEBUI_DEPLOYMENT_ID)
				) {
					await unregisterServiceWorkers();
					location.href = location.href;
					return;
				}
			}

			// Send heartbeat every 30 seconds
			heartbeatInterval = setInterval(() => {
				if (_socket.connected) {
					console.log('Sending heartbeat');
					_socket.emit('heartbeat', {});
				}
			}, 30000);

			if (deploymentId !== null) {
				WEBUI_DEPLOYMENT_ID.set(deploymentId);
			}

			if (version !== null) {
				WEBUI_VERSION.set(version);
			}

			console.log('version', version);

			if (localStorage.getItem('token')) {
				// Emit user-join event with auth token
				_socket.emit('user-join', { auth: { token: localStorage.token } });
			} else {
				console.warn('No token found in localStorage, user-join event not emitted');
			}
		});

		_socket.on('reconnect_attempt', (attempt) => {
			console.log('reconnect_attempt', attempt);
		});

		_socket.on('reconnect_failed', () => {
			console.log('reconnect_failed');
		});

		_socket.on('disconnect', (reason, details) => {
			console.log(`Socket ${_socket.id} disconnected due to ${reason}`);

			if (heartbeatInterval) {
				clearInterval(heartbeatInterval);
				heartbeatInterval = null;
			}

			if (details) {
				console.log('Additional details:', details);
			}
		});
	};

	const executePythonAsWorker = async (id, code, cb) => {
		let result = null;
		let stdout = null;
		let stderr = null;

		let executing = true;
		let packages = [
			/\bimport\s+requests\b|\bfrom\s+requests\b/.test(code) ? 'requests' : null,
			/\bimport\s+bs4\b|\bfrom\s+bs4\b/.test(code) ? 'beautifulsoup4' : null,
			/\bimport\s+numpy\b|\bfrom\s+numpy\b/.test(code) ? 'numpy' : null,
			/\bimport\s+pandas\b|\bfrom\s+pandas\b/.test(code) ? 'pandas' : null,
			/\bimport\s+matplotlib\b|\bfrom\s+matplotlib\b/.test(code) ? 'matplotlib' : null,
			/\bimport\s+seaborn\b|\bfrom\s+seaborn\b/.test(code) ? 'seaborn' : null,
			/\bimport\s+sklearn\b|\bfrom\s+sklearn\b/.test(code) ? 'scikit-learn' : null,
			/\bimport\s+scipy\b|\bfrom\s+scipy\b/.test(code) ? 'scipy' : null,
			/\bimport\s+re\b|\bfrom\s+re\b/.test(code) ? 'regex' : null,
			/\bimport\s+seaborn\b|\bfrom\s+seaborn\b/.test(code) ? 'seaborn' : null,
			/\bimport\s+sympy\b|\bfrom\s+sympy\b/.test(code) ? 'sympy' : null,
			/\bimport\s+tiktoken\b|\bfrom\s+tiktoken\b/.test(code) ? 'tiktoken' : null,
			/\bimport\s+pytz\b|\bfrom\s+pytz\b/.test(code) ? 'pytz' : null
		].filter(Boolean);

		const pyodideWorker = new PyodideWorker();

		pyodideWorker.postMessage({
			id: id,
			code: code,
			packages: packages
		});

		setTimeout(() => {
			if (executing) {
				executing = false;
				stderr = 'Execution Time Limit Exceeded';
				pyodideWorker.terminate();

				if (cb) {
					cb(
						JSON.parse(
							JSON.stringify(
								{
									stdout: stdout,
									stderr: stderr,
									result: result
								},
								(_key, value) => (typeof value === 'bigint' ? value.toString() : value)
							)
						)
					);
				}
			}
		}, 60000);

		pyodideWorker.onmessage = (event) => {
			console.log('pyodideWorker.onmessage', event);
			const { id, ...data } = event.data;

			console.log(id, data);

			data['stdout'] && (stdout = data['stdout']);
			data['stderr'] && (stderr = data['stderr']);
			data['result'] && (result = data['result']);

			if (cb) {
				cb(
					JSON.parse(
						JSON.stringify(
							{
								stdout: stdout,
								stderr: stderr,
								result: result
							},
							(_key, value) => (typeof value === 'bigint' ? value.toString() : value)
						)
					)
				);
			}

			executing = false;
		};

		pyodideWorker.onerror = (event) => {
			console.log('pyodideWorker.onerror', event);

			if (cb) {
				cb(
					JSON.parse(
						JSON.stringify(
							{
								stdout: stdout,
								stderr: stderr,
								result: result
							},
							(_key, value) => (typeof value === 'bigint' ? value.toString() : value)
						)
					)
				);
			}
			executing = false;
		};
	};

	const toResponsesInput = (messages = []) => {
		const normalizeResponsesPart = (role, part) => {
			if (!part || typeof part !== 'object') return null;
			const partType = part.type;

			if (partType === 'text') {
				return {
					type: role === 'assistant' ? 'output_text' : 'input_text',
					text: part.text ?? part.content ?? ''
				};
			}

			if (partType === 'image_url') {
				return {
					type: 'input_image',
					image_url: part.image_url?.url ?? part.image_url
				};
			}

			if (partType === 'refusal') {
				return role === 'assistant' ? part : null;
			}

			if (partType === 'input_text') {
				return role === 'assistant'
					? { type: 'output_text', text: part.text ?? '' }
					: part;
			}

			if (partType === 'output_text') {
				return role === 'assistant'
					? part
					: { type: 'input_text', text: part.text ?? '' };
			}

			if (['input_image', 'input_file'].includes(partType)) {
				return ['user', 'system', 'developer'].includes(role) ? part : null;
			}

			return null;
		};

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
							.map((part) => {
								return normalizeResponsesPart(message.role, part);
							})
							.filter(Boolean)
					};
				}

				return null;
			})
			.flat()
			.filter(Boolean);
	};

	const toResponsesTools = (tools = []) =>
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

	const toResponsesPayload = (formData = {}) => {
		const normalizeReasoningSummary = (value, fallback = 'auto') => {
			if (typeof value !== 'string') return fallback;
			const normalized = value.trim().toLowerCase();
			return ['auto', 'concise', 'detailed'].includes(normalized) ? normalized : fallback;
		};


		const normalizeReasoningEffortParam = (value) => {
			if (typeof value !== 'string') return undefined;
			const normalized = value.trim().toLowerCase();
			if (!normalized) return undefined;
			return normalized;
		};

		const normalizeVerbosityParam = (value) => {
			if (typeof value !== 'string') return undefined;
			const normalized = value.trim().toLowerCase();
			if (!normalized || normalized === 'none') return undefined;
			return normalized;
		};

		const payload = {
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
		const summaryCandidate =
			formData?.reasoning?.summary ?? params?.summary ?? formData?.summary ?? 'auto';
		const effortCandidate = normalizeReasoningEffortParam(
			formData?.reasoning?.effort ?? params?.reasoning_effort
		);
		const verbosityCandidate = normalizeVerbosityParam(formData?.verbosity ?? params?.verbosity);
		payload.reasoning = {
			summary: normalizeReasoningSummary(summaryCandidate, 'auto'),
			...(effortCandidate ? { effort: effortCandidate } : {})
		};

		if (verbosityCandidate) {
			payload.verbosity = verbosityCandidate;
		}

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

	const responsesToChatCompletion = (response = {}) => {
		const output = response.output ?? [];
		const extractReasoningSummaryFromOutput = (items = []) =>
			(items ?? [])
				.filter((item) => item?.type === 'reasoning')
				.flatMap((item) => item?.summary ?? [])
				.map((summaryItem) => summaryItem?.text ?? '')
				.filter((text) => typeof text === 'string' && text.trim() !== '')
				.join('\n');

		const extractReasoningSummaryFromRoot = (reasoning = null) => {
			const value = Array.isArray(reasoning?.summary)
				? reasoning.summary.map((s) => s?.text ?? '').join('\n')
				: reasoning?.summary ?? '';

			const normalized = typeof value === 'string' ? value.trim().toLowerCase() : '';
			// Avoid showing config-like summary values as reasoning content.
			if (['auto', 'concise', 'detailed'].includes(normalized)) {
				return '';
			}

			return typeof value === 'string' ? value : '';
		};
		const tool_calls = output
			.filter((item) => item?.type === 'function_call')
			.map((item) => ({
				id: item.call_id ?? item.id,
				type: 'function',
				function: {
					name: item.name,
					arguments: typeof item.arguments === 'string' ? item.arguments : JSON.stringify(item.arguments ?? {})
				}
			}));

		const text = output
			.filter((item) => item?.type === 'message')
			.flatMap((item) => item.content ?? [])
			.filter((part) => part?.type === 'output_text')
			.map((part) => part.text ?? '')
			.join('');

		const reasoningSummary =
			extractReasoningSummaryFromOutput(output) ||
			extractReasoningSummaryFromRoot(response.reasoning);

		return {
			id: response.id,
			model: response.model,
			choices: [
				{
					index: 0,
					message: {
						role: 'assistant',
						content: text,
						...(tool_calls.length ? { tool_calls } : {}),
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

	const emitChatStreamFromResponses = async (channel, response) => {
		const chatLike = responsesToChatCompletion(response);
		const message = chatLike.choices?.[0]?.message ?? {};

		if (message.reasoning_content) {
			$socket?.emit(channel, `data: ${JSON.stringify({ choices: [{ delta: { reasoning_content: message.reasoning_content } }] })}`);
		}

		(message.tool_calls ?? []).forEach((toolCall, index) => {
			$socket?.emit(
				channel,
				`data: ${JSON.stringify({
					choices: [
						{
							delta: {
								tool_calls: [
									{
										index,
										id: toolCall.id,
										type: 'function',
										function: toolCall.function
									}
								]
							}
						}
					]
				})}`
			);
		});

		if (message.content) {
			$socket?.emit(channel, `data: ${JSON.stringify({ choices: [{ delta: { content: message.content } }] })}`);
		}

		if (chatLike.usage) {
			$socket?.emit(channel, `data: ${JSON.stringify({ usage: chatLike.usage })}`);
		}

		$socket?.emit(channel, 'data: [DONE]');
	};

	const emitChatStreamFromResponsesSSE = async (channel, responseBody) => {
		const extractReasoningSummaryFromOutput = (items = []) =>
			(items ?? [])
				.filter((item) => item?.type === 'reasoning')
				.flatMap((item) => item?.summary ?? [])
				.map((summaryItem) => summaryItem?.text ?? '')
				.filter((text) => typeof text === 'string' && text.trim() !== '')
				.join('\n');

		const extractReasoningSummaryFromRoot = (reasoning = null) => {
			const value = Array.isArray(reasoning?.summary)
				? reasoning.summary.map((s) => s?.text ?? '').join('\n')
				: reasoning?.summary ?? '';

			const normalized = typeof value === 'string' ? value.trim().toLowerCase() : '';
			if (['auto', 'concise', 'detailed'].includes(normalized)) {
				return '';
			}

			return typeof value === 'string' ? value : '';
		};

		const reader = responseBody.getReader();
		const decoder = new TextDecoder();
		let buffer = '';
		const state = {
			functionCalls: {},
			callIndexes: {},
			emittedCallIds: new Set(),
			textEmitted: false
		};

		const emitDelta = (delta) => {
			$socket?.emit(channel, `data: ${JSON.stringify({ choices: [{ delta }] })}`);
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
					$socket?.emit(channel, 'data: [DONE]');
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

							const index =
								state.callIndexes[callId] ?? Object.keys(state.callIndexes).length;
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
							.filter((item) => item?.type === 'message')
							.flatMap((item) => item?.content ?? [])
							.filter((part) => part?.type === 'output_text')
							.map((part) => part?.text ?? '')
							.join('');
						if (text) emitDelta({ content: text });
					}

					if (response?.usage) {
						$socket?.emit(
							channel,
							`data: ${JSON.stringify({
								usage: {
									prompt_tokens: response.usage.input_tokens ?? 0,
									completion_tokens: response.usage.output_tokens ?? 0,
									total_tokens: response.usage.total_tokens ?? 0
								}
							})}`
						);
					}

					$socket?.emit(channel, 'data: [DONE]');
					return;
				}
			}
		}

		$socket?.emit(channel, 'data: [DONE]');
	};


	const chatEventHandler = async (event, cb) => {
		const chat = $page.url.pathname.includes(`/c/${event.chat_id}`);

		let isFocused = document.visibilityState !== 'visible';
		if (window.electronAPI) {
			const res = await window.electronAPI.send({
				type: 'window:isFocused'
			});
			if (res) {
				isFocused = res.isFocused;
			}
		}

		await tick();
		const type = event?.data?.type ?? null;
		const data = event?.data?.data ?? null;

		if ((event.chat_id !== $chatId && !$temporaryChatEnabled) || isFocused) {
			if (type === 'chat:completion') {
				const { done, content, title } = data;

				if (done) {
					if ($settings?.notificationSoundAlways ?? false) {
						playingNotificationSound.set(true);

						const audio = new Audio(`/audio/notification.mp3`);
						audio.play().finally(() => {
							// Ensure the global state is reset after the sound finishes
							playingNotificationSound.set(false);
						});
					}

					if ($isLastActiveTab) {
						if ($settings?.notificationEnabled ?? false) {
							new Notification(`${title} • Open WebUI`, {
								body: content,
								icon: `${WEBUI_BASE_URL}/static/favicon.png`
							});
						}
					}

					toast.custom(NotificationToast, {
						componentProps: {
							onClick: () => {
								goto(`/c/${event.chat_id}`);
							},
							content: content,
							title: title
						},
						duration: 15000,
						unstyled: true
					});
				}
			} else if (type === 'chat:title') {
				currentChatPage.set(1);
				await chats.set(await getChatList(localStorage.token, $currentChatPage));
			} else if (type === 'chat:tags') {
				tags.set(await getAllTags(localStorage.token));
			}
		} else if (data?.session_id === $socket.id) {
			if (type === 'execute:python') {
				console.log('execute:python', data);
				executePythonAsWorker(data.id, data.code, cb);
			} else if (type === 'request:chat:completion') {
				console.log(data, $socket.id);
				const { session_id, channel, form_data, model } = data;

				try {
					const directConnections = $settings?.directConnections ?? {};

					if (directConnections) {
						const urlIdx = model?.urlIdx;

						const OPENAI_API_URL = directConnections.OPENAI_API_BASE_URLS[urlIdx];
						const OPENAI_API_KEY = directConnections.OPENAI_API_KEYS[urlIdx];
						const API_CONFIG = directConnections.OPENAI_API_CONFIGS[urlIdx] ?? {};
						const providerType = getProviderTypeFromModel(model, directConnections);
						const endpointType = getEndpointTypeFromProvider(providerType);

						try {
							if (API_CONFIG?.prefix_id) {
								const prefixId = API_CONFIG.prefix_id;
								form_data['model'] = form_data['model'].replace(`${prefixId}.`, ``);
							}

							const requestBody =
								endpointType === 'responses'
									? toResponsesPayloadUtil(form_data)
									: form_data;

							const [res, controller] =
								endpointType === 'responses'
									? await responsesCompletion(OPENAI_API_KEY, requestBody, OPENAI_API_URL)
									: await chatCompletion(OPENAI_API_KEY, requestBody, OPENAI_API_URL);

							if (res) {
								// raise if the response is not ok
								if (!res.ok) {
									throw await res.json();
								}

								if (form_data?.stream ?? false) {
									cb({
										status: true
									});
									console.log({ status: true });

									if (endpointType === 'responses') {
										await streamResponsesSSEToChatLinesUtil(res.body, (line) => {
												$socket?.emit(channel, line);
											});
									} else {
										// res will either be SSE or JSON
										const reader = res.body.getReader();
										const decoder = new TextDecoder();

										const processStream = async () => {
											while (true) {
												// Read data chunks from the response stream
												const { done, value } = await reader.read();
												if (done) {
													break;
												}

												// Decode the received chunk
												const chunk = decoder.decode(value, { stream: true });

												// Process lines within the chunk
												const lines = chunk.split('\n').filter((line) => line.trim() !== '');

												for (const line of lines) {
													console.log(line);
													$socket?.emit(channel, line);
												}
											}
										};

										// Process the stream in the background
										await processStream();
									}
								} else {
									const data = await res.json();
									cb(endpointType === 'responses' ? responsesToChatCompletionUtil(data) : data);
								}
							} else {
								throw new Error('An error occurred while fetching the completion');
							}
						} catch (error) {
							console.error('chatCompletion', error);
							cb(error);
						}
					}
				} catch (error) {
					console.error('chatCompletion', error);
					cb(error);
				} finally {
					$socket.emit(channel, {
						done: true
					});
				}
			} else {
				console.log('chatEventHandler', event);
			}
		}
	};

	const channelEventHandler = async (event) => {
		console.log('channelEventHandler', event);
		if (event.data?.type === 'typing') {
			return;
		}

		// handle channel created event
		if (event.data?.type === 'channel:created') {
			const res = await getChannels(localStorage.token).catch(async (error) => {
				return null;
			});

			if (res) {
				await channels.set(
					res.sort(
						(a, b) =>
							['', null, 'group', 'dm'].indexOf(a.type) - ['', null, 'group', 'dm'].indexOf(b.type)
					)
				);
			}

			return;
		}

		// check url path
		const channel = $page.url.pathname.includes(`/channels/${event.channel_id}`);

		let isFocused = document.visibilityState !== 'visible';
		if (window.electronAPI) {
			const res = await window.electronAPI.send({
				type: 'window:isFocused'
			});
			if (res) {
				isFocused = res.isFocused;
			}
		}

		if ((!channel || isFocused) && event?.user?.id !== $user?.id) {
			await tick();
			const type = event?.data?.type ?? null;
			const data = event?.data?.data ?? null;

			if ($channels) {
				if ($channels.find((ch) => ch.id === event.channel_id) && $channelId !== event.channel_id) {
					channels.set(
						$channels.map((ch) => {
							if (ch.id === event.channel_id) {
								if (type === 'message') {
									return {
										...ch,
										unread_count: (ch.unread_count ?? 0) + 1,
										last_message_at: event.created_at
									};
								}
							}
							return ch;
						})
					);
				} else {
					const res = await getChannels(localStorage.token).catch(async (error) => {
						return null;
					});

					if (res) {
						await channels.set(
							res.sort(
								(a, b) =>
									['', null, 'group', 'dm'].indexOf(a.type) -
									['', null, 'group', 'dm'].indexOf(b.type)
							)
						);
					}
				}
			}

			if (type === 'message') {
				const title = `${data?.user?.name}${event?.channel?.type !== 'dm' ? ` (#${event?.channel?.name})` : ''}`;

				if ($isLastActiveTab) {
					if ($settings?.notificationEnabled ?? false) {
						new Notification(`${title} • Open WebUI`, {
							body: data?.content,
							icon: `${WEBUI_API_BASE_URL}/users/${data?.user?.id}/profile/image`
						});
					}
				}

				toast.custom(NotificationToast, {
					componentProps: {
						onClick: () => {
							goto(`/channels/${event.channel_id}`);
						},
						content: data?.content,
						title: `${title}`
					},
					duration: 15000,
					unstyled: true
				});
			}
		}
	};

	const TOKEN_EXPIRY_BUFFER = 60; // seconds
	const checkTokenExpiry = async () => {
		const exp = $user?.expires_at; // token expiry time in unix timestamp
		const now = Math.floor(Date.now() / 1000); // current time in unix timestamp

		if (!exp) {
			// If no expiry time is set, do nothing
			return;
		}

		if (now >= exp - TOKEN_EXPIRY_BUFFER) {
			const res = await userSignOut();
			user.set(null);
			localStorage.removeItem('token');

			location.href = res?.redirect_url ?? '/auth';
		}
	};

	onMount(async () => {
		let touchstartY = 0;

		function isNavOrDescendant(el) {
			const nav = document.querySelector('nav'); // change selector if needed
			return nav && (el === nav || nav.contains(el));
		}

		document.addEventListener('touchstart', (e) => {
			if (!isNavOrDescendant(e.target)) return;
			touchstartY = e.touches[0].clientY;
		});

		document.addEventListener('touchmove', (e) => {
			if (!isNavOrDescendant(e.target)) return;
			const touchY = e.touches[0].clientY;
			const touchDiff = touchY - touchstartY;
			if (touchDiff > 50 && window.scrollY === 0) {
				showRefresh = true;
				e.preventDefault();
			}
		});

		document.addEventListener('touchend', (e) => {
			if (!isNavOrDescendant(e.target)) return;
			if (showRefresh) {
				showRefresh = false;
				location.reload();
			}
		});

		if (typeof window !== 'undefined') {
			if (window.applyTheme) {
				window.applyTheme();
			}
		}

		if (window?.electronAPI) {
			const info = await window.electronAPI.send({
				type: 'app:info'
			});

			if (info) {
				isApp.set(true);
				appInfo.set(info);

				const data = await window.electronAPI.send({
					type: 'app:data'
				});

				if (data) {
					appData.set(data);
				}
			}
		}

		// Listen for messages on the BroadcastChannel
		bc.onmessage = (event) => {
			if (event.data === 'active') {
				isLastActiveTab.set(false); // Another tab became active
			}
		};

		// Set yourself as the last active tab when this tab is focused
		const handleVisibilityChange = () => {
			if (document.visibilityState === 'visible') {
				isLastActiveTab.set(true); // This tab is now the active tab
				bc.postMessage('active'); // Notify other tabs that this tab is active

				// Check token expiry when the tab becomes active
				checkTokenExpiry();
			}
		};

		// Add event listener for visibility state changes
		document.addEventListener('visibilitychange', handleVisibilityChange);

		// Call visibility change handler initially to set state on load
		handleVisibilityChange();

		theme.set(localStorage.theme);

		mobile.set(window.innerWidth < BREAKPOINT);

		const onResize = () => {
			if (window.innerWidth < BREAKPOINT) {
				mobile.set(true);
			} else {
				mobile.set(false);
			}
		};
		window.addEventListener('resize', onResize);

		user.subscribe(async (value) => {
			if (value) {
				$socket?.off('events', chatEventHandler);
				$socket?.off('events:channel', channelEventHandler);

				$socket?.on('events', chatEventHandler);
				$socket?.on('events:channel', channelEventHandler);

				const userSettings = await getUserSettings(localStorage.token);
				if (userSettings) {
					settings.set(userSettings.ui);
				} else {
					settings.set(JSON.parse(localStorage.getItem('settings') ?? '{}'));
				}
				setTextScale($settings?.textScale ?? 1);

				// Set up the token expiry check
				if (tokenTimer) {
					clearInterval(tokenTimer);
				}
				tokenTimer = setInterval(checkTokenExpiry, 15000);
			} else {
				$socket?.off('events', chatEventHandler);
				$socket?.off('events:channel', channelEventHandler);
			}
		});

		let backendConfig = null;
		try {
			backendConfig = await getBackendConfig();
			console.log('Backend config:', backendConfig);
		} catch (error) {
			console.error('Error loading backend config:', error);
		}
		// Initialize i18n even if we didn't get a backend config,
		// so `/error` can show something that's not `undefined`.

		initI18n(localStorage?.locale);
		if (!localStorage.locale) {
			const languages = await getLanguages();
			const browserLanguages = navigator.languages
				? navigator.languages
				: [navigator.language || navigator.userLanguage];
			const lang = backendConfig.default_locale
				? backendConfig.default_locale
				: bestMatchingLanguage(languages, browserLanguages, 'en-US');
			changeLanguage(lang);
			dayjs.locale(lang);
		}

		if (backendConfig) {
			// Save Backend Status to Store
			await config.set(backendConfig);
			await WEBUI_NAME.set(backendConfig.name);

			if ($config) {
				await setupSocket($config.features?.enable_websocket ?? true);

				const currentUrl = `${window.location.pathname}${window.location.search}`;
				const encodedUrl = encodeURIComponent(currentUrl);

				if (localStorage.token) {
					// Get Session User Info
					const sessionUser = await getSessionUser(localStorage.token).catch((error) => {
						toast.error(`${error}`);
						return null;
					});

					if (sessionUser) {
						await user.set(sessionUser);
						await config.set(await getBackendConfig());
					} else {
						// Redirect Invalid Session User to /auth Page
						localStorage.removeItem('token');
						await goto(`/auth?redirect=${encodedUrl}`);
					}
				} else {
					// Don't redirect if we're already on the auth page
					// Needed because we pass in tokens from OAuth logins via URL fragments
					if ($page.url.pathname !== '/auth') {
						await goto(`/auth?redirect=${encodedUrl}`);
					}
				}
			}
		} else {
			// Redirect to /error when Backend Not Detected
			await goto(`/error`);
		}

		await tick();

		if (
			document.documentElement.classList.contains('her') &&
			document.getElementById('progress-bar')
		) {
			loadingProgress.subscribe((value) => {
				const progressBar = document.getElementById('progress-bar');

				if (progressBar) {
					progressBar.style.width = `${value}%`;
				}
			});

			await loadingProgress.set(100);

			document.getElementById('splash-screen')?.remove();

			const audio = new Audio(`/audio/greeting.mp3`);
			const playAudio = () => {
				audio.play();
				document.removeEventListener('click', playAudio);
			};

			document.addEventListener('click', playAudio);

			loaded = true;
		} else {
			document.getElementById('splash-screen')?.remove();
			loaded = true;
		}

		return () => {
			window.removeEventListener('resize', onResize);
		};
	});
</script>

<svelte:head>
	<title>{$WEBUI_NAME}</title>
	<link crossorigin="anonymous" rel="icon" href="{WEBUI_BASE_URL}/static/favicon.png" />

	<meta name="apple-mobile-web-app-title" content={$WEBUI_NAME} />
	<meta name="description" content={$WEBUI_NAME} />
	<link
		rel="search"
		type="application/opensearchdescription+xml"
		title={$WEBUI_NAME}
		href="/opensearch.xml"
		crossorigin="use-credentials"
	/>
</svelte:head>

{#if showRefresh}
	<div class=" py-5">
		<Spinner className="size-5" />
	</div>
{/if}

{#if loaded}
	{#if $isApp}
		<div class="flex flex-row h-screen">
			<AppSidebar />

			<div class="w-full flex-1 max-w-[calc(100%-4.5rem)]">
				<slot />
			</div>
		</div>
	{:else}
		<slot />
	{/if}
{/if}

<Toaster
	theme={$theme.includes('dark')
		? 'dark'
		: $theme === 'system'
			? window.matchMedia('(prefers-color-scheme: dark)').matches
				? 'dark'
				: 'light'
			: 'light'}
	richColors
	position="top-right"
	closeButton
/>
