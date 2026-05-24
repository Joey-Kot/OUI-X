<script lang="ts">
	import { toast } from 'svelte-sonner';
	import { createEventDispatcher, onMount, getContext } from 'svelte';
	const dispatch = createEventDispatcher();

	import { getBackendConfig } from '$lib/apis';
	import {
		getAudioConfig,
		updateAudioConfig,
		getModels as _getModels,
		getVoices as _getVoices
	} from '$lib/apis/audio';
	import { config } from '$lib/stores';

	import SensitiveInput from '$lib/components/common/SensitiveInput.svelte';

	import { TTS_OUTPUT_FORMAT, TTS_RESPONSE_SPLIT } from '$lib/types';

	import type { Writable } from 'svelte/store';
	import type { i18n as i18nType } from 'i18next';
	import Textarea from '$lib/components/common/Textarea.svelte';

	const i18n = getContext<Writable<i18nType>>('i18n');

	export let saveHandler: () => void;

	type TTSVoice = SpeechSynthesisVoice & { id?: string };

	// Audio
	let TTS_OPENAI_API_BASE_URL = '';
	let TTS_OPENAI_API_KEY = '';
	let TTS_API_KEY = '';
	let TTS_ENGINE = '';
	let TTS_MODEL = '';
	let TTS_VOICE = '';
	let TTS_OPENAI_PARAMS = '';
	let TTS_GEMINI_API_BASE_URL = '';
	let TTS_GEMINI_API_KEY = '';
	let TTS_GEMINI_PARAMS = '';
	let TTS_GEMINI_SCENE = '';
	let TTS_GEMINI_SAMPLE_CONTEXT = '';
	let TTS_GEMINI_STYLE = '';
	let TTS_GEMINI_PACE = '';
	let TTS_GEMINI_ACCENT = '';
	let TTS_GEMINI_TEMPERATURE = 1;
	let TTS_QWEN_API_BASE_URL = '';
	let TTS_QWEN_API_KEY = '';
	let TTS_QWEN_PARAMS = '';
	let TTS_SPLIT_ON: TTS_RESPONSE_SPLIT = TTS_RESPONSE_SPLIT.PUNCTUATION;
	let TTS_OUTPUT_FORMAT_VALUE: TTS_OUTPUT_FORMAT = TTS_OUTPUT_FORMAT.DEFAULT;
	let TTS_AZURE_SPEECH_REGION = '';
	let TTS_AZURE_SPEECH_BASE_URL = '';
	let TTS_AZURE_SPEECH_OUTPUT_FORMAT = '';

	let STT_OPENAI_API_BASE_URL = '';
	let STT_OPENAI_API_KEY = '';
	let STT_ENGINE = 'openai';
	let STT_MODEL = '';
	let STT_SUPPORTED_CONTENT_TYPES = '';
	let STT_AZURE_API_KEY = '';
	let STT_AZURE_REGION = '';
	let STT_AZURE_LOCALES = '';
	let STT_AZURE_BASE_URL = '';
	let STT_AZURE_MAX_SPEAKERS = '';

	// eslint-disable-next-line no-undef
	let voices: TTSVoice[] = [];
	let models: Awaited<ReturnType<typeof _getModels>>['models'] = [];

	const getModels = async () => {
		if (TTS_ENGINE === '') {
			models = [];
		} else {
			const res = await _getModels(localStorage.token).catch((e) => {
				toast.error(`${e}`);
			});

			if (res) {
				console.log(res);
				models = res.models;
			}
		}
	};

	const getVoices = async () => {
		if (TTS_ENGINE === '') {
			const getVoicesLoop = setInterval(() => {
				voices = speechSynthesis.getVoices();

				// do your loop
				if (voices.length > 0) {
					clearInterval(getVoicesLoop);
					voices.sort((a, b) => a.name.localeCompare(b.name, $i18n.resolvedLanguage));
				}
			}, 100);
		} else {
			const res = await _getVoices(localStorage.token).catch((e) => {
				toast.error(`${e}`);
			});

			if (res) {
				console.log(res);
				voices = res.voices;
				voices.sort((a, b) => a.name.localeCompare(b.name, $i18n.resolvedLanguage));
			}
		}
	};

	const updateConfigHandler = async () => {
		let openaiParams = {};
		let geminiParams = {};
		let qwenParams = {};
		try {
			openaiParams = TTS_OPENAI_PARAMS ? JSON.parse(TTS_OPENAI_PARAMS) : {};
			TTS_OPENAI_PARAMS = JSON.stringify(openaiParams, null, 2);
			geminiParams = TTS_GEMINI_PARAMS ? JSON.parse(TTS_GEMINI_PARAMS) : {};
			TTS_GEMINI_PARAMS = JSON.stringify(geminiParams, null, 2);
			qwenParams = TTS_QWEN_PARAMS ? JSON.parse(TTS_QWEN_PARAMS) : {};
			TTS_QWEN_PARAMS = JSON.stringify(qwenParams, null, 2);
		} catch (e) {
			toast.error($i18n.t('Invalid JSON format for Parameters'));
			return;
		}

		const res = await updateAudioConfig(localStorage.token, {
			tts: {
				OPENAI_API_BASE_URL: TTS_OPENAI_API_BASE_URL,
				OPENAI_API_KEY: TTS_OPENAI_API_KEY,
				OPENAI_PARAMS: openaiParams,
				GEMINI_API_BASE_URL: TTS_GEMINI_API_BASE_URL,
				GEMINI_API_KEY: TTS_GEMINI_API_KEY,
				GEMINI_PARAMS: geminiParams,
				GEMINI_SCENE: TTS_GEMINI_SCENE,
				GEMINI_SAMPLE_CONTEXT: TTS_GEMINI_SAMPLE_CONTEXT,
				GEMINI_STYLE: TTS_GEMINI_STYLE,
				GEMINI_PACE: TTS_GEMINI_PACE,
				GEMINI_ACCENT: TTS_GEMINI_ACCENT,
				GEMINI_TEMPERATURE: TTS_GEMINI_TEMPERATURE,
				QWEN_API_BASE_URL: TTS_QWEN_API_BASE_URL,
				QWEN_API_KEY: TTS_QWEN_API_KEY,
				QWEN_PARAMS: qwenParams,
				API_KEY: TTS_API_KEY,
				ENGINE: TTS_ENGINE,
				MODEL: TTS_MODEL,
				VOICE: TTS_VOICE,
				AZURE_SPEECH_REGION: TTS_AZURE_SPEECH_REGION,
				AZURE_SPEECH_BASE_URL: TTS_AZURE_SPEECH_BASE_URL,
				AZURE_SPEECH_OUTPUT_FORMAT: TTS_AZURE_SPEECH_OUTPUT_FORMAT,
				SPLIT_ON: TTS_SPLIT_ON,
				OUTPUT_FORMAT: TTS_OUTPUT_FORMAT_VALUE
			},
			stt: {
				OPENAI_API_BASE_URL: STT_OPENAI_API_BASE_URL,
				OPENAI_API_KEY: STT_OPENAI_API_KEY,
				ENGINE: ['openai', 'web', 'azure'].includes(STT_ENGINE) ? STT_ENGINE : 'openai',
				MODEL: STT_MODEL,
				SUPPORTED_CONTENT_TYPES: STT_SUPPORTED_CONTENT_TYPES.split(','),
				AZURE_API_KEY: STT_AZURE_API_KEY,
				AZURE_REGION: STT_AZURE_REGION,
				AZURE_LOCALES: STT_AZURE_LOCALES,
				AZURE_BASE_URL: STT_AZURE_BASE_URL,
				AZURE_MAX_SPEAKERS: STT_AZURE_MAX_SPEAKERS
			}
		});

		if (res) {
			saveHandler();
			config.set(await getBackendConfig());
		}
	};

	onMount(async () => {
		const res = await getAudioConfig(localStorage.token);

		if (res) {
			console.log(res);
			TTS_OPENAI_API_BASE_URL = res.tts.OPENAI_API_BASE_URL;
			TTS_OPENAI_API_KEY = res.tts.OPENAI_API_KEY;
			TTS_OPENAI_PARAMS = JSON.stringify(res?.tts?.OPENAI_PARAMS ?? '', null, 2);
			TTS_GEMINI_API_BASE_URL =
				res.tts.GEMINI_API_BASE_URL || 'https://generativelanguage.googleapis.com';
			TTS_GEMINI_API_KEY = res.tts.GEMINI_API_KEY || '';
			TTS_GEMINI_PARAMS = JSON.stringify(res?.tts?.GEMINI_PARAMS ?? '', null, 2);
			TTS_GEMINI_SCENE = res.tts.GEMINI_SCENE || '';
			TTS_GEMINI_SAMPLE_CONTEXT = res.tts.GEMINI_SAMPLE_CONTEXT || '';
			TTS_GEMINI_STYLE = res.tts.GEMINI_STYLE || '';
			TTS_GEMINI_PACE = res.tts.GEMINI_PACE || '';
			TTS_GEMINI_ACCENT = res.tts.GEMINI_ACCENT || '';
			TTS_GEMINI_TEMPERATURE = res.tts.GEMINI_TEMPERATURE ?? 1;
			TTS_QWEN_API_BASE_URL =
				res.tts.QWEN_API_BASE_URL || 'https://dashscope.aliyuncs.com/api/v1';
			TTS_QWEN_API_KEY = res.tts.QWEN_API_KEY || '';
			TTS_QWEN_PARAMS = JSON.stringify(res?.tts?.QWEN_PARAMS ?? '', null, 2);
			TTS_API_KEY = res.tts.API_KEY;

			TTS_ENGINE = res.tts.ENGINE;
			TTS_MODEL = res.tts.MODEL;
			TTS_VOICE = res.tts.VOICE;

			TTS_SPLIT_ON = res.tts.SPLIT_ON || TTS_RESPONSE_SPLIT.PUNCTUATION;
			TTS_OUTPUT_FORMAT_VALUE = res.tts.OUTPUT_FORMAT || TTS_OUTPUT_FORMAT.DEFAULT;

			TTS_AZURE_SPEECH_REGION = res.tts.AZURE_SPEECH_REGION;
			TTS_AZURE_SPEECH_BASE_URL = res.tts.AZURE_SPEECH_BASE_URL;
			TTS_AZURE_SPEECH_OUTPUT_FORMAT = res.tts.AZURE_SPEECH_OUTPUT_FORMAT;

			STT_OPENAI_API_BASE_URL = res.stt.OPENAI_API_BASE_URL;
			STT_OPENAI_API_KEY = res.stt.OPENAI_API_KEY;

			STT_ENGINE = ['openai', 'web', 'azure'].includes(res.stt.ENGINE) ? res.stt.ENGINE : 'openai';
			STT_MODEL = res.stt.MODEL;
			STT_SUPPORTED_CONTENT_TYPES = (res?.stt?.SUPPORTED_CONTENT_TYPES ?? []).join(',');
			STT_AZURE_API_KEY = res.stt.AZURE_API_KEY;
			STT_AZURE_REGION = res.stt.AZURE_REGION;
			STT_AZURE_LOCALES = res.stt.AZURE_LOCALES;
			STT_AZURE_BASE_URL = res.stt.AZURE_BASE_URL;
			STT_AZURE_MAX_SPEAKERS = res.stt.AZURE_MAX_SPEAKERS;
		}

		await getVoices();
		await getModels();
	});
</script>

<form
	class="flex flex-col h-full justify-between space-y-3 text-sm"
	on:submit|preventDefault={async () => {
		await updateConfigHandler();
		dispatch('save');
	}}
>
	<div class=" space-y-3 overflow-y-scroll scrollbar-hidden h-full">
		<div class="flex flex-col gap-3">
			<div>
				<div class=" mt-0.5 mb-2.5 text-base font-medium">{$i18n.t('Speech-to-Text')}</div>

				<hr class=" border-gray-100/30 dark:border-gray-850/30 my-2" />

				{#if STT_ENGINE !== 'web'}
					<div class="mb-2">
						<div class=" mb-1.5 text-xs font-medium">{$i18n.t('Supported MIME Types')}</div>
						<div class="flex w-full">
							<div class="flex-1">
								<input
									class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
									bind:value={STT_SUPPORTED_CONTENT_TYPES}
									placeholder={$i18n.t(
										'e.g., audio/wav,audio/mpeg,video/* (leave blank for defaults)'
									)}
								/>
							</div>
						</div>
					</div>
				{/if}

				<div class="mb-2 py-0.5 flex w-full justify-between">
					<div class=" self-center text-xs font-medium">{$i18n.t('Speech-to-Text Engine')}</div>
					<div class="flex items-center relative">
						<select
							class="dark:bg-gray-900 cursor-pointer w-fit pr-8 rounded-sm px-2 p-1 text-xs bg-transparent outline-hidden text-right"
							bind:value={STT_ENGINE}
							placeholder={$i18n.t('Select an engine')}
						>
							<option value="openai">{$i18n.t('OpenAI')}</option>
							<option value="web">{$i18n.t('Web API')}</option>
							<option value="azure">{$i18n.t('Azure AI Speech')}</option>
						</select>
					</div>
				</div>

				{#if STT_ENGINE === 'openai'}
					<div>
						<div class="mt-1 flex gap-2 mb-1">
							<input
								class="flex-1 w-full bg-transparent outline-hidden"
								placeholder={$i18n.t('API Base URL')}
								bind:value={STT_OPENAI_API_BASE_URL}
								required
							/>

							<SensitiveInput placeholder={$i18n.t('API Key')} bind:value={STT_OPENAI_API_KEY} />
						</div>
					</div>

					<hr class="border-gray-100/30 dark:border-gray-850/30 my-2" />

					<div>
						<div class=" mb-1.5 text-xs font-medium">{$i18n.t('STT Model')}</div>
						<div class="flex w-full">
							<div class="flex-1">
								<input
									list="model-list"
									class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
									bind:value={STT_MODEL}
									placeholder={$i18n.t('Select a model')}
								/>

								<datalist id="model-list">
									<option value="whisper-1"></option>
								</datalist>
							</div>
						</div>
					</div>
				{:else if STT_ENGINE === 'azure'}
					<div>
						<div class="mt-1 flex gap-2 mb-1">
							<SensitiveInput
								placeholder={$i18n.t('API Key')}
								bind:value={STT_AZURE_API_KEY}
								required
							/>
						</div>

						<hr class="border-gray-100/30 dark:border-gray-850/30 my-2" />

						<div>
							<div class=" mb-1.5 text-xs font-medium">{$i18n.t('Azure Region')}</div>
							<div class="flex w-full">
								<div class="flex-1">
									<input
										class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
										bind:value={STT_AZURE_REGION}
										placeholder={$i18n.t('e.g., westus (leave blank for eastus)')}
									/>
								</div>
							</div>
						</div>

						<div>
							<div class=" mb-1.5 text-xs font-medium">{$i18n.t('Language Locales')}</div>
							<div class="flex w-full">
								<div class="flex-1">
									<input
										class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
										bind:value={STT_AZURE_LOCALES}
										placeholder={$i18n.t('e.g., en-US,ja-JP (leave blank for auto-detect)')}
									/>
								</div>
							</div>
						</div>

						<div>
							<div class=" mb-1.5 text-xs font-medium">{$i18n.t('Endpoint URL')}</div>
							<div class="flex w-full">
								<div class="flex-1">
									<input
										class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
										bind:value={STT_AZURE_BASE_URL}
										placeholder={$i18n.t('(leave blank for to use commercial endpoint)')}
									/>
								</div>
							</div>
						</div>

						<div>
							<div class=" mb-1.5 text-xs font-medium">{$i18n.t('Max Speakers')}</div>
							<div class="flex w-full">
								<div class="flex-1">
									<input
										class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
										bind:value={STT_AZURE_MAX_SPEAKERS}
										placeholder={$i18n.t('e.g., 3, 4, 5 (leave blank for default)')}
									/>
								</div>
							</div>
						</div>
					</div>
				{/if}
			</div>

			<div>
				<div class=" mt-0.5 mb-2.5 text-base font-medium">{$i18n.t('Text-to-Speech')}</div>

				<hr class=" border-gray-100/30 dark:border-gray-850/30 my-2" />

				<div class="mb-2 py-0.5 flex w-full justify-between">
					<div class=" self-center text-xs font-medium">{$i18n.t('Text-to-Speech Engine')}</div>
					<div class="flex items-center relative">
						<select
							class=" dark:bg-gray-900 w-fit pr-8 cursor-pointer rounded-sm px-2 p-1 text-xs bg-transparent outline-hidden text-right"
							bind:value={TTS_ENGINE}
							placeholder={$i18n.t('Select a mode')}
							on:change={async (e) => {
								const selectedEngine = (e.currentTarget as HTMLSelectElement).value;

								if (selectedEngine === 'openai') {
									TTS_VOICE = 'alloy';
									TTS_MODEL = 'gpt-4o-mini-tts';
								} else if (selectedEngine === 'gemini') {
									TTS_VOICE = 'Zephyr';
									TTS_MODEL = 'gemini-3.1-flash-tts-preview';
									TTS_GEMINI_API_BASE_URL =
										TTS_GEMINI_API_BASE_URL || 'https://generativelanguage.googleapis.com';
								} else if (selectedEngine === 'qwen') {
									TTS_VOICE = 'Cherry';
									TTS_MODEL = 'qwen3-tts-flash';
									TTS_QWEN_API_BASE_URL =
										TTS_QWEN_API_BASE_URL || 'https://dashscope.aliyuncs.com/api/v1';
								} else {
									TTS_VOICE = '';
									TTS_MODEL = '';
								}

								await updateConfigHandler();
								await getVoices();
								await getModels();
							}}
						>
							<option value="">{$i18n.t('Web API')}</option>
							<option value="openai">{$i18n.t('OpenAI')}</option>
							<option value="gemini">{$i18n.t('Gemini')}</option>
							<option value="qwen">{$i18n.t('Qwen')}</option>
							<option value="azure">{$i18n.t('Azure AI Speech')}</option>
						</select>
					</div>
				</div>

				{#if TTS_ENGINE === 'openai'}
					<div>
						<div class="mt-1 flex gap-2 mb-1">
							<input
								class="flex-1 w-full bg-transparent outline-hidden"
								placeholder={$i18n.t('API Base URL')}
								bind:value={TTS_OPENAI_API_BASE_URL}
								required
							/>

							<SensitiveInput placeholder={$i18n.t('API Key')} bind:value={TTS_OPENAI_API_KEY} />
						</div>
					</div>
				{:else if TTS_ENGINE === 'gemini'}
					<div>
						<div class="mt-1 flex gap-2 mb-1">
							<input
								class="flex-1 w-full bg-transparent outline-hidden"
								placeholder={$i18n.t('Gemini Base URL')}
								bind:value={TTS_GEMINI_API_BASE_URL}
								required
							/>

							<SensitiveInput
								placeholder={$i18n.t('Gemini API Key')}
								bind:value={TTS_GEMINI_API_KEY}
								required
							/>
						</div>
					</div>
				{:else if TTS_ENGINE === 'qwen'}
					<div>
						<div class="mt-1 flex gap-2 mb-1">
							<input
								class="flex-1 w-full bg-transparent outline-hidden"
								placeholder={$i18n.t('Qwen Base URL')}
								bind:value={TTS_QWEN_API_BASE_URL}
								required
							/>

							<SensitiveInput
								placeholder={$i18n.t('Qwen API Key')}
								bind:value={TTS_QWEN_API_KEY}
								required
							/>
						</div>
					</div>
				{:else if TTS_ENGINE === 'azure'}
					<div>
						<div class="mt-1 flex gap-2 mb-1">
							<SensitiveInput placeholder={$i18n.t('API Key')} bind:value={TTS_API_KEY} required />
						</div>

						<hr class="border-gray-100/30 dark:border-gray-850/30 my-2" />

						<div>
							<div class=" mb-1.5 text-xs font-medium">{$i18n.t('Azure Region')}</div>
							<div class="flex w-full">
								<div class="flex-1">
									<input
										class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
										bind:value={TTS_AZURE_SPEECH_REGION}
										placeholder={$i18n.t('e.g., westus (leave blank for eastus)')}
									/>
								</div>
							</div>
						</div>

						<div>
							<div class=" mb-1.5 text-xs font-medium">{$i18n.t('Endpoint URL')}</div>
							<div class="flex w-full">
								<div class="flex-1">
									<input
										class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
										bind:value={TTS_AZURE_SPEECH_BASE_URL}
										placeholder={$i18n.t('(leave blank for to use commercial endpoint)')}
									/>
								</div>
							</div>
						</div>
					</div>
				{/if}

				<div class="mb-2">
					{#if TTS_ENGINE === ''}
						<div>
							<div class=" mb-1.5 text-xs font-medium">{$i18n.t('TTS Voice')}</div>
							<div class="flex w-full">
								<div class="flex-1">
									<select
										class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
										bind:value={TTS_VOICE}
									>
										<option value="" selected={TTS_VOICE !== ''}>{$i18n.t('Default')}</option>
										{#each voices as voice}
											<option
												value={voice.voiceURI}
												class="bg-gray-100 dark:bg-gray-700"
												selected={TTS_VOICE === voice.voiceURI}>{voice.name}</option
											>
										{/each}
									</select>
								</div>
							</div>
						</div>
					{:else if TTS_ENGINE === 'openai'}
						<div class=" flex gap-2">
							<div class="w-full">
								<div class=" mb-1.5 text-xs font-medium">{$i18n.t('TTS Voice')}</div>
								<div class="flex w-full">
									<div class="flex-1">
										<input
											list="voice-list"
											class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
											bind:value={TTS_VOICE}
											placeholder={$i18n.t('Select a voice')}
										/>

										<datalist id="voice-list">
											{#each voices as voice}
												<option value={voice.id}>{voice.name}</option>
											{/each}
										</datalist>
									</div>
								</div>
							</div>
							<div class="w-full">
								<div class=" mb-1.5 text-xs font-medium">{$i18n.t('TTS Model')}</div>
								<div class="flex w-full">
									<div class="flex-1">
										<input
											list="tts-model-list"
											class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
											bind:value={TTS_MODEL}
											placeholder={$i18n.t('Select a model')}
										/>

										<datalist id="tts-model-list">
											{#each models as model}
												<option value={model.id} class="bg-gray-50 dark:bg-gray-700"></option>
											{/each}
										</datalist>
									</div>
								</div>
							</div>
						</div>

						<div class="mt-2 mb-1 text-xs text-gray-400 dark:text-gray-500">
							<div class="w-full">
								<div class=" mb-1.5 text-xs font-medium">{$i18n.t('Additional Parameters')}</div>
								<div class="flex w-full">
									<div class="flex-1">
										<Textarea
											className="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
											bind:value={TTS_OPENAI_PARAMS}
											placeholder={$i18n.t('Enter additional parameters in JSON format')}
											minSize={100}
										/>
									</div>
								</div>
							</div>
						</div>
					{:else if TTS_ENGINE === 'gemini'}
						<div class=" flex gap-2">
							<div class="w-full">
								<div class=" mb-1.5 text-xs font-medium">{$i18n.t('TTS Voice')}</div>
								<div class="flex w-full">
									<div class="flex-1">
										<input
											list="voice-list"
											class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
											bind:value={TTS_VOICE}
											placeholder={$i18n.t('Select a voice')}
										/>

										<datalist id="voice-list">
											{#each voices as voice}
												<option value={voice.id}>{voice.name}</option>
											{/each}
										</datalist>
									</div>
								</div>
							</div>
							<div class="w-full">
								<div class=" mb-1.5 text-xs font-medium">{$i18n.t('TTS Model')}</div>
								<div class="flex w-full">
									<div class="flex-1">
										<input
											list="tts-model-list"
											class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
											bind:value={TTS_MODEL}
											placeholder={$i18n.t('Select a model')}
										/>

										<datalist id="tts-model-list">
											{#each models as model}
												<option value={model.id} class="bg-gray-50 dark:bg-gray-700"></option>
											{/each}
										</datalist>
									</div>
								</div>
							</div>
						</div>
						<div class="mt-2">
							<div class=" mb-1.5 text-xs font-medium">{$i18n.t('Temperature')}</div>
							<div class="flex w-full">
								<div class="flex-1">
									<input
										type="number"
										step="0.1"
										min="0"
										max="2"
										class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
										bind:value={TTS_GEMINI_TEMPERATURE}
									/>
								</div>
							</div>
						</div>

						<div class="mt-2 grid grid-cols-1 md:grid-cols-3 gap-2">
							<div>
								<div class=" mb-1.5 text-xs font-medium">{$i18n.t('Style')}</div>
								<input
									list="gemini-style-list"
									class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
									bind:value={TTS_GEMINI_STYLE}
									placeholder={$i18n.t('Select or enter a style')}
								/>
								<datalist id="gemini-style-list">
									<option value="Vocal Smile">
										The &quot;Vocal Smile&quot;: The soft palate is raised to keep the tone bright,
										sunny, and explicitly inviting.
									</option>
									<option value="Newscaster">
										Professional, authoritative, clear articulation with standard broadcast cadence.
									</option>
									<option value="Whisper">Intimate, breathy, close-to-mic proximity effect.</option>
									<option value="Empathetic">
										Warm, understanding, soft tone with gentle inflections.
									</option>
									<option value="Promo/Hype">
										High energy, punchy consonants, elongated vowels on excitement words.
									</option>
									<option value="Deadpan">Flat affect, minimal pitch variation, dry delivery.</option>
								</datalist>
							</div>

							<div>
								<div class=" mb-1.5 text-xs font-medium">{$i18n.t('Pace')}</div>
								<input
									list="gemini-pace-list"
									class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
									bind:value={TTS_GEMINI_PACE}
									placeholder={$i18n.t('Select or enter a pace')}
								/>
								<datalist id="gemini-pace-list">
									<option value="Natural">Natural conversational pace.</option>
									<option value="Rapid Fire">
										Fast, energetic, no dead air. Sentences overlap slightly.
									</option>
									<option value="The Drift">
										Slow, liquid, zero urgency. Long pauses for breath.
									</option>
									<option value="Staccato">
										Short, clipped sentences with distinct pauses between words.
									</option>
								</datalist>
							</div>

							<div>
								<div class=" mb-1.5 text-xs font-medium">{$i18n.t('Accent')}</div>
								<input
									list="gemini-accent-list"
									class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
									bind:value={TTS_GEMINI_ACCENT}
									placeholder={$i18n.t('Select or enter an accent')}
								/>
								<datalist id="gemini-accent-list">
									<option value="Neutral"></option>
									<option value="American (Gen)"></option>
									<option value="American (Valley)"></option>
									<option value="American (South)"></option>
									<option value="British (RP)"></option>
									<option value="British (Brixton)"></option>
									<option value="Transatlantic"></option>
									<option value="Australian"></option>
								</datalist>
							</div>
						</div>

						<div class="mt-2">
							<div class=" mb-1.5 text-xs font-medium">{$i18n.t('Scene')}</div>
							<div class="flex w-full">
								<div class="flex-1">
									<Textarea
										className="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
										bind:value={TTS_GEMINI_SCENE}
										placeholder={$i18n.t('e.g., A quiet night')}
										minSize={64}
									/>
								</div>
							</div>
						</div>

						<div class="mt-2">
							<div class=" mb-1.5 text-xs font-medium">{$i18n.t('Sample Context')}</div>
							<div class="flex w-full">
								<div class="flex-1">
									<Textarea
										className="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
										bind:value={TTS_GEMINI_SAMPLE_CONTEXT}
										placeholder={$i18n.t('e.g., Just finished a busy day at work')}
										minSize={64}
									/>
								</div>
							</div>
						</div>

						<div class="mt-2 mb-1 text-xs text-gray-400 dark:text-gray-500">
							<div class="w-full">
								<div class=" mb-1.5 text-xs font-medium">{$i18n.t('Additional Parameters')}</div>
								<div class="flex w-full">
									<div class="flex-1">
										<Textarea
											className="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
											bind:value={TTS_GEMINI_PARAMS}
											placeholder={$i18n.t('Enter additional parameters in JSON format')}
											minSize={100}
										/>
									</div>
								</div>
							</div>
						</div>
					{:else if TTS_ENGINE === 'qwen'}
						<div class=" flex gap-2">
							<div class="w-full">
								<div class=" mb-1.5 text-xs font-medium">{$i18n.t('TTS Voice')}</div>
								<div class="flex w-full">
									<div class="flex-1">
										<input
											list="voice-list"
											class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
											bind:value={TTS_VOICE}
											placeholder={$i18n.t('Select a voice')}
										/>

										<datalist id="voice-list">
											{#each voices as voice}
												<option value={voice.id}>{voice.name}</option>
											{/each}
										</datalist>
									</div>
								</div>
							</div>
							<div class="w-full">
								<div class=" mb-1.5 text-xs font-medium">{$i18n.t('TTS Model')}</div>
								<div class="flex w-full">
									<div class="flex-1">
										<input
											list="tts-model-list"
											class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
											bind:value={TTS_MODEL}
											placeholder={$i18n.t('Select a model')}
										/>

										<datalist id="tts-model-list">
											{#each models as model}
												<option value={model.id} class="bg-gray-50 dark:bg-gray-700"></option>
											{/each}
										</datalist>
									</div>
								</div>
							</div>
						</div>

						<div class="mt-2 mb-1 text-xs text-gray-400 dark:text-gray-500">
							<div class="w-full">
								<div class=" mb-1.5 text-xs font-medium">{$i18n.t('Additional Parameters')}</div>
								<div class="flex w-full">
									<div class="flex-1">
										<Textarea
											className="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
											bind:value={TTS_QWEN_PARAMS}
											placeholder={$i18n.t('Enter additional input parameters in JSON format')}
											minSize={100}
										/>
									</div>
								</div>
							</div>
						</div>
					{:else if TTS_ENGINE === 'azure'}
						<div class=" flex gap-2">
							<div class="w-full">
								<div class=" mb-1.5 text-xs font-medium">{$i18n.t('TTS Voice')}</div>
								<div class="flex w-full">
									<div class="flex-1">
										<input
											list="voice-list"
											class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
											bind:value={TTS_VOICE}
											placeholder={$i18n.t('Select a voice')}
										/>

										<datalist id="voice-list">
											{#each voices as voice}
												<option value={voice.id}>{voice.name}</option>
											{/each}
										</datalist>
									</div>
								</div>
							</div>
							<div class="w-full">
								<div class=" mb-1.5 text-xs font-medium">
									{$i18n.t('Output format')}
									<a
										href="https://learn.microsoft.com/en-us/azure/ai-services/speech-service/rest-text-to-speech?tabs=streaming#audio-outputs"
										target="_blank"
									>
										<small>{$i18n.t('Available list')}</small>
									</a>
								</div>
								<div class="flex w-full">
									<div class="flex-1">
										<input
											list="tts-model-list"
											class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
											bind:value={TTS_AZURE_SPEECH_OUTPUT_FORMAT}
											placeholder={$i18n.t('Select an output format')}
										/>
									</div>
								</div>
							</div>
						</div>
					{/if}
				</div>

				<div class="pt-0.5 flex w-full justify-between">
					<div class="self-center text-xs font-medium">{$i18n.t('Response splitting')}</div>
					<div class="flex items-center relative">
						<select
							class="dark:bg-gray-900 w-fit pr-8 cursor-pointer rounded-sm px-2 p-1 text-xs bg-transparent outline-hidden text-right"
							aria-label={$i18n.t('Select how to split message text for TTS requests')}
							bind:value={TTS_SPLIT_ON}
						>
							{#each Object.values(TTS_RESPONSE_SPLIT) as split}
								<option value={split}
									>{$i18n.t(split.charAt(0).toUpperCase() + split.slice(1))}</option
								>
							{/each}
						</select>
					</div>
				</div>
				<div class="mt-2 mb-1 text-xs text-gray-400 dark:text-gray-500">
					{$i18n.t(
						"Control how message text is split for TTS requests. 'Punctuation' splits into sentences, 'paragraphs' splits into paragraphs, and 'none' keeps the message as a single string."
					)}
				</div>

				<div class="pt-0.5 flex w-full justify-between">
					<div class="self-center text-xs font-medium">{$i18n.t('Audio output format')}</div>
					<div class="flex items-center relative">
						<select
							class="dark:bg-gray-900 w-fit pr-8 cursor-pointer rounded-sm px-2 p-1 text-xs bg-transparent outline-hidden text-right"
							aria-label={$i18n.t('Select TTS audio output format')}
							bind:value={TTS_OUTPUT_FORMAT_VALUE}
						>
							{#each Object.values(TTS_OUTPUT_FORMAT) as format}
								<option value={format}>
									{format === TTS_OUTPUT_FORMAT.DEFAULT
										? $i18n.t('Default')
										: format === TTS_OUTPUT_FORMAT.WEBM
											? 'webm (opus)'
											: format}
								</option>
							{/each}
						</select>
					</div>
				</div>
			</div>
		</div>
	</div>
	<div class="flex justify-end text-sm font-medium">
		<button
			class="px-3.5 py-1.5 text-sm font-medium bg-black hover:bg-gray-900 text-white dark:bg-white dark:text-black dark:hover:bg-gray-100 transition rounded-full"
			type="submit"
		>
			{$i18n.t('Save')}
		</button>
	</div>
</form>
