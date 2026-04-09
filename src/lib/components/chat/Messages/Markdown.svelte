<script>
	import { onDestroy } from 'svelte';
	import { replaceTokens, processResponseContent } from '$lib/utils';
	import { user } from '$lib/stores';

	import { lexChatMarkdown } from '$lib/utils/marked/chat-markdown';

	import MarkdownTokens from './Markdown/MarkdownTokens.svelte';

	export let id = '';
	export let content;
	export let done = true;
	export let model = null;
	export let save = false;
	export let preview = false;

	export let paragraphTag = 'p';
	export let editCodeBlock = true;
	export let topPadding = false;

	export let sourceIds = [];
	export let sourceLabels = [];

	export let onSave = () => {};
	export let onUpdate = () => {};

	export let onPreview = () => {};

	export let onSourceClick = () => {};
	export let onTaskClick = () => {};
	export let onToolCallContextInjectionChange = () => {};
	export let toolCallContextInjectionToggleEnabled = false;

	let tokens = [];
	let processedContent = '';
	let streamParseTimer = null;
	let lastTokenizedAt = 0;

	const STREAM_PARSE_THROTTLE_MS = 100;

	const clearStreamParseTimer = () => {
		if (streamParseTimer) {
			clearTimeout(streamParseTimer);
			streamParseTimer = null;
		}
	};

	const tokenizeContent = (value) => {
		tokens = lexChatMarkdown(value);
		lastTokenizedAt = Date.now();
	};

	const scheduleTokenization = (value, isDone) => {
		clearStreamParseTimer();

		if (!value) {
			tokens = [];
			return;
		}

		if (isDone) {
			tokenizeContent(value);
			return;
		}

		const elapsed = Date.now() - lastTokenizedAt;
		const delay = Math.max(0, STREAM_PARSE_THROTTLE_MS - elapsed);

		if (delay === 0) {
			tokenizeContent(value);
			return;
		}

		streamParseTimer = setTimeout(() => {
			if (processedContent !== value || done) {
				streamParseTimer = null;
				return;
			}

			tokenizeContent(value);
			streamParseTimer = null;
		}, delay);
	};

	$: processedContent = content
		? replaceTokens(processResponseContent(content), model?.name, $user?.name)
		: '';

	$: scheduleTokenization(processedContent, done);

	onDestroy(() => {
		clearStreamParseTimer();
	});
</script>

{#key id}
	<MarkdownTokens
		{tokens}
		{id}
		{done}
		{save}
		{preview}
		{paragraphTag}
		{editCodeBlock}
		{sourceIds}
		{sourceLabels}
		{topPadding}
		{onTaskClick}
		{onSourceClick}
		{onToolCallContextInjectionChange}
		{toolCallContextInjectionToggleEnabled}
		{onSave}
		{onUpdate}
		{onPreview}
	/>
{/key}
