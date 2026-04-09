<script>
	import { onDestroy, onMount, tick, getContext } from 'svelte';
	const i18n = getContext('i18n');

	import Markdown from './Markdown.svelte';
	import {
		artifactCode,
		chatId,
		mobile,
		settings,
		showArtifacts,
		showControls,
		showEmbeds,
		showOverview
	} from '$lib/stores';
	import FloatingButtons from '../ContentRenderer/FloatingButtons.svelte';
	import { createMessagesList } from '$lib/utils';

	export let id;
	export let content;

	export let history;
	export let messageId;

	export let selectedModels = [];

	export let done = true;
	export let model = null;
	export let sources = null;

	export let save = false;
	export let preview = false;
	export let floatingButtons = true;

	export let editCodeBlock = true;
	export let topPadding = false;

	export let onSave = (e) => {};
	export let onSourceClick = (e) => {};
	export let onTaskClick = (e) => {};
	export let onAddMessages = (e) => {};
	export let onToolCallContextInjectionChange = (e) => {};
	export let toolCallContextInjectionToggleEnabled = false;

	let contentContainerElement;
	let floatingButtonsElement;
	let hasDocumentListeners = false;
	let hasContentSelectionListener = false;
	let sourceIds = [];
	let sourceLabels = [];
	let floatingButtonMessages = [];

	const bindContentSelectionListener = () => {
		if (!contentContainerElement || hasContentSelectionListener) {
			return;
		}

		contentContainerElement.addEventListener('mouseup', updateButtonPosition);
		hasContentSelectionListener = true;
	};

	const unbindContentSelectionListener = () => {
		if (!contentContainerElement || !hasContentSelectionListener) {
			return;
		}

		contentContainerElement.removeEventListener('mouseup', updateButtonPosition);
		hasContentSelectionListener = false;
	};

	const bindDocumentListeners = () => {
		if (hasDocumentListeners) {
			return;
		}

		document.addEventListener('mouseup', documentMouseupHandler);
		document.addEventListener('keydown', keydownHandler);
		hasDocumentListeners = true;
	};

	const unbindDocumentListeners = () => {
		if (!hasDocumentListeners) {
			return;
		}

		document.removeEventListener('mouseup', documentMouseupHandler);
		document.removeEventListener('keydown', keydownHandler);
		hasDocumentListeners = false;
	};

	const documentMouseupHandler = (event) => {
		const buttonsContainerElement = document.getElementById(`floating-buttons-${id}`);
		if (
			contentContainerElement?.contains(event.target) ||
			buttonsContainerElement?.contains(event.target)
		) {
			return;
		}

		closeFloatingButtons();
	};

	const updateButtonPosition = (event) => {
		if (!contentContainerElement?.contains(event.target)) {
			closeFloatingButtons();
			return;
		}

		setTimeout(async () => {
			await tick();

			if (!contentContainerElement?.contains(event.target)) return;

			const selection = window.getSelection();

			if (selection && selection.rangeCount > 0 && selection.toString().trim().length > 0) {
				const buttonsContainerElement = document.getElementById(`floating-buttons-${id}`);
				if (!buttonsContainerElement) {
					return;
				}

				const range = selection.getRangeAt(0);
				const rect = range.getBoundingClientRect();

				const parentRect = contentContainerElement.getBoundingClientRect();

				// Adjust based on parent rect
				const top = rect.bottom - parentRect.top;
				const left = rect.left - parentRect.left;

				buttonsContainerElement.style.display = 'block';

				// Calculate space available on the right
				const spaceOnRight = parentRect.width - left;
				let halfScreenWidth = $mobile ? window.innerWidth / 2 : window.innerWidth / 3;

				if (spaceOnRight < halfScreenWidth) {
					const right = parentRect.right - rect.right;
					buttonsContainerElement.style.right = `${right}px`;
					buttonsContainerElement.style.left = 'auto'; // Reset left
				} else {
					// Enough space, position using 'left'
					buttonsContainerElement.style.left = `${left}px`;
					buttonsContainerElement.style.right = 'auto'; // Reset right
				}
				buttonsContainerElement.style.top = `${top + 5}px`; // +5 to add some spacing

				bindDocumentListeners();
			} else {
				closeFloatingButtons();
			}
		}, 0);
	};

	const closeFloatingButtons = () => {
		const buttonsContainerElement = document.getElementById(`floating-buttons-${id}`);
		if (buttonsContainerElement) {
			buttonsContainerElement.style.display = 'none';
		}

		if (floatingButtonsElement) {
			// check if closeHandler is defined

			if (typeof floatingButtonsElement?.closeHandler === 'function') {
				// call the closeHandler function
				floatingButtonsElement?.closeHandler();
			}
		}

		unbindDocumentListeners();
	};

	const keydownHandler = (e) => {
		if (e.key === 'Escape') {
			closeFloatingButtons();
		}
	};

	onMount(() => {
		if (floatingButtons) {
			bindContentSelectionListener();
		}
	});

	$: if (contentContainerElement) {
		if (floatingButtons) {
			bindContentSelectionListener();
		} else {
			closeFloatingButtons();
			unbindContentSelectionListener();
		}
	}

	const buildSourceMetadata = (sourceItems = [], currentModel = null) => {
		const ids = [];
		const labels = [];
		const citationsDisabled = currentModel?.info?.meta?.capabilities?.citations == false;

		for (const source of sourceItems ?? []) {
			for (let index = 0; index < (source.document ?? []).length; index++) {
				if (citationsDisabled) {
					ids.push('N/A');
					labels.push({ title: 'N/A', index: labels.length + 1 });
					continue;
				}

				const metadata = source.metadata?.[index];
				const sourceId = metadata?.source ?? 'N/A';
				let title = source?.source?.name ?? sourceId;

				if (metadata?.name) {
					title = metadata.name;
				} else if (sourceId.startsWith('http://') || sourceId.startsWith('https://')) {
					title = sourceId;
				}

				ids.push(title);
				labels.push({ title, index: labels.length + 1 });
			}
		}

		return { ids, labels };
	};

	$: ({ ids: sourceIds, labels: sourceLabels } = buildSourceMetadata(sources, model));
	$: floatingButtonMessages =
		floatingButtons && model && history && messageId ? createMessagesList(history, messageId) : [];

	onDestroy(() => {
		unbindContentSelectionListener();
		unbindDocumentListeners();
	});
</script>

<div bind:this={contentContainerElement}>
	<Markdown
		{id}
		{content}
		{model}
		{save}
		{preview}
		{done}
		{editCodeBlock}
		{topPadding}
		{sourceIds}
		{sourceLabels}
		{onSourceClick}
		{onTaskClick}
		{onToolCallContextInjectionChange}
		{toolCallContextInjectionToggleEnabled}
		{onSave}
		onUpdate={async (token) => {
			const { lang, text: code } = token;

			if (
				($settings?.detectArtifacts ?? true) &&
				(['html', 'svg'].includes(lang) || (lang === 'xml' && code.includes('svg'))) &&
				!$mobile &&
				$chatId
			) {
				await tick();
				showArtifacts.set(true);
				showControls.set(true);
			}
		}}
		onPreview={async (value) => {
			console.log('Preview', value);
			await artifactCode.set(value);
			await showControls.set(true);
			await showArtifacts.set(true);
			await showOverview.set(false);
			await showEmbeds.set(false);
		}}
	/>
</div>

{#if floatingButtons && model}
	<FloatingButtons
		bind:this={floatingButtonsElement}
		{id}
		{messageId}
		actions={$settings?.floatingActionButtons ?? []}
		model={(selectedModels ?? []).includes(model?.id)
			? model?.id
			: (selectedModels ?? []).length > 0
				? selectedModels.at(0)
				: model?.id}
		messages={floatingButtonMessages}
		onAdd={({ modelId, parentId, messages }) => {
			console.log(modelId, parentId, messages);
			onAddMessages({ modelId, parentId, messages });
			closeFloatingButtons();
		}}
	/>
{/if}
