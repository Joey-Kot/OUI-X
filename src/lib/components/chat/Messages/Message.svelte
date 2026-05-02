<script lang="ts">
	import { toast } from 'svelte-sonner';

	import { tick, getContext, onMount, createEventDispatcher } from 'svelte';
	const dispatch = createEventDispatcher();
	const i18n = getContext('i18n');

	import { settings } from '$lib/stores';
	import { copyToClipboard } from '$lib/utils';

	import MultiResponseMessages from './MultiResponseMessages.svelte';
	import ResponseMessage from './ResponseMessage.svelte';
	import UserMessage from './UserMessage.svelte';

	export let chatId;
	export let selectedModels = [];
	export let idx = 0;

	export let history;
	export let messageId;

	export let user;

	export let setInputText: Function = () => {};
	export let gotoMessage;
	export let showPreviousMessage;
	export let showNextMessage;
	export let updateChat;

	export let editMessage;
	export let saveMessage;
	export let deleteMessage;
	export let actionMessage;
	export let submitMessage;

	export let regenerateResponse;
	export let continueResponse;
	export let mergeResponses;

	export let addMessages;
	export let triggerScroll;
	export let readOnly = false;
	export let editCodeBlock = true;
	export let topPadding = false;

	let message = null;
	let parentMessage = null;
	let siblingIds = [];
	let isSingleModelResponseThread = true;

	$: message = history?.messages?.[messageId] ?? null;
	$: parentMessage = message?.parentId ? history?.messages?.[message.parentId] : null;
	$: isSingleModelResponseThread = (parentMessage?.models?.length ?? 1) === 1;
	$: {
		if (!message || !history?.messages) {
			siblingIds = [];
		} else if (message.parentId !== null) {
			siblingIds = parentMessage?.childrenIds ?? [];
		} else {
			siblingIds = Object.values(history.messages)
				.filter((item) => item.parentId === null)
				.map((item) => item.id);
		}
	}
</script>

<div
	role="listitem"
	class="flex flex-col justify-between px-5 mb-3 w-full {($settings?.widescreenMode ?? null)
		? 'max-w-full'
		: 'max-w-5xl'} mx-auto rounded-lg group"
>
	{#if message}
		{#if message.role === 'user'}
			<UserMessage
					{user}
					{chatId}
					{history}
					{messageId}
					isFirstMessage={idx === 0}
					siblings={siblingIds}
					{selectedModels}
					{gotoMessage}
					{showPreviousMessage}
					{showNextMessage}
					{editMessage}
					{deleteMessage}
					{readOnly}
					{editCodeBlock}
					{topPadding}
			/>
		{:else if isSingleModelResponseThread}
			<ResponseMessage
				{chatId}
				{history}
				{messageId}
				{selectedModels}
				isLastMessage={message.id === history.currentId}
				siblings={siblingIds}
				{setInputText}
				{gotoMessage}
				{showPreviousMessage}
				{showNextMessage}
				{updateChat}
				{editMessage}
				{saveMessage}
				{actionMessage}
				{submitMessage}
				{deleteMessage}
				{continueResponse}
				{regenerateResponse}
				{addMessages}
				{readOnly}
				{editCodeBlock}
				{topPadding}
			/>
		{:else}
			{#key messageId}
				<MultiResponseMessages
					bind:history
						{chatId}
						{messageId}
						{selectedModels}
						isLastMessage={message.id === history?.currentId}
						{setInputText}
						{updateChat}
						{editMessage}
					{saveMessage}
					{actionMessage}
					{submitMessage}
					{deleteMessage}
					{continueResponse}
					{regenerateResponse}
					{mergeResponses}
					{triggerScroll}
					{addMessages}
					{readOnly}
					{editCodeBlock}
					{topPadding}
				/>
			{/key}
		{/if}
	{/if}
</div>
