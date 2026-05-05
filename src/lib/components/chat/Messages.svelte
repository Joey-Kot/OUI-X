<script lang="ts">
	import { v4 as uuidv4 } from 'uuid';
	import {
		chats,
		config,
		settings,
		user as _user,
		mobile,
		currentChatPage,
		temporaryChatEnabled
	} from '$lib/stores';
	import { tick, getContext, onMount, createEventDispatcher } from 'svelte';
	const dispatch = createEventDispatcher();

	import { toast } from 'svelte-sonner';
	import { getChatList, updateChatById } from '$lib/apis/chats';
	import { copyToClipboard, extractCurlyBraceWords } from '$lib/utils';
	import { createDisabledContextTruncation } from '$lib/utils/context-truncation';

	import Message from './Messages/Message.svelte';
	import Spinner from '../common/Spinner.svelte';
	import { computeVirtualWindow } from './virtualization';

	import ChatPlaceholder from './ChatPlaceholder.svelte';

	const i18n = getContext('i18n');

	export let className = 'h-full flex pt-8';

	export let chatId = '';
	export let user = $_user;

	export let prompt;
	export let history = {};
	export let selectedModels;
	export let atSelectedModel;

	let messages = [];

	export let setInputText: Function = () => {};

	export let sendMessage: Function;
	export let continueResponse: Function;
	export let regenerateResponse: Function;
	export let mergeResponses: Function;

	export let chatActionHandler: Function;
	export let showMessage: Function = () => {};
	export let submitMessage: Function = () => {};
	export let addMessages: Function = () => {};
	export let contextTruncation = createDisabledContextTruncation();

	export let readOnly = false;
	export let editCodeBlock = true;

	export let topPadding = false;
	export let bottomPadding = false;
	export let autoScroll;

	export let onSelect = (e) => {};

	export let messagesCount: number | null = 20;
	let messagesLoading = false;
	let canLoadMoreOnTop = true;

	const MESSAGE_ESTIMATED_HEIGHT = 220;
	const MESSAGE_OVERSCAN_PX = 2000;
	const MESSAGE_OVERSCAN_ITEMS = 4;

	let containerElement: HTMLElement | null = null;
	let viewportHeight = 0;
	let scrollTop = 0;
	let measuredHeights: Map<string, number> = new Map();

	let visibleStartIdx = 0;
	let renderedMessages = [];
	let topSpacerHeight = 0;
	let bottomSpacerHeight = 0;
	let isAdjustingScrollTop = false;

	const updateViewportMetrics = () => {
		const element = containerElement ?? document.getElementById('messages-container');
		if (!element) {
			return;
		}

		containerElement = element;
		scrollTop = element.scrollTop;
		viewportHeight = element.clientHeight;
	};

	const normalizeContainerScrollTop = (nextScrollTop: number) => {
		if (!containerElement) {
			return;
		}

		if (Math.abs(containerElement.scrollTop - nextScrollTop) < 1) {
			return;
		}

		isAdjustingScrollTop = true;
		containerElement.scrollTop = nextScrollTop;
		scrollTop = nextScrollTop;

		window.requestAnimationFrame(() => {
			isAdjustingScrollTop = false;
		});
	};

	const loadMoreMessages = async () => {
		// scroll slightly down to disable continuous loading
		const element = document.getElementById('messages-container');
		if (!element) {
			return;
		}
		element.scrollTop = element.scrollTop + 100;
		updateViewportMetrics();

		messagesLoading = true;
		if (messagesCount !== null) {
			messagesCount += 20;
		}

		await tick();
		updateViewportMetrics();

		messagesLoading = false;
	};

	const maybeLoadMoreMessages = () => {
		if (!containerElement || messagesLoading || messages.at(0)?.parentId === null) {
			return;
		}

		if (containerElement.scrollTop > 260) {
			canLoadMoreOnTop = true;
			return;
		}

		if (canLoadMoreOnTop && containerElement.scrollTop <= 120) {
			canLoadMoreOnTop = false;
			loadMoreMessages();
		}
	};

	const observeMessageHeight = (node: HTMLElement, messageId: string) => {
		let currentMessageId = messageId;

		const updateHeight = () => {
			const nextHeight = node.offsetHeight;
			if (!nextHeight) {
				return;
			}

			if (measuredHeights.get(currentMessageId) === nextHeight) {
				return;
			}

			const nextMeasuredHeights = new Map(measuredHeights);
			nextMeasuredHeights.set(currentMessageId, nextHeight);
			measuredHeights = nextMeasuredHeights;
		};

		updateHeight();

		let resizeObserver: ResizeObserver | null = null;
		if (typeof ResizeObserver !== 'undefined') {
			resizeObserver = new ResizeObserver(() => {
				updateHeight();
			});
			resizeObserver.observe(node);
		}

		return {
			update(nextMessageId: string) {
				if (nextMessageId === currentMessageId) {
					return;
				}
				currentMessageId = nextMessageId;
				updateHeight();
			},
			destroy() {
				resizeObserver?.disconnect();
			}
		};
	};

	onMount(() => {
		let animationFrameId = 0;

		const handleScroll = () => {
			updateViewportMetrics();

			if (isAdjustingScrollTop) {
				return;
			}

			maybeLoadMoreMessages();
		};

		const bindContainer = () => {
			const element = document.getElementById('messages-container');
			if (!element) {
				return false;
			}

			containerElement = element;
			updateViewportMetrics();
			containerElement.addEventListener('scroll', handleScroll, { passive: true });
			return true;
		};

		const bindWhenReady = () => {
			if (!bindContainer()) {
				animationFrameId = window.requestAnimationFrame(bindWhenReady);
				return;
			}

			maybeLoadMoreMessages();
		};

		const handleResize = () => {
			updateViewportMetrics();
		};

		bindWhenReady();
		window.addEventListener('resize', handleResize, { passive: true });

		return () => {
			if (animationFrameId) {
				window.cancelAnimationFrame(animationFrameId);
			}
			containerElement?.removeEventListener('scroll', handleScroll);
			window.removeEventListener('resize', handleResize);
		};
	});

	$: if (history.currentId) {
		let _messages = [];

		let message = history.messages[history.currentId];
		while (message && (messagesCount !== null ? _messages.length <= messagesCount : true)) {
			_messages.unshift(message);
			message = message.parentId !== null ? history.messages[message.parentId] : null;
		}

		messages = _messages;
	} else {
		messages = [];
	}

	$: {
		const messageIds = new Set(messages.map((message) => message.id));
		let changed = false;
		const nextMeasuredHeights = new Map<string, number>();

		for (const [messageId, height] of measuredHeights.entries()) {
			if (messageIds.has(messageId)) {
				nextMeasuredHeights.set(messageId, height);
			} else {
				changed = true;
			}
		}

		if (changed) {
			measuredHeights = nextMeasuredHeights;
		}
	}

	$: {
		const messageCount = messages.length;

		if (messageCount === 0) {
			visibleStartIdx = 0;
			renderedMessages = [];
			topSpacerHeight = 0;
			bottomSpacerHeight = 0;
		} else if (viewportHeight <= 0) {
			visibleStartIdx = 0;
			renderedMessages = messages;
			topSpacerHeight = 0;
			bottomSpacerHeight = 0;
		} else {
			const offsets = new Array<number>(messageCount + 1);
			offsets[0] = 0;

			for (let index = 0; index < messageCount; index++) {
				const messageId = messages[index].id;
				const measuredHeight = measuredHeights.get(messageId) ?? MESSAGE_ESTIMATED_HEIGHT;
				offsets[index + 1] = offsets[index] + measuredHeight;
			}

			const virtualWindow = computeVirtualWindow({
				offsets,
				messageCount,
				scrollTop,
				viewportHeight,
				overscanPx: Math.max(MESSAGE_OVERSCAN_PX, viewportHeight * 2)
			});

			const containerMaxScrollTop = containerElement
				? Math.max(0, containerElement.scrollHeight - containerElement.clientHeight)
				: virtualWindow.maxScrollableTop;
			const normalizedContainerScrollTop = Math.min(Math.max(0, scrollTop), containerMaxScrollTop);

			if (Math.abs(normalizedContainerScrollTop - scrollTop) >= 1) {
				normalizeContainerScrollTop(normalizedContainerScrollTop);
			}

			const renderStartIndex = Math.max(0, virtualWindow.startIndex - MESSAGE_OVERSCAN_ITEMS);
			const renderEndIndex = Math.min(messageCount, virtualWindow.endIndex + MESSAGE_OVERSCAN_ITEMS);

			visibleStartIdx = renderStartIndex;
			renderedMessages = messages.slice(renderStartIndex, renderEndIndex);
			topSpacerHeight = offsets[renderStartIndex] ?? 0;
			bottomSpacerHeight = Math.max(
				0,
				virtualWindow.totalHeight - (offsets[renderEndIndex] ?? virtualWindow.totalHeight)
			);

			if (renderedMessages.length === 0) {
				const fallbackStartIndex = Math.min(Math.max(virtualWindow.startIndex, 0), messageCount - 1);
				const fallbackEndIndex = Math.min(messageCount, fallbackStartIndex + 1);

				visibleStartIdx = fallbackStartIndex;
				renderedMessages = messages.slice(fallbackStartIndex, fallbackEndIndex);
				topSpacerHeight = offsets[fallbackStartIndex] ?? 0;
				bottomSpacerHeight = Math.max(
					0,
					virtualWindow.totalHeight - (offsets[fallbackEndIndex] ?? virtualWindow.totalHeight)
				);
			}
		}
	}

	$: if (autoScroll && bottomPadding) {
		(async () => {
			await tick();
			scrollToBottom();
		})();
	}

	const scrollToBottom = () => {
		const element = document.getElementById('messages-container');
		if (!element) {
			return;
		}
		element.scrollTop = element.scrollHeight;
		updateViewportMetrics();
	};

	const updateChat = async () => {
		if (!$temporaryChatEnabled) {
			history = history;
			await tick();
			await updateChatById(localStorage.token, chatId, {
				history: history,
				messages: messages
			});

			currentChatPage.set(1);
			await chats.set(await getChatList(localStorage.token, $currentChatPage));
		}
	};

	const gotoMessage = async (message, idx) => {
		// Determine the correct sibling list (either parent's children or root messages)
		let siblings;
		if (message.parentId !== null) {
			siblings = history.messages[message.parentId].childrenIds;
		} else {
			siblings = Object.values(history.messages)
				.filter((msg) => msg.parentId === null)
				.map((msg) => msg.id);
		}

		// Clamp index to a valid range
		idx = Math.max(0, Math.min(idx, siblings.length - 1));

		let messageId = siblings[idx];

		// If we're navigating to a different message
		if (message.id !== messageId) {
			// Drill down to the deepest child of that branch
			let messageChildrenIds = history.messages[messageId].childrenIds;
			while (messageChildrenIds.length !== 0) {
				messageId = messageChildrenIds.at(-1);
				messageChildrenIds = history.messages[messageId].childrenIds;
			}

			history.currentId = messageId;
		}

		await tick();

		// Optional auto-scroll
		if ($settings?.scrollOnBranchChange ?? true) {
			const element = document.getElementById('messages-container');
			autoScroll = element.scrollHeight - element.scrollTop <= element.clientHeight + 50;

			setTimeout(() => {
				scrollToBottom();
			}, 100);
		}
	};

	const showPreviousMessage = async (message) => {
		if (message.parentId !== null) {
			let messageId =
				history.messages[message.parentId].childrenIds[
					Math.max(history.messages[message.parentId].childrenIds.indexOf(message.id) - 1, 0)
				];

			if (message.id !== messageId) {
				let messageChildrenIds = history.messages[messageId].childrenIds;

				while (messageChildrenIds.length !== 0) {
					messageId = messageChildrenIds.at(-1);
					messageChildrenIds = history.messages[messageId].childrenIds;
				}

				history.currentId = messageId;
			}
		} else {
			let childrenIds = Object.values(history.messages)
				.filter((message) => message.parentId === null)
				.map((message) => message.id);
			let messageId = childrenIds[Math.max(childrenIds.indexOf(message.id) - 1, 0)];

			if (message.id !== messageId) {
				let messageChildrenIds = history.messages[messageId].childrenIds;

				while (messageChildrenIds.length !== 0) {
					messageId = messageChildrenIds.at(-1);
					messageChildrenIds = history.messages[messageId].childrenIds;
				}

				history.currentId = messageId;
			}
		}

		await tick();

		if ($settings?.scrollOnBranchChange ?? true) {
			const element = document.getElementById('messages-container');
			autoScroll = element.scrollHeight - element.scrollTop <= element.clientHeight + 50;

			setTimeout(() => {
				scrollToBottom();
			}, 100);
		}
	};

	const showNextMessage = async (message) => {
		if (message.parentId !== null) {
			let messageId =
				history.messages[message.parentId].childrenIds[
					Math.min(
						history.messages[message.parentId].childrenIds.indexOf(message.id) + 1,
						history.messages[message.parentId].childrenIds.length - 1
					)
				];

			if (message.id !== messageId) {
				let messageChildrenIds = history.messages[messageId].childrenIds;

				while (messageChildrenIds.length !== 0) {
					messageId = messageChildrenIds.at(-1);
					messageChildrenIds = history.messages[messageId].childrenIds;
				}

				history.currentId = messageId;
			}
		} else {
			let childrenIds = Object.values(history.messages)
				.filter((message) => message.parentId === null)
				.map((message) => message.id);
			let messageId =
				childrenIds[Math.min(childrenIds.indexOf(message.id) + 1, childrenIds.length - 1)];

			if (message.id !== messageId) {
				let messageChildrenIds = history.messages[messageId].childrenIds;

				while (messageChildrenIds.length !== 0) {
					messageId = messageChildrenIds.at(-1);
					messageChildrenIds = history.messages[messageId].childrenIds;
				}

				history.currentId = messageId;
			}
		}

		await tick();

		if ($settings?.scrollOnBranchChange ?? true) {
			const element = document.getElementById('messages-container');
			autoScroll = element.scrollHeight - element.scrollTop <= element.clientHeight + 50;

			setTimeout(() => {
				scrollToBottom();
			}, 100);
		}
	};

	const editMessage = async (messageId, { content, files }, submit = true) => {
		if ((selectedModels ?? []).filter((id) => id).length === 0) {
			toast.error($i18n.t('Model not selected'));
			return;
		}
		if (history.messages[messageId].role === 'user') {
			if (submit) {
				// New user message
				let userPrompt = content;
				let userMessageId = uuidv4();

				let userMessage = {
					id: userMessageId,
					parentId: history.messages[messageId].parentId,
					childrenIds: [],
					role: 'user',
					content: userPrompt,
					...(files && { files: files }),
					models: selectedModels,
					timestamp: Math.floor(Date.now() / 1000) // Unix epoch
				};

				let messageParentId = history.messages[messageId].parentId;

				if (messageParentId !== null) {
					history.messages[messageParentId].childrenIds = [
						...history.messages[messageParentId].childrenIds,
						userMessageId
					];
				}

				history.messages[userMessageId] = userMessage;
				history.currentId = userMessageId;

				await tick();
				await sendMessage(history, userMessageId);
			} else {
				// Edit user message
				history.messages[messageId].content = content;
				history.messages[messageId].files = files;
				await updateChat();
			}
		} else {
			if (submit) {
				// New response message
				const responseMessageId = uuidv4();
				const message = history.messages[messageId];
				const parentId = message.parentId;

				const responseMessage = {
					...message,
					id: responseMessageId,
					parentId: parentId,
					childrenIds: [],
					files: undefined,
					content: content,
					timestamp: Math.floor(Date.now() / 1000) // Unix epoch
				};

				history.messages[responseMessageId] = responseMessage;
				history.currentId = responseMessageId;

				// Append messageId to childrenIds of parent message
				if (parentId !== null) {
					history.messages[parentId].childrenIds = [
						...history.messages[parentId].childrenIds,
						responseMessageId
					];
				}

				await updateChat();
			} else {
				// Edit response message
				history.messages[messageId].originalContent = history.messages[messageId].content;
				history.messages[messageId].content = content;
				await updateChat();
			}
		}
	};

	const actionMessage = async (actionId, message, event = null) => {
		await chatActionHandler(chatId, actionId, message.model, message.id, event);
	};

	const saveMessage = async (messageId, message) => {
		history.messages[messageId] = message;
		await updateChat();
	};

	const deleteMessage = async (messageId) => {
		const messageToDelete = history.messages[messageId];
		const parentMessageId = messageToDelete.parentId;
		const childMessageIds = messageToDelete.childrenIds ?? [];

		// Collect all grandchildren
		const grandchildrenIds = childMessageIds.flatMap(
			(childId) => history.messages[childId]?.childrenIds ?? []
		);

		// Update parent's children
		if (parentMessageId && history.messages[parentMessageId]) {
			history.messages[parentMessageId].childrenIds = [
				...history.messages[parentMessageId].childrenIds.filter((id) => id !== messageId),
				...grandchildrenIds
			];
		}

		// Update grandchildren's parent
		grandchildrenIds.forEach((grandchildId) => {
			if (history.messages[grandchildId]) {
				history.messages[grandchildId].parentId = parentMessageId;
			}
		});

		// Delete the message and its children
		[messageId, ...childMessageIds].forEach((id) => {
			delete history.messages[id];
		});

		await tick();

		showMessage({ id: parentMessageId });

		// Update the chat
		await updateChat();
	};

	const triggerScroll = () => {
		if (autoScroll) {
			const element = document.getElementById('messages-container');
			autoScroll = element.scrollHeight - element.scrollTop <= element.clientHeight + 50;
			setTimeout(() => {
				scrollToBottom();
			}, 100);
		}
	};

	const shouldShowContextDividerAfter = (messageId: string, messageIdx: number) =>
		contextTruncation?.enabled &&
		contextTruncation?.cutoffMessageId === messageId &&
		Boolean(messages[messageIdx + 1]);
</script>

<div class={className}>
	{#if Object.keys(history?.messages ?? {}).length == 0}
		<ChatPlaceholder modelIds={selectedModels} {atSelectedModel} {onSelect} />
	{:else}
		<div class="w-full pt-2">
			{#key chatId}
				<section class="w-full" aria-labelledby="chat-conversation">
					<h2 class="sr-only" id="chat-conversation">{$i18n.t('Chat Conversation')}</h2>
					{#if messagesLoading}
						<div class="w-full flex justify-center py-1 text-xs animate-pulse items-center gap-2">
							<Spinner className=" size-4" />
							<div>{$i18n.t('Loading...')}</div>
						</div>
					{/if}
					<ul role="log" aria-live="polite" aria-relevant="additions" aria-atomic="false">
						{#if topSpacerHeight > 0}
							<div aria-hidden="true" style="height: {topSpacerHeight}px;" />
						{/if}

						{#each renderedMessages as message, virtualIdx (message.id)}
							<div use:observeMessageHeight={message.id}>
								<Message
									{chatId}
									bind:history
									{selectedModels}
									messageId={message.id}
									idx={visibleStartIdx + virtualIdx}
									{user}
									{setInputText}
									{gotoMessage}
									{showPreviousMessage}
									{showNextMessage}
									{updateChat}
									{editMessage}
									{deleteMessage}
									{actionMessage}
									{saveMessage}
									{submitMessage}
									{regenerateResponse}
									{continueResponse}
									{mergeResponses}
									{addMessages}
									{triggerScroll}
									{readOnly}
									{editCodeBlock}
									{topPadding}
								/>
								{#if shouldShowContextDividerAfter(message.id, visibleStartIdx + virtualIdx)}
									<div class="w-full py-2">
										<div
											class="flex items-center gap-3 text-xs text-gray-500/75 dark:text-gray-400/70"
										>
											<div
												class="h-px flex-1 border-t border-dashed border-gray-300/60 dark:border-gray-600/50"
											/>
											<span
												class="shrink-0 rounded-full bg-white/70 dark:bg-gray-900/70 px-2 py-0.5 backdrop-blur-xs"
											>
												{$i18n.t('Context has been cleared')}
											</span>
											<div
												class="h-px flex-1 border-t border-dashed border-gray-300/60 dark:border-gray-600/50"
											/>
										</div>
									</div>
								{/if}
							</div>
						{/each}

						{#if bottomSpacerHeight > 0}
							<div aria-hidden="true" style="height: {bottomSpacerHeight}px;" />
						{/if}
					</ul>
				</section>
				<div class="pb-18" />
				{#if bottomPadding}
					<div class="  pb-6" />
				{/if}
			{/key}
		</div>
	{/if}
</div>
