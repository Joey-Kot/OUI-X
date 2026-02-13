<script lang="ts">
	import { getContext } from 'svelte';
	import { goto } from '$app/navigation';
	import { toast } from 'svelte-sonner';

	import Modal from '$lib/components/common/Modal.svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import Switch from '$lib/components/common/Switch.svelte';
	import { getKnowledgeBases } from '$lib/apis/knowledge';
	import { addChatToCollection } from '$lib/utils/chat-to-collection';
	import { decodeString } from '$lib/utils';
	import type { SerializableChat } from '$lib/utils/chat-export';

	const i18n = getContext('i18n');

	export let show = false;
	export let chat: SerializableChat | null = null;
	export let onSuccess: Function = () => {};
	export let onError: Function = () => {};

	type KnowledgeBase = {
		id: string;
		name: string;
		description?: string;
		write_access?: boolean;
	};

	type WritableKnowledgeBase = KnowledgeBase & {
		decodedName: string;
		decodedDescription: string;
	};

	let loading = false;
	let loadingMore = false;
	let submitting = false;
	let selectedKnowledgeId = '';
	let page = 1;
	let total = 0;
	let collections: KnowledgeBase[] = [];
	let writableCollections: WritableKnowledgeBase[] = [];
	let writableCollectionMap: Map<string, WritableKnowledgeBase> = new Map();
	let initialized = false;
	let addThinkingContent = true;
	let addToolCallingContent = true;

	$: writableCollections = collections
		.filter((item) => item?.write_access)
		.map((item) => ({
			...item,
			decodedName: decodeString(item.name),
			decodedDescription: item.description ? decodeString(item.description) : ''
		}));

	$: writableCollectionMap = new Map(writableCollections.map((item) => [item.id, item]));

	const resetState = () => {
		loading = false;
		loadingMore = false;
		submitting = false;
		selectedKnowledgeId = '';
		page = 1;
		total = 0;
		collections = [];
		writableCollections = [];
		writableCollectionMap = new Map();
		initialized = false;
		addThinkingContent = true;
		addToolCallingContent = true;
	};

	const fetchCollections = async (targetPage = 1) => {
		if (targetPage === 1) {
			loading = true;
		} else {
			loadingMore = true;
		}

		const res = await getKnowledgeBases(localStorage.token, targetPage).catch((error) => {
			toast.error(`${error}`);
			return null;
		});

		if (res) {
			total = res.total ?? 0;
			const nextItems = (res.items ?? []) as KnowledgeBase[];
			collections = targetPage === 1 ? nextItems : [...collections, ...nextItems];
			page = targetPage;

			if (targetPage === 1 && nextItems.filter((item) => item?.write_access).length === 0) {
				toast.info($i18n.t('No writable collections found. Please create one in Knowledge workspace.'));
			}
		}

		loading = false;
		loadingMore = false;
		initialized = true;
	};

	const loadMore = async () => {
		if (loadingMore || collections.length >= total) {
			return;
		}
		await fetchCollections(page + 1);
	};

	const close = () => {
		show = false;
	};

	const goToKnowledgeWorkspace = async () => {
		close();
		await goto('/workspace/knowledge');
	};

	const submit = async () => {
		if (!chat || !selectedKnowledgeId || submitting) {
			return;
		}

		submitting = true;
		const selectedKnowledge = writableCollectionMap.get(selectedKnowledgeId);

		await addChatToCollection({
			token: localStorage.token,
			chat,
			knowledgeId: selectedKnowledgeId,
			includeThinkingContent: addThinkingContent,
			includeToolCallingContent: addToolCallingContent
		})
			.then(() => {
				toast.success(
					$i18n.t('Added to collection: {{name}}', {
						name: selectedKnowledge?.decodedName ?? selectedKnowledgeId
					})
				);
				onSuccess({ knowledgeId: selectedKnowledgeId });
				close();
			})
			.catch((error) => {
				toast.error(`${error}`);
				onError(error);
			})
			.finally(() => {
				submitting = false;
			});
	};

	$: if (show && !initialized) {
		fetchCollections(1);
	}

	$: if (!show) {
		resetState();
	}
</script>

<Modal bind:show size="sm">
	<div class="p-4 flex flex-col gap-3">
		<div class="text-base font-medium">{$i18n.t('Add To Collection')}</div>

		{#if loading}
			<div class="py-8 flex justify-center">
				<Spinner className="size-4" />
			</div>
		{:else}
			{#if writableCollections.length === 0}
				<div class="text-sm text-gray-600 dark:text-gray-300">
					{$i18n.t('No writable collections found. Please create one in Knowledge workspace.')}
				</div>
				<div class="flex justify-end">
					<button
						type="button"
						class="px-3 py-1.5 rounded-lg bg-black text-white dark:bg-white dark:text-black text-sm"
						on:click={goToKnowledgeWorkspace}
					>
						{$i18n.t('Go to Knowledge')}
					</button>
				</div>
			{:else}
				<div class="max-h-64 overflow-y-auto border border-gray-100 dark:border-gray-800 rounded-xl p-1 flex flex-col gap-1">
					{#each writableCollections as collection (collection.id)}
						<button
							type="button"
							class="w-full text-left px-3 py-2 rounded-lg text-sm hover:bg-gray-50 dark:hover:bg-gray-800 {selectedKnowledgeId ===
							collection.id
								? 'bg-gray-100 dark:bg-gray-800'
								: ''}"
							on:click={() => {
								selectedKnowledgeId = collection.id;
							}}
						>
							<div class="font-medium line-clamp-1">{collection.decodedName}</div>
							{#if collection.description}
								<div class="text-xs text-gray-500 dark:text-gray-400 line-clamp-1">
									{collection.decodedDescription}
								</div>
							{/if}
						</button>
					{/each}
				</div>

				{#if collections.length < total}
					<div class="flex justify-center">
						<button
							type="button"
							class="text-xs text-gray-500 hover:text-gray-800 dark:hover:text-gray-200"
							on:click={loadMore}
							disabled={loadingMore}
						>
							{#if loadingMore}
								{$i18n.t('Loading...')}
							{:else}
								{$i18n.t('Load more')}
							{/if}
						</button>
					</div>
				{/if}

				<div class="space-y-1">
					<div class="py-0.5 flex w-full justify-between">
						<div id="add-thinking-content-label" class="self-center text-xs">
							{$i18n.t('Add Thinking Content')}
						</div>

						<div class="flex items-center gap-2 p-1">
							<Switch
								ariaLabelledbyId="add-thinking-content-label"
								tooltip={true}
								bind:state={addThinkingContent}
							/>
						</div>
					</div>

					<div class="py-0.5 flex w-full justify-between">
						<div id="add-tool-calling-content-label" class="self-center text-xs">
							{$i18n.t('Add Tool Calling Content')}
						</div>

						<div class="flex items-center gap-2 p-1">
							<Switch
								ariaLabelledbyId="add-tool-calling-content-label"
								tooltip={true}
								bind:state={addToolCallingContent}
							/>
						</div>
					</div>
				</div>

				<div class="flex justify-end gap-2 pt-1">
					<button
						type="button"
						class="px-3 py-1.5 rounded-lg text-sm border border-gray-200 dark:border-gray-700"
						on:click={close}
						disabled={submitting}
					>
						{$i18n.t('Cancel')}
					</button>
					<button
						type="button"
						class="px-3 py-1.5 rounded-lg text-sm bg-black text-white dark:bg-white dark:text-black disabled:opacity-60"
						on:click={submit}
						disabled={!selectedKnowledgeId || submitting}
					>
						{#if submitting}
							{$i18n.t('Adding...')}
						{:else}
							{$i18n.t('Add To Collection')}
						{/if}
					</button>
				</div>
			{/if}
		{/if}
	</div>
</Modal>
