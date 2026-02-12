<script lang="ts">
	import { DropdownMenu } from 'bits-ui';
	import { flyAndScale } from '$lib/utils/transitions';
	import { getContext, createEventDispatcher } from 'svelte';

	import Dropdown from '$lib/components/common/Dropdown.svelte';
	import Tooltip from '$lib/components/common/Tooltip.svelte';
	import GarbageBin from '$lib/components/icons/GarbageBin.svelte';
	import Download from '$lib/components/icons/Download.svelte';
	import DocumentDuplicate from '$lib/components/icons/DocumentDuplicate.svelte';
	import EllipsisHorizontal from '$lib/components/icons/EllipsisHorizontal.svelte';

	const dispatch = createEventDispatcher();
	const i18n = getContext('i18n');

	export let onClose: Function = () => {};
	export let showClone = true;
	export let showDownload = true;
	export let showDelete = true;

	let show = false;
</script>

<Dropdown
	bind:show
	on:change={(e) => {
		if (e.detail === false) {
			onClose();
		}
	}}
	align="end"
>
	<Tooltip content={$i18n.t('More')}>
		<slot
			><button
				class="self-center w-fit text-sm p-1.5 dark:text-gray-300 dark:hover:text-white hover:bg-black/5 dark:hover:bg-white/5 rounded-xl"
				type="button"
				on:click={(e) => {
					e.stopPropagation();
					show = true;
				}}
			>
				<EllipsisHorizontal className="size-5" />
			</button>
		</slot>
	</Tooltip>

	<div slot="content">
		<DropdownMenu.Content
			class="w-full max-w-[170px] rounded-2xl px-1 py-1 border border-gray-100  dark:border-gray-800 z-50 bg-white dark:bg-gray-850 dark:text-white shadow-lg"
			side="bottom"
			align="end"
			transition={flyAndScale}
		>
			{#if showClone}
				<DropdownMenu.Item
					class="flex  gap-2  items-center px-3 py-1.5 text-sm   cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800 rounded-xl"
					on:click={() => {
						dispatch('clone');
					}}
				>
					<DocumentDuplicate />
					<div class="flex items-center">{$i18n.t('Clone')}</div>
				</DropdownMenu.Item>
			{/if}

			{#if showDownload}
				<DropdownMenu.Item
					class="flex  gap-2  items-center px-3 py-1.5 text-sm   cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800 rounded-xl"
					on:click={() => {
						dispatch('download');
					}}
				>
					<Download />
					<div class="flex items-center">{$i18n.t('Download')}</div>
				</DropdownMenu.Item>
			{/if}

			{#if showDelete}
				{#if showClone || showDownload}
					<hr class="border-gray-50/30 dark:border-gray-800/30 my-1" />
				{/if}

				<DropdownMenu.Item
					class="flex  gap-2  items-center px-3 py-1.5 text-sm   cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800 rounded-xl"
					on:click={() => {
						dispatch('delete');
					}}
				>
					<GarbageBin />
					<div class="flex items-center">{$i18n.t('Delete')}</div>
				</DropdownMenu.Item>
			{/if}
		</DropdownMenu.Content>
	</div>
</Dropdown>
