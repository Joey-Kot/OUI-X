<script lang="ts">
	import { getContext } from 'svelte';
	const i18n = getContext('i18n');

	import Tooltip from '$lib/components/common/Tooltip.svelte';
	import Cog6 from '$lib/components/icons/Cog6.svelte';
	import ConfirmDialog from '$lib/components/common/ConfirmDialog.svelte';
	import WrenchAlt from '$lib/components/icons/WrenchAlt.svelte';
	import AddMCPToolServerModal from '$lib/components/AddMCPToolServerModal.svelte';

	export let onDelete = () => {};
	export let onSubmit = () => {};
	export let connection = null;

	let showConfigModal = false;
	let showDeleteConfirmDialog = false;
</script>

<AddMCPToolServerModal
	edit
	bind:show={showConfigModal}
	{connection}
	onDelete={() => {
		showDeleteConfirmDialog = true;
	}}
	onSubmit={(c) => {
		connection = c;
		onSubmit(c);
	}}
/>

<ConfirmDialog
	bind:show={showDeleteConfirmDialog}
	on:confirm={() => {
		onDelete();
		showConfigModal = false;
	}}
/>

<div class="flex w-full gap-2 items-center">
	<Tooltip className="w-full relative" content={''} placement="top-start">
		<div class="flex w-full">
			<div
				class="flex-1 relative flex gap-1.5 items-center {!(connection?.config?.enable ?? true)
					? 'opacity-50'
					: ''}"
			>
				<Tooltip content={$i18n.t('MCP')}>
					<WrenchAlt />
				</Tooltip>

				<div class="capitalize outline-hidden w-full bg-transparent">
					{connection?.info?.name ?? connection?.url}
					<span class="text-gray-500">{connection?.info?.id ?? ''}</span>
				</div>
			</div>
		</div>
	</Tooltip>

	<div class="flex gap-1">
		<Tooltip content={$i18n.t('Configure')} className="self-start">
			<button
				class="self-center p-1 bg-transparent hover:bg-gray-100 dark:bg-gray-900 dark:hover:bg-gray-850 rounded-lg transition"
				on:click={() => {
					showConfigModal = true;
				}}
				type="button"
			>
				<Cog6 />
			</button>
		</Tooltip>
	</div>
</div>
