<script lang="ts">
	import { toast } from 'svelte-sonner';
	import { onMount, getContext } from 'svelte';

	const i18n = getContext('i18n');

	import Spinner from '$lib/components/common/Spinner.svelte';
	import Tooltip from '$lib/components/common/Tooltip.svelte';
	import Plus from '$lib/components/icons/Plus.svelte';
	import Connection from '$lib/components/chat/Settings/Tools/Connection.svelte';
	import MCPConnection from '$lib/components/admin/Settings/MCPConnection.svelte';

	import AddToolServerModal from '$lib/components/AddToolServerModal.svelte';
	import AddMCPToolServerModal from '$lib/components/AddMCPToolServerModal.svelte';
	import {
		getMCPToolServerConnections,
		getToolServerConnections,
		setMCPToolServerConnections,
		setToolServerConnections
	} from '$lib/apis/configs';

	export let saveSettings: Function;

	let openapiServers = null;
	let mcpServers = null;

	let showOpenAPIConnectionModal = false;
	let showMCPConnectionModal = false;

	const updateOpenAPIHandler = async () => {
		const res = await setToolServerConnections(localStorage.token, {
			TOOL_SERVER_CONNECTIONS: openapiServers
		}).catch(() => {
			toast.error($i18n.t('Failed to save connections'));
			return null;
		});

		if (res) {
			toast.success($i18n.t('Connections saved successfully'));
		}
	};

	const updateMCPHandler = async () => {
		const res = await setMCPToolServerConnections(localStorage.token, {
			MCP_TOOL_SERVER_CONNECTIONS: mcpServers
		}).catch(() => {
			toast.error($i18n.t('Failed to save connections'));
			return null;
		});

		if (res) {
			toast.success($i18n.t('Connections saved successfully'));
		}
	};

	onMount(async () => {
		const [openapiRes, mcpRes] = await Promise.all([
			getToolServerConnections(localStorage.token),
			getMCPToolServerConnections(localStorage.token)
		]);

		openapiServers = openapiRes?.TOOL_SERVER_CONNECTIONS ?? [];
		mcpServers = mcpRes?.MCP_TOOL_SERVER_CONNECTIONS ?? [];
	});
</script>

<AddToolServerModal
	bind:show={showOpenAPIConnectionModal}
	onSubmit={async (server) => {
		openapiServers = [...(openapiServers ?? []), server];
		await updateOpenAPIHandler();
	}}
	openapiOnly
/>

<AddMCPToolServerModal
	bind:show={showMCPConnectionModal}
	onSubmit={async (server) => {
		mcpServers = [...(mcpServers ?? []), server];
		await updateMCPHandler();
	}}
/>

<form class="flex flex-col h-full justify-between text-sm" on:submit|preventDefault>
	<div class="overflow-y-scroll scrollbar-hidden h-full">
		{#if openapiServers !== null && mcpServers !== null}
			<div class="mb-3">
				<div class="mt-0.5 mb-2.5 text-base font-medium">{$i18n.t('General')}</div>

				<hr class="border-gray-100/30 dark:border-gray-850/30 my-2" />

				<div class="mb-4 flex flex-col w-full justify-between">
					<div class="flex justify-between items-center mb-0.5">
						<div class="font-medium">{$i18n.t('Manage Tool Servers')}</div>

						<Tooltip content={$i18n.t('Add Connection')}>
							<button class="px-1" on:click={() => { showOpenAPIConnectionModal = true; }} type="button">
								<Plus />
							</button>
						</Tooltip>
					</div>

					<div class="flex flex-col gap-1">
						{#each openapiServers as server, idx}
							<Connection
								bind:connection={server}
								openapiOnly
								onSubmit={updateOpenAPIHandler}
								onDelete={() => {
									openapiServers = openapiServers.filter((_, i) => i !== idx);
									updateOpenAPIHandler();
								}}
							/>
						{/each}
					</div>

					<div class="my-1.5">
						<div class="text-xs text-gray-500">
							{$i18n.t('Connect to your own OpenAPI compatible external tool servers.')}
						</div>
					</div>
				</div>

				<div class="mb-2.5 flex flex-col w-full justify-between">
					<div class="flex justify-between items-center mb-0.5">
						<div class="font-medium">{$i18n.t('Manage MCP Tools')}</div>

						<Tooltip content={$i18n.t('Add Connection')}>
							<button class="px-1" on:click={() => { showMCPConnectionModal = true; }} type="button">
								<Plus />
							</button>
						</Tooltip>
					</div>

					<div class="flex flex-col gap-1">
						{#each mcpServers as server, idx}
							<MCPConnection
								bind:connection={server}
								onSubmit={updateMCPHandler}
								onDelete={() => {
									mcpServers = mcpServers.filter((_, i) => i !== idx);
									updateMCPHandler();
								}}
							/>
						{/each}
					</div>

					<div class="my-1.5">
						<div class="text-xs text-gray-500">
							{$i18n.t('Manage MCP Streamable HTTP and SSE tool servers separately from remote servers.')}
						</div>
					</div>
				</div>
			</div>
		{:else}
			<div class="flex h-full justify-center">
				<div class="my-auto">
					<Spinner className="size-6" />
				</div>
			</div>
		{/if}
	</div>
</form>
