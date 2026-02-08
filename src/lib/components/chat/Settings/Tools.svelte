<script lang="ts"> 
	import { toast } from 'svelte-sonner';
	import { onMount, getContext } from 'svelte';
	import { getToolServersData } from '$lib/apis';
	import { getTools } from '$lib/apis/tools';

	const i18n = getContext('i18n');

	import { settings, toolServers, tools as toolsStore } from '$lib/stores';

	import Spinner from '$lib/components/common/Spinner.svelte';
	import Tooltip from '$lib/components/common/Tooltip.svelte';
	import Plus from '$lib/components/icons/Plus.svelte';
	import Connection from './Tools/Connection.svelte';
	import MCPConnection from '$lib/components/admin/Settings/MCPConnection.svelte';

	import AddToolServerModal from '$lib/components/AddToolServerModal.svelte';
	import AddMCPToolServerModal from '$lib/components/AddMCPToolServerModal.svelte';

	export let saveSettings: Function;

	let servers = null;
	let mcpServers = null;
	let showConnectionModal = false;
	let showMCPConnectionModal = false;

	const getDescriptionClass = () =>
		($settings?.highContrastMode ?? false)
			? 'text-xs text-gray-800 dark:text-gray-100'
			: 'text-xs text-gray-500';

	const updateOpenAPIHandler = async () => {
		await saveSettings({
			toolServers: servers
		});

		let toolServersData = await getToolServersData($settings?.toolServers ?? []);
		toolServersData = toolServersData.filter((data) => {
			if (data.error) {
				toast.error(
					$i18n.t('Failed to connect to {{URL}} OpenAPI tool server', {
						URL: data?.url
					})
				);
				return false;
			}

			return true;
		});
		toolServers.set(toolServersData);
	};

	const updateMCPHandler = async () => {
		await saveSettings({
			mcpToolServers: mcpServers
		});

		const updatedTools = await getTools(localStorage.token).catch(() => null);
		if (updatedTools) {
			toolsStore.set(updatedTools);
		}
	};

	onMount(async () => {
		servers = $settings?.toolServers ?? [];
		mcpServers = $settings?.mcpToolServers ?? [];
	});
</script>

<AddToolServerModal
	bind:show={showConnectionModal}
	onSubmit={async (server) => {
		servers = [...servers, server];
		await updateOpenAPIHandler();
	}}
	direct
/>

<AddMCPToolServerModal
	bind:show={showMCPConnectionModal}
	userScoped
	onSubmit={async (server) => {
		mcpServers = [...mcpServers, server];
		await updateMCPHandler();
	}}
/>

<form
	id="tab-tools"
	class="flex flex-col h-full justify-between text-sm"
	on:submit|preventDefault={async () => {
		await updateOpenAPIHandler();
		await updateMCPHandler();
	}}
>
	<div class="overflow-y-scroll scrollbar-hidden h-full">
		{#if servers !== null && mcpServers !== null}
			<div class="pr-1.5">
				<div class="flex justify-between items-center mb-0.5">
					<div class="font-medium">{$i18n.t('Manage Tool Servers')}</div>

					<Tooltip content={$i18n.t('Add Connection')}>
						<button
							aria-label={$i18n.t('Add Connection')}
							class="px-1"
							on:click={() => {
								showConnectionModal = true;
							}}
							type="button"
						>
							<Plus />
						</button>
					</Tooltip>
				</div>

				<div class="flex flex-col gap-1.5">
					{#each servers as server, idx}
						<Connection
							bind:connection={server}
							direct
							onSubmit={() => {
								updateOpenAPIHandler();
							}}
							onDelete={() => {
								servers = servers.filter((_, i) => i !== idx);
								updateOpenAPIHandler();
							}}
						/>
					{/each}
				</div>

				<div class="my-1.5">
					<div class={getDescriptionClass()}>
						{$i18n.t('Connect to your own OpenAPI compatible external tool servers.')}
						<br />
						{$i18n.t('CORS must be properly configured by the provider to allow requests from Open WebUI.')}
					</div>
				</div>

				<div class="text-xs text-gray-600 dark:text-gray-300 mb-2">
					<a class="underline" href="https://github.com/open-webui/openapi-servers" target="_blank"
						>{$i18n.t('Learn more about OpenAPI tool servers.')}</a
					>
				</div>

				<div class="mt-4">
					<div class="flex justify-between items-center mb-0.5">
						<div class="font-medium">{$i18n.t('Manage MCP Tools')}</div>

						<Tooltip content={$i18n.t('Add Connection')}>
							<button
								aria-label={$i18n.t('Add Connection')}
								class="px-1"
								on:click={() => {
									showMCPConnectionModal = true;
								}}
								type="button"
							>
								<Plus />
							</button>
						</Tooltip>
					</div>

					<div class="flex flex-col gap-1.5">
						{#each mcpServers as server, idx}
							<MCPConnection
								bind:connection={server}
								userScoped
								onSubmit={() => {
									updateMCPHandler();
								}}
								onDelete={() => {
									mcpServers = mcpServers.filter((_, i) => i !== idx);
									updateMCPHandler();
								}}
							/>
						{/each}
					</div>

					<div class="my-1.5">
						<div class={getDescriptionClass()}>
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

	<div class="flex justify-end pt-3 text-sm font-medium">
		<button
			class="px-3.5 py-1.5 text-sm font-medium bg-black hover:bg-gray-900 text-white dark:bg-white dark:text-black dark:hover:bg-gray-100 transition rounded-full"
			type="submit"
		>
			{$i18n.t('Save')}
		</button>
	</div>
</form>
