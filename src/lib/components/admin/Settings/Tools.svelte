<script lang="ts">
	import { toast } from 'svelte-sonner';
	import { onMount, getContext } from 'svelte';

	const i18n = getContext('i18n');

	import Spinner from '$lib/components/common/Spinner.svelte';
	import Tooltip from '$lib/components/common/Tooltip.svelte';
	import Plus from '$lib/components/icons/Plus.svelte';
	import MCPConnection from '$lib/components/admin/Settings/MCPConnection.svelte';

	import AddMCPToolServerModal from '$lib/components/AddMCPToolServerModal.svelte';
	import {
		getMCPToolServerConnections,
		getToolCallingConfig,
		setMCPToolServerConnections,
		setToolCallingConfig
	} from '$lib/apis/configs';

	export let saveSettings: Function;

	let mcpServers = null;
	let toolCallingConfig = {
		TOOL_CALL_TIMEOUT_SECONDS: 60,
		MAX_TOOL_CALLS_PER_ROUND: 20
	};

	let showMCPConnectionModal = false;

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

	const updateToolCallingConfigHandler = async () => {
		const res = await setToolCallingConfig(localStorage.token, {
			TOOL_CALL_TIMEOUT_SECONDS: Number(toolCallingConfig.TOOL_CALL_TIMEOUT_SECONDS),
			MAX_TOOL_CALLS_PER_ROUND: Number(toolCallingConfig.MAX_TOOL_CALLS_PER_ROUND)
		}).catch(() => {
			toast.error($i18n.t('Failed to save tool calling config'));
			return null;
		});

		if (res) {
			toolCallingConfig = {
				TOOL_CALL_TIMEOUT_SECONDS: res.TOOL_CALL_TIMEOUT_SECONDS,
				MAX_TOOL_CALLS_PER_ROUND: res.MAX_TOOL_CALLS_PER_ROUND
			};
			toast.success($i18n.t('Tool calling config saved successfully'));
		}
	};

	onMount(async () => {
		const [mcpRes, toolCallingRes] = await Promise.all([
			getMCPToolServerConnections(localStorage.token),
			getToolCallingConfig(localStorage.token)
		]);

		mcpServers = mcpRes?.MCP_TOOL_SERVER_CONNECTIONS ?? [];
		toolCallingConfig = {
			TOOL_CALL_TIMEOUT_SECONDS: toolCallingRes?.TOOL_CALL_TIMEOUT_SECONDS ?? 60,
			MAX_TOOL_CALLS_PER_ROUND: toolCallingRes?.MAX_TOOL_CALLS_PER_ROUND ?? 20
		};
	});
</script>

<AddMCPToolServerModal
	bind:show={showMCPConnectionModal}
	onSubmit={async (server) => {
		mcpServers = [...(mcpServers ?? []), server];
		await updateMCPHandler();
	}}
/>

<form
	class="flex flex-col h-full justify-between text-sm"
	on:submit|preventDefault={async () => {
		await updateToolCallingConfigHandler();
	}}
>
	<div class="overflow-y-scroll scrollbar-hidden h-full">
		{#if mcpServers !== null}
			<div class="mb-3">
				<div class="mt-0.5 mb-2.5 text-base font-medium">{$i18n.t('General')}</div>

				<hr class="border-gray-100/30 dark:border-gray-850/30 my-2" />

				<div class="mb-2.5 flex flex-col w-full justify-between">
					<div class="flex justify-between items-center mb-0.5">
						<div class="font-medium">{$i18n.t('Manage MCP Tools')}</div>


						<Tooltip content={$i18n.t('Add Connection')}>
							<button class="px-1" on:click={() => { showMCPConnectionModal = true; }} type="button">
								<Plus />
							</button>
						</Tooltip>
					</div>

					<hr class="border-gray-100/30 dark:border-gray-850/30 my-2" />

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
							{$i18n.t('Manage MCP Streamable HTTP and SSE tool servers.')}
						</div>
					</div>
				</div>

				<div class="mb-2.5 flex flex-col w-full justify-between">
					<div class="font-medium mb-1">{$i18n.t('Tool Calling Config')}</div>

					<hr class="border-gray-100/30 dark:border-gray-850/30 my-2" />
					<div class="flex flex-col gap-2">
						<div class="mb-2.5 flex w-full flex-col">
							<div class="self-start text-xs font-medium mb-1">
								{$i18n.t('Tool Calling Timeout')} ({$i18n.t('seconds')})
							</div>
							<input
								class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
								type="number"
								min="1"
								max="600"
								bind:value={toolCallingConfig.TOOL_CALL_TIMEOUT_SECONDS}
							/>
						</div>
						<div class="mb-2.5 flex w-full flex-col">
							<div class="self-start text-xs font-medium mb-1">
								{$i18n.t('Maximum Number of Tool Calling')} ({$i18n.t('Maximum number of calls per round for a single tool')})
							</div>
							<input
								class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-hidden"
								type="number"
								min="1"
								max="100"
								bind:value={toolCallingConfig.MAX_TOOL_CALLS_PER_ROUND}
							/>
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
