<script lang="ts">
	import { DropdownMenu } from "bits-ui";
	import { getContext, tick } from "svelte";
	import { flyAndScale } from "$lib/utils/transitions";

	import { tools as _tools } from "$lib/stores";

	import { getOAuthClientAuthorizationUrl } from "$lib/apis/configs";
	import { getTools } from "$lib/apis/tools";

	import Knobs from "$lib/components/icons/Knobs.svelte";
	import Dropdown from "$lib/components/common/Dropdown.svelte";
	import Tooltip from "$lib/components/common/Tooltip.svelte";
	import Switch from "$lib/components/common/Switch.svelte";
	import Spinner from "$lib/components/common/Spinner.svelte";
	import Wrench from "$lib/components/icons/Wrench.svelte";

	const i18n = getContext("i18n");

	export let selectedToolIds: string[] = [];
	export let onShowValves: Function;
	export let onClose: Function;
	export let closeOnOutsideClick = true;

	let show = false;
	let tools = null;

	$: if (show) {
		init();
	}

	const init = async () => {
		if ($_tools === null) {
			await _tools.set(await getTools(localStorage.token));
		}

		if ($_tools) {
			tools = $_tools.reduce((a, tool) => {
				a[tool.id] = {
					name: tool.name,
					description: tool.meta.description,
					enabled: selectedToolIds.includes(tool.id),
					...tool
				};
				return a;
			}, {});
		}

		selectedToolIds = selectedToolIds.filter((id) => Object.keys(tools ?? {}).includes(id));
	};
</script>

<Dropdown
	bind:show
	{closeOnOutsideClick}
	on:change={(e) => {
		if (e.detail === false) {
			onClose();
		}
	}}
>
	<Tooltip content={$i18n.t("Tools")} placement="top">
		<slot />
	</Tooltip>
	<div slot="content">
		<DropdownMenu.Content
			class="w-full max-w-70 rounded-2xl px-1 py-1 border border-gray-100 dark:border-gray-800 z-50 bg-white dark:bg-gray-850 dark:text-white shadow-lg max-h-72 overflow-y-auto overflow-x-hidden scrollbar-thin"
			sideOffset={4}
			alignOffset={-6}
			side="bottom"
			align="start"
			transition={flyAndScale}
		>
			{#if tools}
				{#each Object.keys(tools) as toolId}
					<button
						class="relative flex w-full justify-between gap-2 items-center px-3 py-1.5 text-sm cursor-pointer rounded-xl hover:bg-gray-50 dark:hover:bg-gray-800/50"
						on:click={async (e) => {
							if (!(tools[toolId]?.authenticated ?? true)) {
								e.preventDefault();

								const oauthClientId = tools[toolId]?.oauth_client_id;
								let authUrl = "";

								if (oauthClientId) {
									authUrl = getOAuthClientAuthorizationUrl(oauthClientId);
								} else {
									const parts = toolId.split(":");
									const serverId = parts?.at(-1) ?? toolId;
									authUrl = getOAuthClientAuthorizationUrl(serverId, "mcp");
								}

								window.open(authUrl, "_self", "noopener");
							} else {
								tools[toolId].enabled = !tools[toolId].enabled;

								const state = tools[toolId].enabled;
								await tick();

								if (state) {
									selectedToolIds = [...selectedToolIds, toolId];
								} else {
									selectedToolIds = selectedToolIds.filter((id) => id !== toolId);
								}
							}
						}}
					>
						{#if !(tools[toolId]?.authenticated ?? true)}
							<div class="absolute inset-0 opacity-50 rounded-xl cursor-pointer z-10" />
						{/if}
						<div class="flex-1 truncate">
							<div class="flex flex-1 gap-2 items-center">
								<Tooltip content={tools[toolId]?.name ?? ""} placement="top">
									<div class="shrink-0">
										<Wrench />
									</div>
								</Tooltip>
								<Tooltip content={tools[toolId]?.description ?? ""} placement="top-start">
									<div class="truncate">{tools[toolId].name}</div>
								</Tooltip>
							</div>
						</div>

						{#if tools[toolId]?.has_user_valves}
							<div class="shrink-0">
								<Tooltip content={$i18n.t("Valves")}>
									<button
										class="self-center w-fit text-sm text-gray-600 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 transition rounded-full"
										type="button"
										on:click={(e) => {
											e.stopPropagation();
											e.preventDefault();
											onShowValves({
												type: "tool",
												id: toolId
											});
										}}
									>
										<Knobs />
									</button>
								</Tooltip>
							</div>
						{/if}

						<div class="shrink-0">
							<Switch state={tools[toolId].enabled} />
						</div>
					</button>
				{/each}
			{:else}
				<div class="py-4">
					<Spinner />
				</div>
			{/if}
		</DropdownMenu.Content>
	</div>
</Dropdown>
