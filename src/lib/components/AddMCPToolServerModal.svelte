<script lang="ts">
	import { toast } from 'svelte-sonner';
	import { getContext, onMount } from 'svelte';
	const i18n = getContext('i18n');

	import Modal from '$lib/components/common/Modal.svelte';
	import Tooltip from '$lib/components/common/Tooltip.svelte';
	import SensitiveInput from '$lib/components/common/SensitiveInput.svelte';
	import Switch from '$lib/components/common/Switch.svelte';
	import AccessControl from './workspace/common/AccessControl.svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import XMark from '$lib/components/icons/XMark.svelte';
	import Textarea from './common/Textarea.svelte';
	import { registerOAuthClient, verifyMCPToolServerConnection } from '$lib/apis/configs';
	import {
		registerUserMCPOAuthClient,
		verifyUserMCPToolServerConnection
	} from '$lib/apis/users';

	export let onSubmit: Function = () => {};
	export let onDelete: Function = () => {};
	export let show = false;
	export let edit = false;
	export let connection = null;
	export let userScoped = false;

	let url = '';
	let transport = 'streamable_http';
	let auth_type = 'none';
	let key = '';
	let headers = '';

	let id = '';
	let name = '';
	let description = '';
	let oauthClientInfo = null;

	let enable = true;
	let accessControl = {};

	let toolsConfig = {};
	let verified = false;
	let verifiedTools = [];

	let loading = false;
	let verifying = false;

	const parseHeaders = () => {
		if (!headers) return undefined;
		try {
			const parsed = JSON.parse(headers);
			if (typeof parsed !== 'object' || Array.isArray(parsed)) {
				throw new Error('invalid headers');
			}
			headers = JSON.stringify(parsed, null, 2);
			return parsed;
		} catch {
			toast.error($i18n.t('Headers must be a valid JSON object'));
			return null;
		}
	};

	const basePayload = (parsedHeaders) => ({
		url,
		transport,
		auth_type,
		headers: parsedHeaders,
		key,
		info: {
			id,
			name,
			description,
			...(oauthClientInfo ? { oauth_client_info: oauthClientInfo } : {})
		},
		config: {
			enable,
			access_control: accessControl,
			tools: toolsConfig
		}
	});

	const verifyHandler = async () => {
		if (!url) {
			toast.error($i18n.t('Please enter a valid URL'));
			return;
		}

		const parsedHeaders = parseHeaders();
		if (headers && parsedHeaders === null) return;

		verifying = true;
		const verifyConnection = userScoped
			? verifyUserMCPToolServerConnection
			: verifyMCPToolServerConnection;

		const res = await verifyConnection(localStorage.token, basePayload(parsedHeaders)).catch(
			() => {
				toast.error($i18n.t('Connection failed'));
				return null;
			}
		);
		verifying = false;

		if (!res) return;

		if (res.oauth_server_metadata) {
			verified = false;
			verifiedTools = [];
			toast.success($i18n.t('Connection successful'));
			return;
		}

		verified = true;
		verifiedTools = res.specs ?? [];

		const nextToolsConfig = { ...toolsConfig };
		for (const tool of verifiedTools) {
			if (!nextToolsConfig[tool.name]) {
				nextToolsConfig[tool.name] = { enabled: true };
			}
		}
		toolsConfig = nextToolsConfig;
		toast.success($i18n.t('Connection successful'));
	};

	const registerOAuthClientHandler = async () => {
		if (!url || !id) {
			toast.error($i18n.t('Please enter a valid URL and ID'));
			return;
		}

		const registerClient = userScoped ? registerUserMCPOAuthClient : registerOAuthClient;

		const res = await registerClient(
			localStorage.token,
			{
				url,
				client_id: id
			},
			userScoped ? undefined : 'mcp'
		).catch(() => {
			toast.error($i18n.t('Registration failed'));
			return null;
		});

		if (res) {
			oauthClientInfo = res?.oauth_client_info ?? null;
			toast.success($i18n.t('Registration successful'));
		}
	};

	const submitHandler = async () => {
		loading = true;
		url = url.replace(/\/$/, '');

		if (id.includes(':') || id.includes('|')) {
			toast.error($i18n.t('ID cannot contain ":" or "|" characters'));
			loading = false;
			return;
		}

		if (auth_type === 'oauth_2.1' && !oauthClientInfo) {
			toast.error($i18n.t('Please register the OAuth client'));
			loading = false;
			return;
		}

		const parsedHeaders = parseHeaders();
		if (headers && parsedHeaders === null) {
			loading = false;
			return;
		}

		await onSubmit({
			...basePayload(parsedHeaders),
			verify_cache: {
				verified,
				verified_at: Math.floor(Date.now() / 1000),
				tools: verifiedTools
			}
		});

		loading = false;
		show = false;
	};

	const init = () => {
		if (!connection) return;

		url = connection?.url ?? '';
		transport = connection?.transport ?? 'streamable_http';
		auth_type = connection?.auth_type ?? 'none';
		headers = connection?.headers ? JSON.stringify(connection.headers, null, 2) : '';
		key = connection?.key ?? '';

		id = connection?.info?.id ?? '';
		name = connection?.info?.name ?? '';
		description = connection?.info?.description ?? '';
		oauthClientInfo = connection?.info?.oauth_client_info ?? null;

		enable = connection?.config?.enable ?? true;
		accessControl = connection?.config?.access_control ?? {};
		toolsConfig = connection?.config?.tools ?? {};

		verified = connection?.verify_cache?.verified ?? false;
		verifiedTools = connection?.verify_cache?.tools ?? [];
	};

	$: if (show) {
		init();
	}

	onMount(() => {
		init();
	});
</script>

<Modal size="sm" bind:show>
	<div class="flex justify-between dark:text-gray-100 px-5 pt-4 pb-2">
		<h1 class="text-lg font-medium self-center font-primary">
			{#if edit}
				{$i18n.t('Edit Connection')}
			{:else}
				{$i18n.t('Add Connection')}
			{/if}
		</h1>

		<button class="self-center" aria-label={$i18n.t('Close Configure Connection Modal')} on:click={() => (show = false)}>
			<XMark className={'size-5'} />
		</button>
	</div>

	<form class="px-5 pb-4 flex flex-col gap-2" on:submit|preventDefault={submitHandler}>
		<label class="text-xs text-gray-500">{$i18n.t('URL')}</label>
		<div class="flex items-center gap-2">
			<input class="w-full text-sm bg-transparent outline-hidden" bind:value={url} placeholder={$i18n.t('API Base URL')} required />
			<Tooltip content={$i18n.t("Verify Connection")} className="shrink-0 flex items-center mr-1">
				<button
					class="self-center p-1 bg-transparent hover:bg-gray-100 dark:bg-gray-900 dark:hover:bg-gray-850 rounded-lg transition disabled:opacity-60 disabled:cursor-not-allowed"
					type="button"
					on:click={verifyHandler}
					aria-label={$i18n.t("Verify Connection")}
					disabled={verifying}
				>
					{#if verifying}
						<Spinner className="size-4" />
					{:else}
						<svg
							xmlns="http://www.w3.org/2000/svg"
							viewBox="0 0 20 20"
							fill="currentColor"
							class="w-4 h-4"
							aria-hidden="true"
						>
							<path
								fill-rule="evenodd"
								d="M15.312 11.424a5.5 5.5 0 01-9.201 2.466l-.312-.311h2.433a.75.75 0 000-1.5H3.989a.75.75 0 00-.75.75v4.242a.75.75 0 001.5 0v-2.43l.31.31a7 7 0 0011.712-3.138.75.75 0 00-1.449-.39zm1.23-3.723a.75.75 0 00.219-.53V2.929a.75.75 0 00-1.5 0V5.36l-.31-.31A7 7 0 003.239 8.188a.75.75 0 101.448.389A5.5 5.5 0 0113.89 6.11l.311.31h-2.432a.75.75 0 000 1.5h4.243a.75.75 0 00.53-.219z"
								clip-rule="evenodd"
							/>
						</svg>
					{/if}
				</button>
			</Tooltip>
			<Switch bind:state={enable} />
		</div>

		<label class="text-xs text-gray-500">{$i18n.t('Transport')}</label>
		<select class="text-sm bg-transparent" bind:value={transport}>
			<option value="streamable_http">{$i18n.t('Streamable HTTP')}</option>
			<option value="sse">{$i18n.t('SSE')}</option>
		</select>

		<label class="text-xs text-gray-500">{$i18n.t('Auth')}</label>
		<div class="flex gap-2">
			<select class="text-sm bg-transparent" bind:value={auth_type}>
				<option value="none">{$i18n.t('None')}</option>
				<option value="bearer">{$i18n.t('Bearer')}</option>
				<option value="session">{$i18n.t('Session')}</option>
				<option value="system_oauth">{$i18n.t('OAuth')}</option>
				<option value="oauth_2.1">{$i18n.t('OAuth 2.1')}</option>
			</select>
			{#if auth_type === 'bearer'}
				<SensitiveInput bind:value={key} placeholder={$i18n.t('API Key')} required={false} />
			{:else if auth_type === 'oauth_2.1'}
				<button type="button" class="text-xs underline" on:click={registerOAuthClientHandler}>{$i18n.t('Register Client')}</button>
			{/if}
		</div>

		<label class="text-xs text-gray-500">{$i18n.t('Headers')}</label>
		<Textarea className="w-full text-sm outline-hidden" bind:value={headers} placeholder={$i18n.t('Enter additional headers in JSON format')} required={false} minSize={30} />

		<label class="text-xs text-gray-500">{$i18n.t('ID')}</label>
		<input class="w-full text-sm bg-transparent outline-hidden" type="text" bind:value={id} placeholder={$i18n.t('Enter ID')} required />

		<label class="text-xs text-gray-500">{$i18n.t('Name')}</label>
		<input class="w-full text-sm bg-transparent outline-hidden" type="text" bind:value={name} placeholder={$i18n.t('Enter name')} required />

		<label class="text-xs text-gray-500">{$i18n.t('Description')}</label>
		<input class="w-full text-sm bg-transparent outline-hidden" type="text" bind:value={description} placeholder={$i18n.t('Enter description')} />

		<div class="mt-2">
			<div class="text-xs text-gray-500 mb-1">{$i18n.t('Tool Enablement')}</div>
			{#if verified && verifiedTools.length > 0}
				{#each verifiedTools as tool}
					<div class="flex justify-between py-1">
						<div class="text-xs">{tool.name}</div>
						<Switch
							state={toolsConfig?.[tool.name]?.enabled ?? true}
							on:change={(e) => {
								toolsConfig = { ...toolsConfig, [tool.name]: { enabled: e.detail } };
							}}
						/>
					</div>
				{/each}
			{:else}
				<div class="text-xs text-gray-500">{$i18n.t('Verify before configuring tools')}</div>
			{/if}
		</div>

		<AccessControl bind:accessControl />

		<div class="flex justify-end gap-2 pt-2">
			{#if edit}
				<button type="button" class="px-3 py-1.5 rounded-full bg-white dark:bg-black" on:click={() => { onDelete(); show = false; }}>{$i18n.t('Delete')}</button>
			{/if}
			<button type="submit" class="px-3 py-1.5 rounded-full bg-black text-white dark:bg-white dark:text-black" disabled={loading}>{$i18n.t('Save')}</button>
		</div>
	</form>
</Modal>
