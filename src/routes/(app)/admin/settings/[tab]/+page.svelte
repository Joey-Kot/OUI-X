<script>
	import { goto } from '$app/navigation';
	import { onMount } from 'svelte';
	import { page } from '$app/stores';

	import Settings from '$lib/components/admin/Settings.svelte';

	onMount(() => {
		const tab = $page.url.pathname.split('/').at(-1);
		if (tab === 'evaluations') {
			goto('/admin/settings/general');
		}
	});
</script>

<Settings />
