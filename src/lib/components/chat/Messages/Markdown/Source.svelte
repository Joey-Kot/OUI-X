<script lang="ts">
	import { decodeString } from '$lib/utils';

	export let id;

	export let title: string = 'N/A';
	export let refIndex: number | null = null;
	export let showIndex = true;
	export let disabled = false;

	export let onClick: Function = () => {};

	// Helper function to return only the domain from a URL
	function getDomain(url: string): string {
		const domain = url.replace('http://', '').replace('https://', '').split(/[/?#]/)[0];

		if (domain.startsWith('www.')) {
			return domain.slice(4);
		}
		return domain;
	}

	const getDisplayTitle = (title: string) => {
		if (!title) return 'N/A';
		if (title.length > 30) {
			return title.slice(0, 15) + '...' + title.slice(-10);
		}
		return title;
	};

	// Helper function to check if text is a URL and return the domain
	function formattedTitle(title: string): string {
		if (title.startsWith('http')) {
			return getDomain(title);
		}

		return title;
	}

	const handleClick = () => {
		if (disabled) return;
		onClick(id);
	};
</script>

{#if title !== 'N/A'}
	<button
		class="text-[10px] w-fit translate-y-[2px] px-2 py-0.5 dark:bg-white/5 dark:text-white/80 dark:hover:text-white bg-gray-50 text-black/80 hover:text-black transition rounded-xl {disabled
			? 'opacity-60 cursor-not-allowed'
			: ''}"
		on:click={handleClick}
	>
		<span class="inline-flex items-center gap-0.5">
			{getDisplayTitle(formattedTitle(decodeString(title)))}
			{#if showIndex && typeof refIndex === 'number'}
				<span>[{refIndex}]</span>
			{/if}
		</span>
	</button>
{/if}
