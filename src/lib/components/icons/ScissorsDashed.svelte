<script lang="ts">
	export let className = 'size-4';
	export let strokeWidth = '2';
</script>

<svg
	xmlns="http://www.w3.org/2000/svg"
	fill="none"
	viewBox="0 0 24 24"
	stroke-width={strokeWidth}
	stroke="currentColor"
	class={className}
>
	<path
		stroke-linecap="round"
		stroke-linejoin="round"
		d="M6.75 8.25a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5Z"
	/>
	<path
		stroke-linecap="round"
		stroke-linejoin="round"
		d="M6.75 20.75a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5Z"
	/>
	<path stroke-linecap="round" stroke-linejoin="round" d="M8.6 7.1 20 18.5" />
	<path stroke-linecap="round" stroke-linejoin="round" d="M8.6 16.9 20 5.5" />
	<path stroke-linecap="round" stroke-dasharray="1.6 2.4" d="M9.75 12h6.5" />
</svg>
