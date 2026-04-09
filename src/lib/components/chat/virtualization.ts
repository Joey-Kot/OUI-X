export interface VirtualWindowInput {
	offsets: number[];
	messageCount: number;
	scrollTop: number;
	viewportHeight: number;
	overscanPx: number;
}

export interface VirtualWindowResult {
	startIndex: number;
	endIndex: number;
	topSpacerHeight: number;
	bottomSpacerHeight: number;
	normalizedScrollTop: number;
	maxScrollableTop: number;
	totalHeight: number;
}

export const lowerBound = (offsets: number[], target: number): number => {
	let low = 0;
	let high = Math.max(0, offsets.length - 1);

	while (low < high) {
		const mid = Math.floor((low + high) / 2);
		if (offsets[mid] < target) {
			low = mid + 1;
		} else {
			high = mid;
		}
	}

	return low;
};

export const computeVirtualWindow = ({
	offsets,
	messageCount,
	scrollTop,
	viewportHeight,
	overscanPx
}: VirtualWindowInput): VirtualWindowResult => {
	if (messageCount <= 0) {
		return {
			startIndex: 0,
			endIndex: 0,
			topSpacerHeight: 0,
			bottomSpacerHeight: 0,
			normalizedScrollTop: 0,
			maxScrollableTop: 0,
			totalHeight: 0
		};
	}

	const totalHeight = offsets[messageCount] ?? offsets.at(-1) ?? 0;
	const safeViewportHeight = Math.max(0, viewportHeight);
	const maxScrollableTop = Math.max(0, totalHeight - safeViewportHeight);
	const normalizedScrollTop = Math.min(Math.max(0, scrollTop), maxScrollableTop);

	if (safeViewportHeight <= 0) {
		return {
			startIndex: 0,
			endIndex: messageCount,
			topSpacerHeight: 0,
			bottomSpacerHeight: 0,
			normalizedScrollTop,
			maxScrollableTop,
			totalHeight
		};
	}

	const startOffset = Math.max(0, normalizedScrollTop - overscanPx);
	const endOffset = normalizedScrollTop + safeViewportHeight + overscanPx;

	const rawStartIndex = lowerBound(offsets, startOffset);
	const startIndex = Math.min(Math.max(rawStartIndex, 0), messageCount - 1);

	const rawEndIndex = lowerBound(offsets, endOffset) + 1;
	let endIndex = Math.min(Math.max(rawEndIndex, startIndex + 1), messageCount);
	if (endIndex <= startIndex) {
		endIndex = Math.min(messageCount, startIndex + 1);
	}

	const topSpacerHeight = offsets[startIndex] ?? 0;
	const bottomSpacerHeight = Math.max(0, totalHeight - (offsets[endIndex] ?? totalHeight));

	return {
		startIndex,
		endIndex,
		topSpacerHeight,
		bottomSpacerHeight,
		normalizedScrollTop,
		maxScrollableTop,
		totalHeight
	};
};
