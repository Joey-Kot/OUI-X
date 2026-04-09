import { describe, expect, it } from 'vitest';

import { computeVirtualWindow } from '../virtualization';

describe('computeVirtualWindow', () => {
	it('computes a valid range in normal viewport scrolling', () => {
		const offsets = [0, 120, 260, 430, 650];

		const result = computeVirtualWindow({
			offsets,
			messageCount: 4,
			scrollTop: 180,
			viewportHeight: 220,
			overscanPx: 100
		});

		expect(result.startIndex).toBeGreaterThanOrEqual(0);
		expect(result.startIndex).toBeLessThan(4);
		expect(result.endIndex).toBeGreaterThan(result.startIndex);
		expect(result.endIndex).toBeLessThanOrEqual(4);
	});

	it('normalizes scrollTop when it exceeds maximum scrollable range', () => {
		const offsets = [0, 200, 400, 600];

		const result = computeVirtualWindow({
			offsets,
			messageCount: 3,
			scrollTop: 10_000,
			viewportHeight: 240,
			overscanPx: 80
		});

		expect(result.maxScrollableTop).toBe(360);
		expect(result.normalizedScrollTop).toBe(360);
		expect(result.endIndex).toBeGreaterThan(result.startIndex);
	});

	it('never returns an empty range for one-message lists', () => {
		const offsets = [0, 180];

		const result = computeVirtualWindow({
			offsets,
			messageCount: 1,
			scrollTop: 9999,
			viewportHeight: 200,
			overscanPx: 100
		});

		expect(result.startIndex).toBe(0);
		expect(result.endIndex).toBe(1);
	});

	it('returns full range when viewportHeight is zero', () => {
		const offsets = [0, 100, 260, 420];

		const result = computeVirtualWindow({
			offsets,
			messageCount: 3,
			scrollTop: 200,
			viewportHeight: 0,
			overscanPx: 100
		});

		expect(result.startIndex).toBe(0);
		expect(result.endIndex).toBe(3);
	});

	it('never produces an empty render window across a wide scroll range', () => {
		const offsets = [0, 110, 240, 390, 560];
		const messageCount = 4;

		for (let scrollTop = 0; scrollTop <= 2_000; scrollTop += 25) {
			const result = computeVirtualWindow({
				offsets,
				messageCount,
				scrollTop,
				viewportHeight: 180,
				overscanPx: 60
			});

			expect(result.startIndex).toBeGreaterThanOrEqual(0);
			expect(result.startIndex).toBeLessThan(messageCount);
			expect(result.endIndex).toBeGreaterThan(result.startIndex);
			expect(result.endIndex).toBeLessThanOrEqual(messageCount);
		}
	});
});
