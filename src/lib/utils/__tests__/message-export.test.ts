import { describe, expect, it } from 'vitest';

import {
	buildMessageExportText,
	getCitationSourceCount,
	stripCitationTokens
} from '../message-export';

describe('stripCitationTokens', () => {
	it('removes square and full-width citation tokens when ids are in source range', () => {
		const input = 'Alpha [1] Beta【2】Gamma [3, 4]';
		expect(stripCitationTokens(input, 4)).toBe('Alpha BetaGamma');
	});

	it('removes adjacent citation blocks as one sequence', () => {
		const input = 'Text [1][2, 3]【4】 end';
		expect(stripCitationTokens(input, 4)).toBe('Text end');
	});

	it('keeps footnotes and out-of-range references', () => {
		const input = 'Keep [^1] and [99] here';
		expect(stripCitationTokens(input, 3)).toBe('Keep [^1] and [99] here');
	});

	it('does not alter content when source count is zero', () => {
		const input = 'Year [2026] should stay';
		expect(stripCitationTokens(input, 0)).toBe('Year [2026] should stay');
	});
});

describe('getCitationSourceCount', () => {
	it('counts source entries from document arrays', () => {
		const sources = [{ document: ['a', 'b'] }, { document: ['c'] }];
		expect(getCitationSourceCount(sources)).toBe(3);
	});

	it('falls back to metadata length and single-item count', () => {
		const sources = [{ metadata: [{}, {}] }, { foo: 'bar' }];
		expect(getCitationSourceCount(sources as Array<Record<string, unknown>>)).toBe(3);
	});
});

describe('buildMessageExportText', () => {
	it('removes details and citations and appends watermark when configured', () => {
		const message = {
			content: 'Hello [1]<details type="reasoning">secret</details>',
			sources: [{ document: ['doc-1'] }]
		};

		expect(
			buildMessageExportText(message, {
				includeWatermark: true,
				watermark: 'wm'
			})
		).toBe('Hello\n\nwm');
	});
});

