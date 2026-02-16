import { describe, expect, it } from 'vitest';

import { citationExtension } from '../marked/citation-extension';

describe('citationExtension', () => {
	const extension = citationExtension();
	const tokenizer = extension.tokenizer as (src: string) => any;

	it('tokenizes double-bracket citations', () => {
		const token = tokenizer('[[1]] tail');
		expect(token?.type).toBe('citation');
		expect(token?.raw).toBe('[[1]]');
		expect(token?.ids).toEqual([1]);
	});

	it('tokenizes adjacent double-bracket citations as one token', () => {
		const token = tokenizer('[[1]][[2, 3]] tail');
		expect(token?.raw).toBe('[[1]][[2, 3]]');
		expect(token?.ids).toEqual([1, 2, 3]);
	});

	it('tokenizes full-width double-bracket citations', () => {
		const token = tokenizer('【【1】】 tail');
		expect(token?.type).toBe('citation');
		expect(token?.raw).toBe('【【1】】');
		expect(token?.ids).toEqual([1]);
	});

	it('tokenizes mixed adjacent citation formats as one token', () => {
		const token = tokenizer('[[1]]【【2, 3】】 tail');
		expect(token?.raw).toBe('[[1]]【【2, 3】】');
		expect(token?.ids).toEqual([1, 2, 3]);
	});

	it('does not tokenize single-bracket content', () => {
		expect(tokenizer('[1]')).toBeUndefined();
		expect(tokenizer('【1】')).toBeUndefined();
		expect(tokenizer('[2026]')).toBeUndefined();
		expect(tokenizer('[^1]')).toBeUndefined();
	});
});
