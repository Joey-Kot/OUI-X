import { describe, expect, it } from 'vitest';

import { createOpenAITextStream } from '$lib/apis/streaming';

const makeSseStream = (events: string[]) =>
	new ReadableStream<Uint8Array>({
		start(controller) {
			const encoder = new TextEncoder();
			controller.enqueue(encoder.encode(events.join('')));
			controller.close();
		}
	});

describe('createOpenAITextStream responses parsing', () => {
	it('emits text for response.output_text.delta events', async () => {
		const stream = makeSseStream([
			'event: response.output_text.delta\n',
			'data: {"type":"response.output_text.delta","delta":"hello "}\n\n',
			'event: response.output_text.delta\n',
			'data: {"type":"response.output_text.delta","delta":"world"}\n\n',
			'event: response.completed\n',
			'data: {"type":"response.completed","response":{"output":[]}}\n\n'
		]);

		const iterator = await createOpenAITextStream(stream, false);
		let text = '';
		for await (const update of iterator) {
			if (update.value) {
				text += update.value;
			}
			if (update.done) {
				break;
			}
		}

		expect(text).toBe('hello world');
	});

	it('falls back to response.completed output_text when no deltas emitted', async () => {
		const stream = makeSseStream([
			'event: response.completed\n',
			'data: {"type":"response.completed","response":{"output":[{"type":"message","content":[{"type":"output_text","text":"fallback text"}]}]}}\n\n'
		]);

		const iterator = await createOpenAITextStream(stream, false);
		let text = '';
		for await (const update of iterator) {
			if (update.value) {
				text += update.value;
			}
			if (update.done) {
				break;
			}
		}

		expect(text).toBe('fallback text');
	});

	it('does not emit reasoning from response.reasoning.summary config string', async () => {
		const stream = makeSseStream([
			'event: response.completed\n',
			'data: {"type":"response.completed","response":{"output":[],"reasoning":{"summary":"detailed"}}}\n\n'
		]);

		const iterator = await createOpenAITextStream(stream, false);
		let reasoning = '';
		for await (const update of iterator) {
			if (update.reasoning) {
				reasoning += update.reasoning;
			}
			if (update.done) {
				break;
			}
		}

		expect(reasoning).toBe('');
	});

	it('emits reasoning from output reasoning summary text', async () => {
		const stream = makeSseStream([
			'event: response.completed\n',
			'data: {"type":"response.completed","response":{"output":[{"type":"reasoning","summary":[{"type":"summary_text","text":"line 1"},{"type":"summary_text","text":"line 2"}]}]}}\n\n'
		]);

		const iterator = await createOpenAITextStream(stream, false);
		let reasoning = '';
		for await (const update of iterator) {
			if (update.reasoning) {
				reasoning += update.reasoning;
			}
			if (update.done) {
				break;
			}
		}

		expect(reasoning).toBe('line 1\nline 2');
	});

	it('streams reasoning from response.reasoning_summary_text.delta events', async () => {
		const stream = makeSseStream([
			'event: response.reasoning_summary_text.delta\n',
			'data: {"type":"response.reasoning_summary_text.delta","delta":"step "}\n\n',
			'event: response.reasoning_summary_text.delta\n',
			'data: {"type":"response.reasoning_summary_text.delta","delta":"one"}\n\n',
			'event: response.completed\n',
			'data: {"type":"response.completed","response":{"output":[]}}\n\n'
		]);

		const iterator = await createOpenAITextStream(stream, false);
		let reasoning = '';
		for await (const update of iterator) {
			if (update.reasoning) {
				reasoning += update.reasoning;
			}
			if (update.done) {
				break;
			}
		}

		expect(reasoning).toBe('step one');
	});

	it('inserts newline between streamed reasoning summary parts', async () => {
		const stream = makeSseStream([
			'event: response.reasoning_summary_part.added\n',
			'data: {"type":"response.reasoning_summary_part.added"}\n\n',
			'event: response.reasoning_summary_text.delta\n',
			'data: {"type":"response.reasoning_summary_text.delta","delta":"line 1"}\n\n',
			'event: response.reasoning_summary_part.done\n',
			'data: {"type":"response.reasoning_summary_part.done"}\n\n',
			'event: response.reasoning_summary_part.added\n',
			'data: {"type":"response.reasoning_summary_part.added"}\n\n',
			'event: response.reasoning_summary_text.delta\n',
			'data: {"type":"response.reasoning_summary_text.delta","delta":"line 2"}\n\n',
			'event: response.completed\n',
			'data: {"type":"response.completed","response":{"output":[]}}\n\n'
		]);

		const iterator = await createOpenAITextStream(stream, false);
		let reasoning = '';
		for await (const update of iterator) {
			if (update.reasoning) {
				reasoning += update.reasoning;
			}
			if (update.done) {
				break;
			}
		}

		expect(reasoning).toBe('line 1\nline 2');
	});

	it('does not duplicate streamed reasoning on response.completed fallback extraction', async () => {
		const stream = makeSseStream([
			'event: response.reasoning_summary_text.delta\n',
			'data: {"type":"response.reasoning_summary_text.delta","delta":"already streamed"}\n\n',
			'event: response.completed\n',
			'data: {"type":"response.completed","response":{"output":[{"type":"reasoning","summary":[{"type":"summary_text","text":"already streamed"}]}]}}\n\n'
		]);

		const iterator = await createOpenAITextStream(stream, false);
		let reasoning = '';
		for await (const update of iterator) {
			if (update.reasoning) {
				reasoning += update.reasoning;
			}
			if (update.done) {
				break;
			}
		}

		expect(reasoning).toBe('already streamed');
	});
});
