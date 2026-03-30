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
});

