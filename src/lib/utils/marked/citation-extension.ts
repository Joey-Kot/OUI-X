export function citationExtension() {
	return {
		name: 'citation',
		level: 'inline' as const,

		start(src: string) {
			// Trigger on [1] / [1,2] and full-width variants 【1】 / 【1,2】
			return src.search(/(?:\[(\d[\d,\s]*)\]|【(\d[\d,\s]*)】)/);
		},

		tokenizer(src: string) {
			// Avoid matching footnotes
			if (/^\[\^/.test(src)) return;

			// Match one or more adjacent blocks like:
			// "[1][2,3]" / "【1】【2,3】" / "[1]【2]"
			const rule = /^((?:\[(?:\d[\d,\s]*)\]|【(?:\d[\d,\s]*)】))+/;
			const match = rule.exec(src);
			if (!match) return;

			const raw = match[0];

			// Extract ALL bracket groups inside the big match
			const groupRegex = /(?:\[([\d,\s]+)\]|【([\d,\s]+)】)/g;
			const ids: number[] = [];
			let m: RegExpExecArray | null;

			while ((m = groupRegex.exec(raw))) {
				const rawIds = m[1] ?? m[2] ?? '';
				const parsed = rawIds
					.split(',')
					.map((n) => parseInt(n.trim(), 10))
					.filter((n) => !isNaN(n));

				ids.push(...parsed);
			}

			return {
				type: 'citation',
				raw,
				ids // merged list
			};
		},

		renderer(token: any) {
			// e.g. "1,2,3"
			return token.ids.join(',');
		}
	};
}

export default function () {
	return {
		extensions: [citationExtension()]
	};
}
