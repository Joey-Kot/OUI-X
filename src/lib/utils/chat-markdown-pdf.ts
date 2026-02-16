import DOMPurify from 'dompurify';
import hljs from 'highlight.js';
import { Marked } from 'marked';

import markedExtension from '$lib/utils/marked/extension';
import markedKatexExtension from '$lib/utils/marked/katex-extension';
import { disableSingleTilde } from '$lib/utils/marked/strikethrough-extension';
import { mentionExtension } from '$lib/utils/marked/mention-extension';
import footnoteExtension from '$lib/utils/marked/footnote-extension';
import citationExtension from '$lib/utils/marked/citation-extension';

const createMarkedParser = () => {
	const options = {
		throwOnError: false,
		breaks: true
	};

	const parser = new Marked({
		gfm: true,
		breaks: true
	} as any);

	parser.setOptions({
		highlight(code: string, lang: string) {
			const language = hljs.getLanguage(lang) ? lang : 'plaintext';
			return hljs.highlight(code, { language }).value;
		}
	} as any);

	parser.use(markedKatexExtension(options));
	parser.use(markedExtension(options));
	parser.use(citationExtension(options));
	parser.use(footnoteExtension(options));
	parser.use(disableSingleTilde);
	parser.use({
		extensions: [mentionExtension({ triggerChar: '@' }), mentionExtension({ triggerChar: '#' })]
	});

	return parser;
};

const markedParser = createMarkedParser();

const escapeHtml = (value: string): string =>
	value
		.replaceAll('&', '&amp;')
		.replaceAll('<', '&lt;')
		.replaceAll('>', '&gt;')
		.replaceAll('"', '&quot;')
		.replaceAll("'", '&#39;');

const waitForImages = async (doc: Document, timeoutMs = 6000): Promise<void> => {
	const images = Array.from(doc.images ?? []);
	if (images.length === 0) {
		return;
	}

	await Promise.all(
		images.map(
			(image) =>
				new Promise<void>((resolve) => {
					if (image.complete) {
						resolve();
						return;
					}

					let done = false;
					const finish = () => {
						if (done) {
							return;
						}

						done = true;
						image.removeEventListener('load', finish);
						image.removeEventListener('error', finish);
						resolve();
					};

					image.addEventListener('load', finish, { once: true });
					image.addEventListener('error', finish, { once: true });
					setTimeout(finish, timeoutMs);
				})
		)
	);
};

const buildPrintDocument = (title: string, htmlContent: string) => {
	const safeTitle = escapeHtml(title || 'Chat');

	return `<!doctype html>
<html>
<head>
	<meta charset="utf-8" />
	<meta name="viewport" content="width=device-width, initial-scale=1" />
	<base href="${escapeHtml(window.location.href)}" />
	<style>
		@page {
			margin: 12mm 14mm;
		}

		html, body {
			background: #fff;
			color: #111;
			font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
			font-size: 14px;
			line-height: 1.55;
			margin: 0;
			padding: 0;
		}

		.document {
			max-width: 980px;
			margin: 0 auto;
		}

		h1 {
			font-size: 28px;
			line-height: 1.3;
			margin: 0 0 18px;
			word-break: break-word;
		}

		h2, h3, h4, h5, h6 {
			line-height: 1.35;
			margin-top: 1.3em;
			margin-bottom: 0.6em;
		}

		p, ul, ol, blockquote, pre, table {
			margin: 0.7em 0;
		}

		a {
			color: #0f4cbd;
			text-decoration: underline;
			word-break: break-word;
		}

		img {
			max-width: 100%;
			height: auto;
			page-break-inside: avoid;
			break-inside: avoid;
		}

		table {
			width: 100%;
			border-collapse: collapse;
			table-layout: auto;
		}

		th, td {
			border: 1px solid #d7d7d7;
			padding: 8px 10px;
			text-align: left;
			vertical-align: top;
		}

		blockquote {
			border-left: 4px solid #dadde3;
			padding-left: 12px;
			color: #4a5568;
		}

		pre, code {
			font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
		}

		pre {
			background: #f7f7f8;
			border-radius: 8px;
			padding: 12px;
			white-space: pre-wrap;
			word-break: break-word;
			overflow-wrap: anywhere;
			page-break-inside: auto;
			break-inside: auto;
		}

		code {
			white-space: pre-wrap;
			word-break: break-word;
			overflow-wrap: anywhere;
		}

		hr {
			border: 0;
			border-top: 1px solid #e4e4e4;
			margin: 1.2em 0;
		}

		details {
			display: block;
		}

		details > summary {
			font-weight: 600;
			cursor: default;
		}

		.citation-chip {
			font-size: 11px;
		}
	</style>
</head>
<body>
	<div class="document">
		<h1>${safeTitle}</h1>
		<article class="markdown-body">${htmlContent}</article>
	</div>
</body>
</html>`;
};

const openPrintFrame = (): HTMLIFrameElement => {
	const iframe = document.createElement('iframe');
	iframe.setAttribute('aria-hidden', 'true');
	iframe.style.position = 'fixed';
	iframe.style.right = '0';
	iframe.style.bottom = '0';
	iframe.style.width = '0';
	iframe.style.height = '0';
	iframe.style.border = '0';
	document.body.appendChild(iframe);
	return iframe;
};

export const exportChatMarkdownAsPdf = async ({
	title,
	markdown
}: {
	title: string;
	markdown: string;
}): Promise<void> => {
	const renderedHtml = markedParser.parse(markdown ?? '') as string;
	const sanitizedHtml = DOMPurify.sanitize(renderedHtml);

	const iframe = openPrintFrame();
	const doc = iframe.contentDocument;
	const printWindow = iframe.contentWindow;

	if (!doc || !printWindow) {
		iframe.remove();
		return;
	}

	doc.open();
	doc.write(buildPrintDocument(title, sanitizedHtml));
	doc.close();

	doc.querySelectorAll('details').forEach((node) => {
		(node as HTMLDetailsElement).open = true;
	});

	await waitForImages(doc);
	await new Promise((resolve) => setTimeout(resolve, 80));

	let cleaned = false;
	const cleanup = () => {
		if (cleaned) {
			return;
		}

		cleaned = true;
		printWindow.removeEventListener('afterprint', cleanup);
		if (document.body.contains(iframe)) {
			document.body.removeChild(iframe);
		}
	};

	printWindow.addEventListener('afterprint', cleanup, { once: true });
	printWindow.focus();
	printWindow.print();
	setTimeout(cleanup, 2500);
};
