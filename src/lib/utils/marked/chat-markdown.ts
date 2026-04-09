import { Marked, type TokensList } from 'marked';

import markedExtension from '$lib/utils/marked/extension';
import markedKatexExtension from '$lib/utils/marked/katex-extension';
import { disableSingleTilde } from '$lib/utils/marked/strikethrough-extension';
import { mentionExtension } from '$lib/utils/marked/mention-extension';
import footnoteExtension from '$lib/utils/marked/footnote-extension';
import citationExtension from '$lib/utils/marked/citation-extension';

const extensionOptions = {
	throwOnError: false,
	breaks: true
};

const chatMarked = new Marked({
	breaks: true
});

chatMarked.use(markedKatexExtension(extensionOptions));
chatMarked.use(markedExtension(extensionOptions));
chatMarked.use(citationExtension(extensionOptions));
chatMarked.use(footnoteExtension(extensionOptions));
chatMarked.use(disableSingleTilde);
chatMarked.use({
	extensions: [mentionExtension({ triggerChar: '@' }), mentionExtension({ triggerChar: '#' })]
});

export const lexChatMarkdown = (content: string): TokensList => {
	return chatMarked.lexer(content);
};

