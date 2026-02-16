import { deleteFileById, uploadFile } from '$lib/apis/files';
import { addFileToKnowledgeById } from '$lib/apis/knowledge';
import {
	buildChatMarkdownFileName,
	serializeChatToMarkdown,
	type SerializableChat
} from '$lib/utils/chat-export';

type AddChatToCollectionParams = {
	token: string;
	chat: SerializableChat;
	knowledgeId: string;
	fileName?: string;
	includeThinkingContent?: boolean;
	includeToolCallingContent?: boolean;
};

export const addChatToCollection = async ({
	token,
	chat,
	knowledgeId,
	fileName,
	includeThinkingContent = true,
	includeToolCallingContent = true
}: AddChatToCollectionParams): Promise<{ fileId: string; knowledgeId: string }> => {
	const markdown = serializeChatToMarkdown(chat, {
		includeThinkingContent,
		includeToolCallingContent,
		excludeCitations: true
	});
	if (!markdown.trim()) {
		throw new Error('Chat is empty or unavailable.');
	}

	const resolvedFileName = fileName ?? buildChatMarkdownFileName(chat);
	const file = new File([markdown], resolvedFileName, { type: 'text/markdown' });

	const uploadedFile = await uploadFile(token, file, {
		knowledge_id: knowledgeId,
		conversation_ingest_mode: 'standard'
	});

	if (!uploadedFile?.id) {
		throw new Error('Failed to upload file.');
	}

	try {
		await addFileToKnowledgeById(token, knowledgeId, uploadedFile.id);
		return { fileId: uploadedFile.id, knowledgeId };
	} catch (error) {
		try {
			await deleteFileById(token, uploadedFile.id);
		} catch (cleanupError) {
			console.warn('Failed to cleanup uploaded file after add-to-collection failure', cleanupError);
		}
		throw error;
	}
};
