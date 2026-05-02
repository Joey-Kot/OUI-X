import { v4 as uuidv4 } from 'uuid';
import { toast } from 'svelte-sonner';

import { uploadFile } from '$lib/apis/files';
import { convertHeicToJpeg, extractContentFromFile, getImageCompressionMetadata } from '$lib/utils';

type ChatFileUploadOptions = {
	files: any[];
	setFiles: (files: any[]) => void;
	inputFiles: File[];
	selectedModels: string[];
	models: any[];
	config: any;
	settings: any;
	user: any;
	temporaryChatEnabled: boolean;
	i18n: any;
	process?: boolean;
	itemData?: Record<string, any>;
};

const getModelCapabilities = (selectedModels: string[], models: any[]) => {
	const modelIds = selectedModels ?? [];

	return {
		visionCapableModels: modelIds.filter(
			(model) => models?.find((m) => m.id === model)?.info?.meta?.capabilities?.vision ?? true
		),
		fileUploadCapableModels: modelIds.filter(
			(model) => models?.find((m) => m.id === model)?.info?.meta?.capabilities?.file_upload ?? true
		)
	};
};

export const processChatInputFiles = async ({
	files,
	setFiles,
	inputFiles,
	selectedModels,
	models,
	config,
	settings,
	user,
	temporaryChatEnabled,
	i18n,
	process = true,
	itemData = {}
}: ChatFileUploadOptions) => {
	let currentFiles = files ?? [];
	const modelIds = selectedModels ?? [];
	const { visionCapableModels, fileUploadCapableModels } = getModelCapabilities(modelIds, models);

	const updateFiles = (nextFiles: any[]) => {
		currentFiles = nextFiles;
		setFiles(nextFiles);
	};

	const uploadFileHandler = async (file: File, shouldProcess = true, fileItemData = {}) => {
		if (user?.role !== 'admin' && !(user?.permissions?.chat?.file_upload ?? true)) {
			toast.error(i18n.t('You do not have permission to upload files.'));
			return null;
		}

		if (fileUploadCapableModels.length !== modelIds.length) {
			toast.error(i18n.t('Model(s) do not support file upload'));
			return null;
		}

		const tempItemId = uuidv4();
		const fileItem = {
			type: 'file',
			file: '',
			id: null,
			url: '',
			name: file.name,
			collection_name: '',
			status: 'uploading',
			size: file.size,
			error: '',
			itemId: tempItemId,
			...fileItemData
		};

		if (fileItem.size == 0) {
			toast.error(i18n.t('You cannot upload an empty file.'));
			return null;
		}

		updateFiles([...currentFiles, fileItem]);

		if (!temporaryChatEnabled) {
			try {
				let metadata = null;
				if (
					(file.type.startsWith('audio/') || file.type.startsWith('video/')) &&
					settings?.audio?.stt?.language
				) {
					metadata = {
						language: settings?.audio?.stt?.language
					};
				}

				if (file.type.startsWith('image/')) {
					metadata = {
						...(metadata ?? {}),
						...(getImageCompressionMetadata(settings) ?? {})
					};
				}

				const conversationEmbeddingEnabled =
					settings?.conversationFileUploadEmbedding === true ||
					(config?.file?.conversation_file_upload_embedding ?? false);
				metadata = {
					...(metadata ?? {}),
					conversation_ingest_mode: conversationEmbeddingEnabled ? 'standard' : 'direct_context'
				};

				const uploadedFile = await uploadFile(localStorage.token, file, metadata, shouldProcess);

				if (uploadedFile) {
					console.log('File upload completed:', {
						id: uploadedFile.id,
						name: fileItem.name,
						collection: uploadedFile?.meta?.collection_name
					});

					if (uploadedFile.error) {
						console.warn('File upload warning:', uploadedFile.error);
						toast.warning(uploadedFile.error);
					}

					fileItem.status = 'uploaded';
					fileItem.file = uploadedFile;
					fileItem.id = uploadedFile.id;
					fileItem.collection_name =
						uploadedFile?.meta?.collection_name || uploadedFile?.collection_name;
					fileItem.active_collection_name = uploadedFile?.meta?.active_collection_name;
					fileItem.conversation_upload_knowledge_id =
						uploadedFile?.meta?.conversation_upload_knowledge_id;
					fileItem.content_type = uploadedFile.meta?.content_type || uploadedFile.content_type;
					fileItem.url = `${uploadedFile.id}`;

					updateFiles(currentFiles);
				} else {
					updateFiles(currentFiles.filter((item) => item?.itemId !== tempItemId));
				}
			} catch (e) {
				toast.error(`${e}`);
				updateFiles(currentFiles.filter((item) => item?.itemId !== tempItemId));
			}
		} else {
			const content = await extractContentFromFile(file).catch((error) => {
				toast.error(i18n.t('Failed to extract content from the file: {{error}}', { error: error }));
				return null;
			});

			if (content === null) {
				toast.error(i18n.t('Failed to extract content from the file.'));
				updateFiles(currentFiles.filter((item) => item?.itemId !== tempItemId));
				return null;
			} else {
				console.log('Extracted content from file:', {
					name: file.name,
					size: file.size,
					content: content
				});

				fileItem.status = 'uploaded';
				fileItem.type = 'text';
				fileItem.content = content;
				fileItem.id = uuidv4();

				updateFiles(currentFiles);
			}
		}

		return fileItem;
	};

	console.log('Input files handler called with:', inputFiles);

	if (
		(config?.file?.max_count ?? null) !== null &&
		currentFiles.length + inputFiles.length > config?.file?.max_count
	) {
		toast.error(
			i18n.t(`You can only chat with a maximum of {{maxCount}} file(s) at a time.`, {
				maxCount: config?.file?.max_count
			})
		);
		return;
	}

	await Promise.all(
		inputFiles.map(async (file) => {
			console.log('Processing file:', {
				name: file.name,
				type: file.type,
				size: file.size,
				extension: file.name.split('.').at(-1)
			});

			if (
				(config?.file?.max_size ?? null) !== null &&
				file.size > (config?.file?.max_size ?? 0) * 1024 * 1024
			) {
				console.log('File exceeds max size limit:', {
					fileSize: file.size,
					maxSize: (config?.file?.max_size ?? 0) * 1024 * 1024
				});
				toast.error(
					i18n.t(`File size should not exceed {{maxSize}} MB.`, {
						maxSize: config?.file?.max_size
					})
				);
				return;
			}

			if (file.type.startsWith('image/')) {
				if (visionCapableModels.length === 0) {
					toast.error(i18n.t('Selected model(s) do not support image inputs'));
					return;
				}

				if (temporaryChatEnabled) {
					const imageFile = file.type === 'image/heic' ? await convertHeicToJpeg(file) : file;
					const url = await new Promise<string | ArrayBuffer | null>((resolve) => {
						const reader = new FileReader();
						reader.onload = (event) => resolve(event.target?.result ?? null);
						reader.onerror = () => resolve(null);
						reader.readAsDataURL(imageFile);
					});

					if (url) {
						updateFiles([
							...currentFiles,
							{
								type: 'image',
								url
							}
						]);
					}
				} else {
					await uploadFileHandler(file, false, itemData);
				}
			} else {
				await uploadFileHandler(file, process, itemData);
			}
		})
	);
};
