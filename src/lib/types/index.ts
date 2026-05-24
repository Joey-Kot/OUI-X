export type Banner = {
	id: string;
	type: string;
	title?: string;
	content: string;
	url?: string;
	dismissible?: boolean;
	timestamp: number;
};

export enum TTS_RESPONSE_SPLIT {
	PUNCTUATION = 'punctuation',
	PARAGRAPHS = 'paragraphs',
	NONE = 'none'
}

export enum TTS_OUTPUT_FORMAT {
	DEFAULT = 'default',
	WEBM = 'webm',
	MP3 = 'mp3',
	FLAC = 'flac',
	WAV = 'wav'
}
