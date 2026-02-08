<script lang="ts">
  import { getContext } from 'svelte';
  import Modal from '$lib/components/common/Modal.svelte';

  const i18n = getContext('i18n');

  export let show = false;
  export let loading = false;
  export let mode: 'default' | 'custom' = 'default';
  export let globalDefaults: Record<string, any> = {};
  export let overrides: Record<string, any> = {};
  export let onSave: Function = () => {};

  type FieldMode = 'default' | 'custom';
  type FieldDef = {
    key: string;
    label: string;
    type: 'text' | 'number' | 'select';
    hint?: string;
    options?: Array<{ value: string; label: string }>;
    step?: string;
    min?: number;
    max?: number;
  };

  const reindexHint =
    'After updating or changing the embedding model, you must reindex the knowledge base for the changes to take effect. You can do this using the "Reindex" button below.';

  const fields: FieldDef[] = [
    {
      key: 'RAG_EMBEDDING_ENGINE',
      label: 'Embedding Model Engine',
      type: 'select',
      options: [
        { value: 'openai', label: 'OpenAI' },
        { value: 'azure_openai', label: 'Azure OpenAI' }
      ]
    },
    { key: 'RAG_EMBEDDING_MODEL', label: 'Embedding Model', type: 'text', hint: reindexHint },
    {
      key: 'TEXT_SPLITTER',
      label: 'Text Splitter',
      type: 'select',
      options: [
        { value: '', label: 'Default (Character)' },
        { value: 'token', label: 'Token (Tiktoken)' },
        { value: 'token_voyage', label: 'Token (Voyage)' }
      ]
    },
    { key: 'VOYAGE_TOKENIZER_MODEL', label: 'Voyage Tokenizer Model', type: 'text' },
    { key: 'CHUNK_SIZE', label: 'Chunk Size', type: 'number', min: 0, hint: reindexHint },
    { key: 'CHUNK_OVERLAP', label: 'Chunk Overlap', type: 'number', min: 0, hint: reindexHint },
    {
      key: 'RAG_EMBEDDING_BATCH_SIZE',
      label: 'Embedding Batch Size',
      type: 'number',
      min: -2,
      max: 16000,
      step: '1'
    },
    {
      key: 'RAG_RERANKING_ENGINE',
      label: 'Reranking Engine',
      type: 'select',
      options: [
        { value: 'external', label: 'External' },
        { value: 'voyage', label: 'Voyage' }
      ]
    },
    { key: 'RAG_RERANKING_MODEL', label: 'Reranking Model', type: 'text' },
    { key: 'TOP_K', label: 'Top K', type: 'number', min: 0 },
    { key: 'TOP_K_RERANKER', label: 'Top K Reranker', type: 'number', min: 0 },
    {
      key: 'RELEVANCE_THRESHOLD',
      label: 'Relevance Threshold',
      type: 'number',
      min: 0,
      step: '0.01'
    }
  ];

  let localMode: 'default' | 'custom' = 'default';
  let fieldModes: Record<string, FieldMode> = {};
  let localValues: Record<string, any> = {};

  const effectiveValue = (key: string) => {
    if (fieldModes[key] === 'custom') {
      return localValues[key];
    }

    return globalDefaults[key];
  };

  const isVisible = (key: string) => {
    if (key === 'VOYAGE_TOKENIZER_MODEL') {
      return effectiveValue('TEXT_SPLITTER') === 'token_voyage';
    }

    if (key === 'TOP_K_RERANKER' || key === 'RELEVANCE_THRESHOLD') {
      const model = effectiveValue('RAG_RERANKING_MODEL');
      return typeof model === 'string' && model.trim() !== '';
    }

    return true;
  };

  const initFields = () => {
    localMode = mode;

    const nextFieldModes: Record<string, FieldMode> = {};
    const nextLocalValues: Record<string, any> = {};

    for (const field of fields) {
      const hasOverride = overrides[field.key] !== undefined;
      nextFieldModes[field.key] = hasOverride ? 'custom' : 'default';
      nextLocalValues[field.key] = hasOverride ? overrides[field.key] : globalDefaults[field.key];
    }

    fieldModes = nextFieldModes;
    localValues = nextLocalValues;
  };

  $: if (show) {
    initFields();
  }

  const setFieldValue = (key: string, value: any) => {
    localValues = {
      ...localValues,
      [key]: value
    };
  };

  const toggleFieldMode = (key: string) => {
    const current = fieldModes[key] ?? 'default';
    const next: FieldMode = current === 'default' ? 'custom' : 'default';

    fieldModes = {
      ...fieldModes,
      [key]: next
    };

    if (next === 'default') {
      setFieldValue(key, globalDefaults[key]);
    } else if (localValues[key] === undefined) {
      setFieldValue(key, globalDefaults[key]);
    }
  };

  const handleSelectChange = (key: string, value: string) => {
    setFieldValue(key, value);

    if (key === 'RAG_EMBEDDING_ENGINE' && ['openai', 'azure_openai'].includes(value)) {
      if (fieldModes.RAG_EMBEDDING_MODEL === 'custom') {
        setFieldValue('RAG_EMBEDDING_MODEL', 'text-embedding-3-small');
      }
    }

    if (key === 'RAG_RERANKING_ENGINE' && ['external', 'voyage'].includes(value)) {
      if (fieldModes.RAG_RERANKING_MODEL === 'custom') {
        setFieldValue('RAG_RERANKING_MODEL', '');
      }
    }
  };

  const save = async () => {
    const nextOverrides: Record<string, any> = {};

    if (localMode === 'custom') {
      for (const field of fields) {
        if (fieldModes[field.key] !== 'custom') {
          continue;
        }

        const value = localValues[field.key];
        if (value === '' || value === undefined || value === null) {
          continue;
        }

        nextOverrides[field.key] = value;
      }
    }

    await onSave({
      mode: localMode,
      overrides: nextOverrides
    });
  };
</script>

<Modal bind:show size="lg">
  <div class="p-5 text-sm">
    <div class="text-lg font-medium mb-4">{$i18n.t('Config')}</div>

    <div class="flex gap-2 mb-4">
      <button
        type="button"
        class="px-3 py-1.5 rounded-full text-xs {localMode === 'default'
          ? 'bg-gray-900 text-white dark:bg-white dark:text-black'
          : 'bg-gray-100 dark:bg-gray-850'}"
        on:click={() => {
          localMode = 'default';
        }}
      >
        {$i18n.t('Default')}
      </button>
      <button
        type="button"
        class="px-3 py-1.5 rounded-full text-xs {localMode === 'custom'
          ? 'bg-gray-900 text-white dark:bg-white dark:text-black'
          : 'bg-gray-100 dark:bg-gray-850'}"
        on:click={() => {
          localMode = 'custom';
        }}
      >
        {$i18n.t('Custom')}
      </button>
    </div>

    {#if localMode === 'custom'}
      <div class="space-y-3 max-h-[60vh] overflow-y-auto pr-1">
        {#each fields as field}
          {#if isVisible(field.key)}
            <div>
              <div class="flex justify-between items-center mb-1">
                <div class="text-xs font-medium">{$i18n.t(field.label)}</div>
                <button
                  type="button"
                  class="text-xs px-2 py-0.5 rounded {fieldModes[field.key] === 'custom'
                    ? 'bg-gray-900 text-white dark:bg-white dark:text-black'
                    : 'bg-gray-100 text-gray-500 dark:bg-gray-850 dark:text-gray-400'}"
                  on:click={() => toggleFieldMode(field.key)}
                >
                  {$i18n.t(fieldModes[field.key] === 'custom' ? 'Custom' : 'Default')}
                </button>
              </div>

              {#if field.type === 'select'}
                <select
                  class="w-full rounded-lg py-2 px-3 text-sm outline-hidden {fieldModes[field.key] === 'custom'
                    ? 'bg-white dark:bg-gray-900'
                    : 'bg-gray-50 text-gray-500 dark:bg-gray-850 dark:text-gray-400'}"
                  value={localValues[field.key]}
                  disabled={fieldModes[field.key] === 'default'}
                  on:change={(e) => handleSelectChange(field.key, e.currentTarget.value)}
                >
                  {#each field.options || [] as option}
                    <option value={option.value}>{$i18n.t(option.label)}</option>
                  {/each}
                </select>
              {:else}
                <input
                  class="w-full rounded-lg py-2 px-3 text-sm outline-hidden {fieldModes[field.key] === 'custom'
                    ? 'bg-white dark:bg-gray-900'
                    : 'bg-gray-50 text-gray-500 dark:bg-gray-850 dark:text-gray-400'}"
                  type={field.type}
                  value={localValues[field.key]}
                  on:input={(e) => {
                    const raw = e.currentTarget.value;
                    if (field.type === 'number') {
                      setFieldValue(field.key, raw === '' ? '' : Number(raw));
                    } else {
                      setFieldValue(field.key, raw);
                    }
                  }}
                  disabled={fieldModes[field.key] === 'default'}
                  min={field.type === 'number' ? field.min : undefined}
                  max={field.type === 'number' ? field.max : undefined}
                  step={field.step}
                />
              {/if}

              {#if field.hint}
                <div class="mt-1 text-xs text-gray-400 dark:text-gray-500">{$i18n.t(field.hint)}</div>
              {/if}
            </div>
          {/if}
        {/each}
      </div>
    {/if}

    <div class="mt-5 flex justify-end gap-2">
      <button
        type="button"
        class="px-3 py-1.5 rounded-full text-sm bg-gray-100 dark:bg-gray-850"
        on:click={() => {
          show = false;
        }}
      >
        {$i18n.t('Cancel')}
      </button>
      <button
        type="button"
        class="px-3 py-1.5 rounded-full text-sm bg-black text-white dark:bg-white dark:text-black disabled:opacity-50"
        disabled={loading}
        on:click={save}
      >
        {$i18n.t('Save')}
      </button>
    </div>
  </div>
</Modal>
