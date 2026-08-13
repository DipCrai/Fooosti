import gc
import os
import re

from modules import config

_HF_REPO = 'huihui-ai/Qwen2.5-1.5B-Instruct-abliterated'


def _hf_endpoint():
    return os.environ.get('HF_MIRROR', 'https://huggingface.co').rstrip('/')

_cache = {}
_CACHE_MAX = 512

_RE_NON_LATIN = re.compile(r'[\u0400-\u04FF\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF\u0600-\u06FF]')

_SYSTEM_PROMPT = (
    'You are an expert at crafting high-quality Stable Diffusion prompts. '
    'The user gives a description in any language. '
    'Translate it to English and rewrite it into a detailed, coherent, high-quality Stable Diffusion prompt. '
    'Include subject, appearance, pose, camera angle, lighting, environment, mood, style, and quality tags. '
    'Keep it as a single paragraph. Do not add explanations. Output only the rewritten prompt.'
)

_MAX_NEW_TOKENS = 180


def _looks_english(text: str) -> bool:
    if not text.strip():
        return True
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    other = sum(1 for ch in text if not ch.isascii() and ch.isalpha())
    return latin >= other and not _RE_NON_LATIN.search(text)


def _use_cuda() -> bool:
    if config.prompt_translator_device == 'cpu':
        return False
    if config.prompt_translator_device == 'cuda':
        return True
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        return torch.cuda.mem_get_info(0)[0] >= 5 * 1024 ** 3
    except Exception:
        return False


def is_available() -> bool:
    return os.path.isfile(os.path.join(config.path_prompt_translator, 'model.safetensors'))


def download():
    from huggingface_hub import snapshot_download

    local_dir = config.path_prompt_translator
    os.makedirs(local_dir, exist_ok=True)
    if is_available():
        return
    print(f'[Prompt Translator] downloading {_HF_REPO} to {local_dir} ...')
    snapshot_download(repo_id=_HF_REPO, local_dir=local_dir, endpoint=_hf_endpoint())
    print('[Prompt Translator] download complete.')


def _run(user_text: str) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    local_dir = config.path_prompt_translator
    if not is_available():
        return user_text

    use_cuda = _use_cuda()
    device = 'cuda' if use_cuda else 'cpu'
    print(f'[Prompt Translator] loading on {device} ...')
    tokenizer = AutoTokenizer.from_pretrained(local_dir, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(local_dir, torch_dtype='float32').to(device)
    model.eval()
    try:
        messages = [
            {'role': 'system', 'content': _SYSTEM_PROMPT},
            {'role': 'user', 'content': user_text},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors='pt').to(device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=90,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.3,
            )
        generated = outputs[0][inputs['input_ids'].shape[1]:]
        result = tokenizer.decode(generated, skip_special_tokens=True).strip()
        return _strip_non_latin_tail(result)
    finally:
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()


def _strip_non_latin_tail(result: str) -> str:
    m = _RE_NON_LATIN.search(result)
    if m:
        result = result[:m.start()].rstrip(' ,.;:')
    return result


def translate_and_enhance(text: str) -> str:
    text = (text or '').strip()
    if not text or not config.enable_prompt_translator:
        return text
    if _looks_english(text) and not config.prompt_translator_enhance_english:
        return text
    key = text
    cached = _cache.get(key)
    if cached is not None:
        return cached
    try:
        result = _run(text)
        if not result:
            result = text
        if len(_cache) >= _CACHE_MAX:
            _cache.clear()
        _cache[key] = result
        return result
    except Exception as e:
        print(f'[Prompt Translator] failed, keeping original prompt: {e}')
        return text
