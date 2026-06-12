import os
import json
import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODELS_DIR   = os.path.join(os.path.dirname(__file__), '..', 'models')
ADAPTER_PATH = os.path.join(MODELS_DIR, 'lora_adapter')
META_PATH    = os.path.join(MODELS_DIR, 'lora_meta.json')
# Локальная модель (скачана через `hf download`), иначе fallback на HuggingFace Hub
_LOCAL_MODEL = os.path.join(MODELS_DIR, 'Qwen2.5-7B')
MODEL_NAME   = _LOCAL_MODEL if os.path.isdir(_LOCAL_MODEL) else 'Qwen/Qwen2.5-7B'

# Сила LoRA-адаптера: 2.0 (как обучено) душит содержание, 1.2 — баланс стиль/связность
LORA_SCALING = 1.2

STARTER_PROMPTS = [
    'Вах, слушай,',
    'Клянусь мамой,',
    'Э, дорогой,',
    'Вай вай вай!',
    'Слушай, брат,',
    'Клянусь честью,',
]


@st.cache_resource(show_spinner='Загружаем Qwen2.5-7B + LoRA адаптер...')
def load_model():
    from transformers import BitsAndBytesConfig
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    tok.pad_token    = tok.eos_token
    tok.padding_side = 'right'

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit           = True,
        bnb_4bit_quant_type    = 'nf4',
        bnb_4bit_compute_dtype = torch.float16,
        bnb_4bit_use_double_quant = True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config = bnb_cfg,
        device_map          = 'auto',
        torch_dtype         = torch.float16,
    )
    mdl = PeftModel.from_pretrained(base, ADAPTER_PATH)

    # Ослабляем адаптер до LORA_SCALING (обучен с alpha/r = 2.0)
    for module in mdl.modules():
        if hasattr(module, 'scaling') and isinstance(module.scaling, dict):
            for adapter in module.scaling:
                module.scaling[adapter] = LORA_SCALING

    mdl.eval()
    return mdl, tok, device


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (
        0x4E00 <= o <= 0x9FFF   or  # китайские иероглифы (CJK Unified)
        0x3400 <= o <= 0x4DBF   or  # CJK Extension A
        0x3040 <= o <= 0x30FF   or  # хирагана + катакана
        0xAC00 <= o <= 0xD7AF   or  # корейский хангыль
        0xF900 <= o <= 0xFAFF       # CJK Compatibility
    )


def cjk_suppress_ids(tok):
    """Токены, содержащие CJK-символы — Qwen изредка их генерирует. Кэшируем на токенайзере."""
    if not hasattr(tok, '_cjk_ids'):
        ids = [
            i for t, i in tok.get_vocab().items()
            if any(_is_cjk(c) for c in tok.convert_tokens_to_string([t]))
        ]
        tok._cjk_ids = ids
    return tok._cjk_ids


def generate(mdl, tok, device, prompt, max_new_tokens, temperature):
    inputs = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            max_new_tokens   = max_new_tokens,
            do_sample        = True,
            temperature      = temperature,
            top_p            = 0.9,
            repetition_penalty = 1.3,
            no_repeat_ngram_size = 3,   # запрет повторять 3-граммы — убирает зацикливания
            suppress_tokens  = cjk_suppress_ids(tok),  # блокируем китайские токены Qwen
            pad_token_id     = tok.eos_token_id,
        )
    new_ids = out[0][inputs['input_ids'].shape[1]:]
    return tok.decode(new_ids, skip_special_tokens=True)


# ── UI ─────────────────────────────────────────────────────────────────────

st.title("Страница 3: Генерация текста — LLM & LoRA")
st.markdown(
    "Модель **Qwen2.5-7B** дообучена на синтетическом датасете кавказского говора "
    "методом **QLoRA** (4-bit квантизация + LoRA адаптер). Сравни генерацию до и после fine-tuning."
)

if not os.path.isdir(ADAPTER_PATH):
    st.warning("Адаптер не найден. Сначала запусти `page3_llm/work.ipynb` до конца.")
    st.stop()

# Метаданные обучения
if os.path.exists(META_PATH):
    with open(META_PATH, encoding='utf-8') as f:
        meta = json.load(f)
    with st.expander("Детали обучения"):
        st.json(meta)

# Настройки генерации
if 'prompt' not in st.session_state:
    st.session_state.prompt = 'Вах, слушай,'

col_l, col_r = st.columns([2, 1])
with col_l:
    st.caption("Быстрые примеры:")
    btn_cols = st.columns(len(STARTER_PROMPTS))
    for i, sp in enumerate(STARTER_PROMPTS):
        if btn_cols[i].button(sp, key=f'sp_{i}', use_container_width=True):
            st.session_state.prompt = sp   # обновляем до отрисовки text_input

    prompt = st.text_input(
        "Начало текста (промпт)",
        key='prompt',
        placeholder='Введи начало фразы...',
    )

with col_r:
    max_new_tokens = st.slider("Макс. токенов", 30, 250, 100)
    temperature    = st.slider("Temperature", 0.5, 1.5, 0.9, 0.05,
                               help="Выше = разнообразнее, ниже = предсказуемее")

if st.button("Сгенерировать", type="primary", use_container_width=True):
    mdl, tok, device = load_model()

    left, right = st.columns(2)

    with left:
        st.subheader("Базовая модель")
        with st.spinner("Генерация..."):
            mdl.disable_adapter_layers()
            base_out = generate(mdl, tok, device, prompt, max_new_tokens, temperature)
        st.markdown(f"**{prompt}** {base_out}")

    with right:
        st.subheader("LoRA (fine-tuned)")
        with st.spinner("Генерация..."):
            mdl.enable_adapter_layers()
            lora_out = generate(mdl, tok, device, prompt, max_new_tokens, temperature)
        st.markdown(f"**{prompt}** {lora_out}")

    mdl.enable_adapter_layers()  # восстанавливаем состояние

    st.info(
        "**Базовая модель** не знает кавказского говора — генерирует нейтральный русский текст.  \n"
        "**LoRA** выучила стиль: характерные восклицания, обращения, клятвы.",
        icon="💡"
    )
