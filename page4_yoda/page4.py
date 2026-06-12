import os
import json
import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODELS_DIR   = os.path.join(os.path.dirname(__file__), '..', 'models')
ADAPTER_PATH = os.path.join(MODELS_DIR, 'modelspage4', 'yoda_adapter')
META_PATH    = os.path.join(MODELS_DIR, 'modelspage4', 'yoda_meta.json')
# Локальная модель (скачана через `hf download`), иначе fallback на HuggingFace Hub
_LOCAL_MODEL = os.path.join(MODELS_DIR, 'Qwen2.5-7B')
MODEL_NAME   = _LOCAL_MODEL if os.path.isdir(_LOCAL_MODEL) else 'Qwen/Qwen2.5-7B'

# Системный промпт — тот же, на котором обучался адаптер
SYSTEM_PROMPT = (
    'Ты — Йода, мудрый мастер джедаев из «Звёздных войн». '
    'Говори с характерной инвертированной синтаксической структурой Йоды: '
    'ставь дополнение или предикатив перед подлежащим и глаголом. '
    'Используй короткие, мудрые, созерцательные фразы на русском языке. '
    'При необходимости упоминай Силу, терпение и равновесие.'
)

STARTER_QUESTIONS = [
    'Как мне стать сильнее?',
    'Что такое Сила?',
    'Как обрести душевный покой?',
    'Мне страшно.',
    'Каков путь к счастью?',
    'Как победить свой гнев?',
]


@st.cache_resource(show_spinner='Загружаем Qwen2.5-7B + LoRA адаптер Йоды...')
def load_model():
    from transformers import BitsAndBytesConfig

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
    mdl.eval()
    return mdl, tok, device


def _is_foreign_letter(ch: str) -> bool:
    """True, если символ — буква НЕ из латиницы и НЕ из кириллицы (китайский, тайский, арабский и т.п.)."""
    if not ch.isalpha():
        return False
    o = ord(ch)
    if 0x41 <= o <= 0x5A or 0x61 <= o <= 0x7A or 0xC0 <= o <= 0x24F:
        return False  # латиница (基)
    if 0x0400 <= o <= 0x052F:
        return False  # кириллица
    return True       # любая другая письменность — блокируем


def foreign_suppress_ids(tok):
    """Токены с иностранными буквами — Qwen изредка их генерирует. Кэшируем на токенайзере."""
    if not hasattr(tok, '_foreign_ids'):
        ids = [
            i for t, i in tok.get_vocab().items()
            if any(_is_foreign_letter(c) for c in tok.convert_tokens_to_string([t]))
        ]
        tok._foreign_ids = ids
    return tok._foreign_ids


def generate_chat(mdl, tok, device, question, max_new_tokens, temperature):
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user',   'content': question},
    ]
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tok(text, return_tensors='pt').to(device)

    # Останавливаем генерацию на границе реплики: <|im_end|>, <|im_start|> + обычный eos
    eos_ids = [tok.eos_token_id]
    for marker in ('<|im_end|>', '<|im_start|>'):
        tid = tok.convert_tokens_to_ids(marker)
        if isinstance(tid, int) and tid >= 0 and tid not in eos_ids:
            eos_ids.append(tid)

    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            max_new_tokens   = max_new_tokens,
            do_sample        = True,
            temperature      = temperature,
            top_p            = 0.9,
            repetition_penalty = 1.3,
            no_repeat_ngram_size = 3,   # запрет повторять 3-граммы — убирает зацикливания
            suppress_tokens  = foreign_suppress_ids(tok),  # блокируем иностранные буквы Qwen
            eos_token_id     = eos_ids,                     # стоп после реплики ассистента
            pad_token_id     = tok.eos_token_id,
        )
    new_ids = out[0][inputs['input_ids'].shape[1]:]
    # Декодируем со спец-токенами и отрезаем всё после первой границы реплики —
    # на случай если модель «продолжила диалог» (base — pretrain-модель)
    raw = tok.decode(new_ids, skip_special_tokens=False)
    for marker in ('<|im_end|>', '<|im_start|>', '<|endoftext|>'):
        idx = raw.find(marker)
        if idx != -1:
            raw = raw[:idx]
    # Подчищаем возможные хвосты ролей и спец-токенов
    answer = raw.replace('<|im_end|>', '').replace('<|im_start|>', '')
    for role in ('assistant', 'user', 'system'):
        if answer.rstrip().endswith(role):
            answer = answer.rstrip()[:-len(role)]
    return answer.strip()


# ── UI ─────────────────────────────────────────────────────────────────────

st.title("Страница 4: Йода-стиль — LLM & LoRA")
st.markdown(
    "Модель **Qwen2.5-7B** дообучена методом **QLoRA** (4-bit + LoRA адаптер) на чат-датасете "
    "в стиле магистра **Йоды** (инвертированный синтаксис, мудрые фразы). "
    "Задай вопрос — сравни ответ базовой модели и адаптера."
)

if not os.path.isdir(ADAPTER_PATH):
    st.warning(
        "Адаптер не найден. Положи веса в `models/modelspage4/yoda_adapter/`."
    )
    st.stop()

# Метаданные обучения
if os.path.exists(META_PATH):
    with open(META_PATH, encoding='utf-8') as f:
        meta = json.load(f)
    with st.expander("Детали обучения"):
        st.json(meta)

# Настройки генерации
if 'yoda_q' not in st.session_state:
    st.session_state.yoda_q = 'Как мне стать сильнее?'

col_l, col_r = st.columns([2, 1])
with col_l:
    st.caption("Быстрые вопросы:")
    btn_cols = st.columns(3)
    for i, q in enumerate(STARTER_QUESTIONS):
        if btn_cols[i % 3].button(q, key=f'yq_{i}', use_container_width=True):
            st.session_state.yoda_q = q   # обновляем до отрисовки text_input

    question = st.text_input(
        "Твой вопрос магистру Йоде",
        key='yoda_q',
        placeholder='Спроси о чём-нибудь...',
    )

with col_r:
    max_new_tokens = st.slider("Макс. токенов", 30, 300, 150)
    temperature    = st.slider("Temperature", 0.5, 1.5, 0.85, 0.05,
                               help="Выше = разнообразнее, ниже = предсказуемее")

if st.button("Спросить Йоду", type="primary", use_container_width=True):
    mdl, tok, device = load_model()

    left, right = st.columns(2)

    with left:
        st.subheader("Базовая модель")
        with st.spinner("Генерация..."):
            mdl.disable_adapter_layers()
            base_out = generate_chat(mdl, tok, device, question, max_new_tokens, temperature)
        st.markdown(base_out)

    with right:
        st.subheader("LoRA — Йода")
        with st.spinner("Генерация..."):
            mdl.enable_adapter_layers()
            yoda_out = generate_chat(mdl, tok, device, question, max_new_tokens, temperature)
        st.markdown(yoda_out)

    mdl.enable_adapter_layers()  # восстанавливаем состояние

    st.info(
        "**Базовая модель** отвечает нейтрально (а как pretrain-модель — часто и не по делу).  \n"
        "**LoRA** выучила речь Йоды: инвертированный порядок слов, краткость, темы Силы и терпения.",
        icon="💡"
    )
