import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import time
from pathlib import Path

def show_page2():
    st.title("📱 Классификация тематики новостей Telegram")
    st.markdown("Определение категории поста: **мода**, **технологии**, **финансы**, **крипта**, **спорт**.")

    # ---------- Пути к моделям ----------
    MODELS_DIR = Path(__file__).parent.parent / "models" / "model_page2"
    model_path = str(MODELS_DIR / "best_model")
    encoder_path = str(MODELS_DIR / "label_encoder.pkl")

    # ---------- Выбор устройства ----------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    @st.cache_resource
    def load_model():
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        label_encoder = joblib.load(encoder_path)
        return tokenizer, model, label_encoder

    try:
        tokenizer, model, label_encoder = load_model()
        st.success(f"✅ Модель загружена на **{device.type.upper()}**")
    except Exception as e:
        st.error(
            f"❌ Ошибка загрузки модели: {e}\n\n"
            "Убедитесь, что папка 'best_model' и файл 'label_encoder.pkl' лежат в `models/model_page2/`."
        )
        return

    # ---------- Примеры текстов (строго для 5 категорий) ----------
    SAMPLE_TEXTS = {
        "👗 Мода": "Balenciaga представил новую коллекцию осень-зима. В тренде – oversized силуэты и яркие аксессуары.",
        "📱 Технологии": "Apple анонсировала складной iPhone с гибким дисплеем и поддержкой ИИ.",
        "💰 Финансы": "Курс рубля укрепился на фоне высоких цен на нефть. Индексы Мосбиржи обновили максимумы.",
        "₿ Крипта": "Биткоин превысил $70 000. Эфириум обновляет протокол для снижения комиссий.",
        "⚽ Спорт": "Российские теннисисты вышли в четвертьфинал Australian Open, обыграв седьмую ракетку мира."
    }
    DEFAULT_EXAMPLES = ["— введите свой текст —"] + list(SAMPLE_TEXTS.values())

    # ---------- Интерфейс ввода ----------
    st.subheader("📝 Введите текст поста")
    sample_choice = st.selectbox(
        "📌 Вставить пример:",
        DEFAULT_EXAMPLES,
        index=0,
        help="Выберите пример, чтобы автоматически заполнить поле ниже."
    )
    default_text = "" if sample_choice.startswith("—") else sample_choice
    user_input = st.text_area("Текст новости:", value=default_text, height=150)

    if st.button("🚀 Определить тематику", type="primary", disabled=not user_input.strip()):
        if not user_input.strip():
            st.warning("Пожалуйста, введите текст.")
        else:
            start_time = time.time()

            # Токенизация
            inputs = tokenizer(
                user_input,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, pred_idx].item()
            predicted_category = label_encoder.inverse_transform([pred_idx])[0]
            inference_time = (time.time() - start_time) * 1000

            # --- Вывод в три колонки ---
            st.subheader("🔍 Результат")
            col1, col2, col3 = st.columns(3)
            col1.metric("Категория", predicted_category)
            col2.metric("Уверенность", f"{confidence:.2%}")
            col3.metric("Время инференса", f"{inference_time:.1f} мс")

            # --- Распределение вероятностей по всем классам ---
            st.caption("📊 Распределение по категориям:")
            all_probs = probs[0].cpu().tolist()
            for cat, p in sorted(zip(label_encoder.classes_, all_probs), key=lambda x: -x[1]):
                st.write(f"**{cat}** — `{p:.2%}`")
                st.progress(p)

    # ---------- Информационный блок ----------
    st.divider()
    st.info(
        "💡 **Совет:** модель лучше всего работает на коротких новостных постах (до 256 токенов). "
        "Используйте примеры выше для быстрого тестирования."
    )

# Если файл запущен напрямую – показываем страницу (для отладки)
if __name__ == "__main__":
    show_page2()