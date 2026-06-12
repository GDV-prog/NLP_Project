import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import time
from pathlib import Path

def show_page2():
    st.title("📱 Классификация тематики новостей Telegram")
    st.write("Страница 2 • Определение категории новостных постов")

    # Веса лежат в общей папке models/model_page2/ (как у страниц 1 и 3)
    MODELS_DIR = Path(__file__).parent.parent / "models" / "model_page2"
    model_path = str(MODELS_DIR / "best_model")
    encoder_path = str(MODELS_DIR / "label_encoder.pkl")

    # Автоматический выбор устройства: CUDA для сервера, MPS для Mac, CPU как запасной
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
        st.success(f"Модель успешно загружена на устройстве: {device.type.upper()}")
    except Exception as e:
        st.error(
            f"Ошибка загрузки модели: {e}. Проверьте, что папка 'best_model' "
            f"и файл 'label_encoder.pkl' лежат внутри папки page2_news/."
        )
        return  # Прерываем выполнение страницы, если модель не загрузилась

    user_input = st.text_area("Вставьте текст поста из Telegram-канала:", height=150)

    if st.button("Определить тематику"):
        if user_input.strip() == "":
            st.warning("Пожалуйста, введите текст.")
        else:
            start_time = time.time()
            
            # Токенизация и перенос тензоров на нужное устройство
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_class_idx = torch.argmax(probs, dim=-1).item()
            
            # Безопасное извлечение уверенности для батча размера 1
            confidence = probs[0, pred_class_idx].item()
            
            predicted_category = label_encoder.inverse_transform([pred_class_idx])[0]
            inference_time = (time.time() - start_time) * 1000 
            
            st.subheader(f"Результат: **{predicted_category}**")
            st.write(f"Уверенность модели: `{confidence:.2%}`")
            st.write(f"Время инференса: `{inference_time:.2f} мс`")

            # Вероятности по всем классам
            st.caption("Распределение по категориям:")
            all_probs = probs[0].cpu().tolist()
            for cat, p in sorted(
                zip(label_encoder.classes_, all_probs), key=lambda x: -x[1]
            ):
                st.write(f"{cat}")
                st.progress(p)


# Запускаем страницу (main.py исполняет файл сверху вниз)
show_page2()
