import streamlit as st

st.set_page_config(page_title="NLP Project", layout="wide")

pg = st.navigation([
    st.Page("page1_reviews/page1.py", title="Страница 1: Классификация отзывов", icon="🏥"),
    st.Page("page2_news/page2.py", title="Страница 2: Классификация новостей", icon="📰"),
    st.Page("page3_llm/page3.py", title="Страница 3: LLM & LoRA", icon="🤖"),
    st.Page("page4_yoda/page4.py", title="Страница 4: Йода-стиль", icon="🧙"),
    st.Page("page5_summary/page5.py", title="Страница 5: Итоги проекта", icon="📋"),
])
pg.run()
