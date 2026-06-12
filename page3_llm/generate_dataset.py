"""
Генератор синтетического датасета с кавказским говором.
Запуск: python generate_dataset.py
Выход: caucasian_speech.jsonl (~1000+ примеров)
"""

import json
import random
import itertools
from pathlib import Path

random.seed(42)

# ── Словари ────────────────────────────────────────────────────────────────

NAMES = ["Гиви", "Важа", "Резо", "Сосо", "Зураб", "Нодар", "Давид", "Ираклий",
         "Беслан", "Руслан", "Аслан", "Тимур", "Мурат", "Хасан", "Ахмед"]

ADDRESSES = ["дорогой", "родной", "брат", "слушай", "дружище", "уважаемый",
             "дарагой", "сынок", "э брат", "слюшай"]

OATHS = ["мамой клянусь", "честью клянусь", "жизнью клянусь",
         "клянусь родная", "вот те крест", "слово даю", "клянусь отцом"]

EXCLAMATIONS = ["Вах!", "Вай вай!", "Э!", "Ай!", "Вах вах вах!", "О-о!", "Эй!"]

AFFIRMATIONS = ["да?", "понимаешь?", "слышишь?", "а?", "да-да!", "точно говорю",
                "правда говорю", "не газую"]

ADJS_GOOD = ["хороший", "умный", "красивый", "щедрый", "уважаемый", "добрый",
              "настоящий", "правильный", "честный", "золотой"]

ADJS_BAD = ["нехороший", "обидный", "неправильный", "плохой", "странный"]

TOPICS_FOOD = [
    "такой шашлык делал — пальчики оближешь, клянусь мамой!",
    "хинкали лепил три часа, понимаешь? Три часа! Вот это уважение!",
    "чай пил, мёд кушал — жизнь хорошая, да?",
    "плов готовил — весь двор пришёл, слово даю!",
    "лаваш горячий брал — руки обжёг, но не пожалел, клянусь!",
    "такой суп варил — соседи через забор смотрели, вах!",
    "баранина была — мягкая, нежная, клянусь честью не обманываю!",
    "помидоры с базара брал — вот такие большие, понимаешь!",
]

TOPICS_FRIEND = [
    "ты мне друг? Друг! Значит я тебя уважаю, да?",
    "друг без уважения — это не друг, это просто человек рядом стоит.",
    "настоящий друг — это когда плохо тебе, а он всё равно рядом, слышишь?",
    "двадцать лет дружим, ни разу не обманул, клянусь мамой!",
    "такой друг у меня есть — золото, а не человек, вах!",
]

TOPICS_MONEY = [
    "деньги — это бумага, дорогой. Уважение — вот настоящее богатство!",
    "зачем много денег, если нет уважения? Объясни мне, да?",
    "дал деньги в долг — он пропал. Вот это обидно, клянусь!",
    "на рынке торговался — продавец хитрый попался, но я хитрее, вах!",
    "триста рублей просил — дал пятьсот. Зачем? Уважение, понимаешь!",
]

TOPICS_COMPLAINT = [
    "ты зачем так говоришь? Обидел меня, слышишь? Нехорошо это!",
    "слушай, зачем моросишь? Скажи прямо, как мужчина!",
    "э, дорогой, зачем обижаешь? Я тебе плохое сделал?",
    "не газуй на меня, я тебе ничего плохого не сделал, клянусь!",
    "такой разговор мне не нравится, понимаешь? Неуважение это!",
    "зачем кричишь? Тихо скажи — я не глухой, слово даю!",
]

TOPICS_RESPECT = [
    "старший сказал — значит правильно. Уважение к старшим — это закон!",
    "мой отец говорил: уважай людей — люди тебя уважать будут, да?",
    "к гостю уважение — это самое главное, слышишь? Самое главное!",
    "в наш дом гость пришёл — всё на стол, всё лучшее, клянусь мамой!",
    "гость — это от Бога, понимаешь? Обидеть гостя нельзя!",
]

TOPICS_WORK = [
    "работал как вол — три смены, без выходных, вах! Вот это труд!",
    "начальник хороший попался — уважает людей, слово даю!",
    "строил дом пять лет, понимаешь? Пять лет! Зато свой, родной!",
    "дело делаешь — делай хорошо, или не берись вообще, да?",
    "руками работать не стыдно, стыдно плохо работать, клянусь честью!",
]

TOPICS_FAMILY = [
    "мама сказала — значит всё, разговор закончен, понимаешь?",
    "брат позвонил из Тбилиси — соскучился, клянусь, очень соскучился!",
    "дети растут — радость это, вах! Настоящая радость!",
    "семья — это всё, дорогой. Без семьи человек — как дерево без корней.",
    "отец восемьдесят лет — ещё сам ходит, сам всё делает, слово даю!",
]

ALL_TOPICS = (TOPICS_FOOD + TOPICS_FRIEND + TOPICS_MONEY +
              TOPICS_COMPLAINT + TOPICS_RESPECT + TOPICS_WORK + TOPICS_FAMILY)

# ── Шаблоны предложений ────────────────────────────────────────────────────

def sentence_greeting():
    exc = random.choice(EXCLAMATIONS)
    addr = random.choice(ADDRESSES)
    name = random.choice(NAMES) if random.random() > 0.4 else ""
    parts = [exc, addr]
    if name:
        parts.append(name + "!")
    return " ".join(parts)


def sentence_oath(content):
    oath = random.choice(OATHS)
    variants = [
        f"{oath}, {content}",
        f"{content}, {oath}!",
        f"Вот {oath} — {content}!",
    ]
    return random.choice(variants)


def sentence_question(subj="ты"):
    questions = [
        f"зачем ты так делаешь, {random.choice(ADDRESSES)}?",
        f"ты почему молчишь, {random.choice(ADDRESSES)}?",
        f"ты меня уважаешь, {random.choice(ADDRESSES)}?",
        f"что случилось, скажи мне, {random.choice(ADDRESSES)}?",
        f"зачем моросишь, {random.choice(ADDRESSES)}?",
        f"ты меня обижаешь, понимаешь?",
        f"ты газуешь на меня или как, {random.choice(ADDRESSES)}?",
    ]
    return random.choice(questions)


def sentence_praise(target="ты"):
    adj = random.choice(ADJS_GOOD)
    variants = [
        f"ты такой {adj} человек, {random.choice(AFFIRMATIONS)}",
        f"вот это {adj} человек, клянусь!",
        f"таких {adj} людей мало, слово даю!",
        f"ты {adj}, это все знают, {random.choice(AFFIRMATIONS)}",
    ]
    return random.choice(variants)


def sentence_topic():
    return random.choice(ALL_TOPICS)


def sentence_closing():
    variants = [
        f"Всё, дорогой, разговор окончен. Уважение!",
        f"Иди, кушай, отдыхай. Всё будет хорошо, {random.choice(OATHS)}!",
        f"Вот так живём, понимаешь? Вот так!",
        f"Это жизнь, дорогой. Жизнь!",
        f"Всё, брат. Дружба — это навсегда, слово даю!",
        f"Обнимаю тебя, родной. Вах, хороший ты человек!",
        f"Пока, дорогой! Заходи, кушать будем!",
    ]
    return random.choice(variants)


# ── Генераторы текстов ─────────────────────────────────────────────────────

def make_short():
    """Короткая реплика (1-2 предложения)"""
    builders = [
        lambda: f"{random.choice(EXCLAMATIONS)} {sentence_topic()}",
        lambda: f"{random.choice(ADDRESSES).capitalize()}, {sentence_topic()}",
        lambda: sentence_oath(sentence_topic()),
        lambda: sentence_question(),
        lambda: sentence_praise(),
        lambda: f"{random.choice(EXCLAMATIONS)} {sentence_praise()} {random.choice(AFFIRMATIONS)}",
    ]
    return random.choice(builders)()


def make_medium():
    """Средний монолог (3-5 предложений)"""
    parts = [
        sentence_greeting(),
        sentence_topic(),
        random.choice([sentence_praise(), sentence_question(), sentence_oath(sentence_topic())]),
        random.choice([sentence_topic(), sentence_closing()]),
    ]
    if random.random() > 0.5:
        parts.append(sentence_closing())
    return " ".join(parts)


def make_long():
    """Длинный монолог (6-9 предложений)"""
    parts = [sentence_greeting()]
    # 2-3 тематических предложения
    for _ in range(random.randint(2, 3)):
        parts.append(sentence_topic())
    parts.append(sentence_oath(sentence_topic()))
    parts.append(sentence_question())
    parts.append(sentence_praise())
    for _ in range(random.randint(1, 2)):
        parts.append(sentence_topic())
    parts.append(sentence_closing())
    return " ".join(parts)


def make_dialogue():
    """Короткий диалог"""
    name_a = random.choice(NAMES)
    name_b = random.choice([n for n in NAMES if n != name_a])
    lines = [
        f"— {name_a}, {sentence_question()}",
        f"— {random.choice(EXCLAMATIONS)} {sentence_topic()} {random.choice(OATHS)}!",
        f"— {sentence_praise()} {random.choice(AFFIRMATIONS)}",
        f"— {sentence_closing()}",
    ]
    if random.random() > 0.5:
        lines.append(f"— Вах, {random.choice(ADDRESSES)}, {sentence_topic()}")
    return "\n".join(lines)


def make_story():
    """Маленькая история"""
    opener = random.choice([
        "Слушай, расскажу тебе одну историю.",
        "Вах, было дело, слушай.",
        "Клянусь мамой, вот что случилось.",
        "Э, дорогой, ты не поверишь.",
    ])
    body = " ".join([sentence_topic() for _ in range(random.randint(2, 4))])
    moral = random.choice([
        "Вот такая жизнь, понимаешь?",
        "Урок на всю жизнь, слово даю!",
        "Вот это я запомнил навсегда, клянусь!",
        "Мудрость это, дорогой. Настоящая мудрость!",
    ])
    return f"{opener} {body} {moral}"


# ── Сборка датасета ────────────────────────────────────────────────────────

def generate_dataset(n=1200):
    samples = []
    makers = [
        (make_short,    0.25),  # 25% коротких
        (make_medium,   0.35),  # 35% средних
        (make_long,     0.20),  # 20% длинных
        (make_dialogue, 0.10),  # 10% диалогов
        (make_story,    0.10),  # 10% историй
    ]
    weights = [w for _, w in makers]
    funcs   = [f for f, _ in makers]

    while len(samples) < n:
        fn = random.choices(funcs, weights=weights, k=1)[0]
        text = fn().strip()
        if len(text) > 20:  # отбрасываем слишком короткие
            samples.append({"text": text})

    # Перемешиваем
    random.shuffle(samples)
    return samples


if __name__ == "__main__":
    out_path = Path(__file__).parent / "caucasian_speech.jsonl"
    data = generate_dataset(n=1200)

    with open(out_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Создано: {len(data)} примеров -> {out_path}")
    print("\nПримеры:")
    for item in random.sample(data, 5):
        print(f"  • {item['text'][:120]}")
