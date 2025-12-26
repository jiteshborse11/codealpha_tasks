# Author ===>> Jitesh Borse


# python
# py --version   (Try this if python is not recognized)
# import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')
# nltk.download('wordnet')
# python faq_chatbot_nltk.py

import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def ensure_nltk_data():
    """
    Download required NLTK resources if not already present.
    This prevents common 'Resource not found' errors.
    """
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ]

    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"[INFO] Downloading NLTK resource: {name} ...")
            nltk.download(name)


ensure_nltk_data()

faqs = [
    {
        "question": "What is your return policy?",
        "answer": "We offer a 30-day return policy on all products, provided they are in original condition."
    },
    {
        "question": "How can I track my order?",
        "answer": "You can track your order using the tracking link sent to your email after dispatch."
    },
    {
        "question": "What payment methods do you accept?",
        "answer": "We accept credit cards, debit cards, UPI, and net banking."
    },
    {
        "question": "Do you offer international shipping?",
        "answer": "Yes, we ship to selected countries. Shipping charges may vary based on location."
    },
    {
        "question": "How can I contact customer support?",
        "answer": "You can reach our support team via email at support@example.com or call +91-9876543210."
    }
]

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
punct_table = str.maketrans("", "", string.punctuation)


def preprocess(text: str) -> str:
    """
    Preprocess text using NLTK:
    - lowercase
    - remove punctuation
    - tokenize
    - remove stopwords
    - lemmatize
    Returns a single string of cleaned tokens.
    """
    text = text.lower()
    text = text.translate(punct_table)

    tokens = nltk.word_tokenize(text)

    cleaned_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words and token.isalpha()
    ]

    return " ".join(cleaned_tokens)


questions = [faq["question"] for faq in faqs]
processed_questions = [preprocess(q) for q in questions]

vectorizer = TfidfVectorizer()
faq_matrix = vectorizer.fit_transform(processed_questions)


def get_best_answer(user_question: str, threshold: float = 0.2) -> str:
    """
    Match user question with FAQ using cosine similarity.
    """
    processed = preprocess(user_question)
    if not processed:
        return "Sorry, I couldn't understand your question. Could you rephrase it?"

    user_vec = vectorizer.transform([processed])
    similarities = cosine_similarity(user_vec, faq_matrix)[0]

    best_idx = similarities.argmax()
    best_score = similarities[best_idx]

    if best_score < threshold:
        return "Sorry, I couldn't find a relevant answer. Please contact support."

    return faqs[best_idx]["answer"]


def chat():
    print("=============================================")
    print("         FAQ Chatbot (Python + NLTK)")
    print("=============================================")
    print("Ask me anything about orders, returns, payments, shipping, etc.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Bot: Thank you! Have a great day.")
            break

        answer = get_best_answer(user_input)
        print("Bot:", answer)
        print()


if __name__ == "__main__":
    chat()
