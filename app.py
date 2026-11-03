import truststore
truststore.inject_into_ssl()

import re
import streamlit as st
from dotenv import load_dotenv
from google.genai.errors import ClientError, ServerError
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.core import get_response_synthesizer
from llama_index.core import PromptTemplate
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI

load_dotenv()

Settings.embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-001")
Settings.llm = GoogleGenAI(model="gemini-3.5-flash-lite", max_retries=1)

CASE_NUMBER_PATTERN = r"\d{4}-\d{3}"

QA_PROMPT = PromptTemplate(
    """אתה עוזר מחקר בארכיון של סוכנות הבילוש "עין הינשוף".
ענה על השאלה אך ורק על סמך קטעי המידע שלמטה. אל תשתמש בידע כללי.

כללים:
1. אם התשובה לא מופיעה בקטעים, כתוב: "לא נמצא מידע על כך במאגר." אל תנחש.
2. אם משהו לא מוזכר בקטעים, אל תקבע שהוא לא קיים. כתוב שהוא לא מופיע במאגר.
3. אל תקבע אשמה, אחריות או מסקנה שלא כתובה במפורש בקטעים. אם יש ראיות או סתירות, תאר אותן כעובדות וציין שאין בחומר מסקנה רשמית.
4. בסוף התשובה ציין רק את שם הקובץ שממנו נלקח המידע, בלי הנתיב (למשל: 07_case_2025-023_bakery_poisoning.md).
5. ענה בעברית, בקצרה ולעניין.

קטעי המידע:
---------------------
{context_str}
---------------------

שאלה: {query_str}
תשובה:"""
)


@st.cache_resource
def load_index():
    storage_context = StorageContext.from_defaults(persist_dir="storage")
    return load_index_from_storage(storage_context)


def find_exact_matches(question):
    case_numbers = re.findall(CASE_NUMBER_PATTERN, question)
    matches = []
    for node in index.docstore.docs.values():
        for case_number in case_numbers:
            if case_number in node.text:
                matches.append(NodeWithScore(node=node, score=1.0))
                break
    return matches


def get_sources(question, top_k):
    sources = find_exact_matches(question)
    found_ids = [source.node.node_id for source in sources]
    retriever = index.as_retriever(similarity_top_k=top_k)
    for source in retriever.retrieve(question):
        if source.node.node_id not in found_ids:
            sources.append(source)
    return sources


def show_sources(sources):
    st.subheader("הקטעים שנשלפו")
    for number, source in enumerate(sources, start=1):
        file_name = source.node.metadata["file_name"]
        title = f"{number}. {file_name}  |  ציון: {source.score:.3f}"
        with st.expander(title):
            st.text(source.node.text)


def show_history(items):
    if len(items) == 0:
        return
    st.divider()
    st.subheader("שאלות קודמות")
    for past in items:
        st.markdown(f"**שאלה:** {past['question']}")
        st.markdown(f"**תשובה:** {past['answer']}")
        st.caption("מקורות: " + ", ".join(past["files"]))
        st.markdown("---")


st.set_page_config(page_title="עין הינשוף", page_icon="🦉", layout="wide")
st.markdown(
    "<style>.stApp { direction: rtl; text-align: right; }</style>",
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []

previous_questions = list(st.session_state.history)

index = load_index()

st.title("🦉 עין הינשוף – מערכת שאלות")

top_k = st.slider("כמה קטעים לשלוף", min_value=1, max_value=10, value=3)
question = st.text_input("שאלה")

column_1, column_2 = st.columns(2)
search_clicked = column_1.button("🔍 חפש קטעים (לא נוגע במכסה של Gemini)")
answer_clicked = column_2.button("💬 ענה (קריאה אחת ל־Gemini)")

if (search_clicked or answer_clicked) and question == "":
    st.warning("כתבי קודם שאלה")

elif search_clicked:
    sources = get_sources(question, top_k)
    show_sources(sources)

elif answer_clicked:
    sources = get_sources(question, top_k)
    synthesizer = get_response_synthesizer(text_qa_template=QA_PROMPT)
    try:
        with st.spinner("Gemini מנסח תשובה..."):
            response = synthesizer.synthesize(question, nodes=sources)
    except ServerError:
        st.error("השרתים של Gemini עמוסים כרגע (שגיאה 503). נסי שוב בעוד דקה.")
        st.stop()
    except ClientError as error:
        if error.code == 429:
            st.error("נגמרה המכסה של Gemini (שגיאה 429). נסי שוב מאוחר יותר.")
        else:
            st.error(f"הבקשה ל־Gemini נכשלה (שגיאה {error.code}).")
        st.stop()

    st.subheader("תשובה")
    st.write(response.response)
    show_sources(sources)

    files = [source.node.metadata["file_name"] for source in sources]
    st.session_state.history.insert(
        0, {"question": question, "answer": response.response, "files": files}
    )

show_history(previous_questions)
