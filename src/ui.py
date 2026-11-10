"""Session-local Streamlit presentation over the existing archive service.

Opening, rerunning, and clearing the page never embed documents or call Gemini.
Only the current submitted question is passed to QAService, without chat history.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import streamlit as st

from config import Settings, load_settings
from src.file_loader import CorpusError, load_corpus
from src.indexer import IndexNotReadyError, IndexingError, validate_index
from src.models import Answer, MAX_QUESTION_CHARS, QuestionError, validate_question
from src.qa_service import QAService
from src.vector_store import VectorStoreError


PAGE_TITLE = "KingdomAI — הארכיון החכם של ממלכת אלתריה"
EXAMPLE_QUESTIONS = (
    "מהם שבעת בתי החותם?",
    "למה בוטלה הכרזת הניצחון בבחירות 320?",
    "מי הוכתר ב־321?",
    "מה היו מסקנות חקירת גשר הערפל?",
)
MAX_HISTORY_TURNS = 20
_ALT_IDS = re.compile(r"\bALT-\d{2}\b")
_STYLES = """
<style>
  :root { --archive-ink: #203d32; --archive-gold: #aa8750; }
  html, body, [data-testid="stApp"] { font-family: "Segoe UI", Arial, sans-serif; }
  main, [data-testid="stMainBlockContainer"], [data-testid="stSidebar"], [data-testid="stBottom"] {
    direction: rtl; text-align: right;
  }
  [data-testid="stMainBlockContainer"] { max-width: 1040px; padding-top: 2.5rem; }
  [data-testid="stSidebar"] { border-right: 1px solid #dcdace; }
  [data-testid="stSidebarContent"] { padding-top: 1.4rem; }
  [data-testid="stHeadingWithActionElements"] h1 {
       color: var(--archive-ink); font-weight: 650; letter-spacing: -.045em;
       font-size: clamp(1.8rem, 3.2vw, 2.8rem); line-height: 1.3; }
  h2, h3 { color: var(--archive-ink); }
  [data-testid="stText"] { font-family: "Segoe UI", Arial, sans-serif;
    line-height: 1.85; white-space: pre-wrap; overflow-wrap: anywhere;
    unicode-bidi: plaintext; }
  [data-testid="stCaptionContainer"] { line-height: 1.8; }
  [data-testid="stChatMessage"] { flex-direction: row; gap: .8rem;
    border: 1px solid #deded3; border-radius: 14px; padding: 1.35rem; }
  [data-testid="stChatMessageContent"] { min-width: 0; }
  [data-testid="stChatInput"] { border-radius: 14px; border-color: #bcc8be; }
  [data-testid="stChatInput"] textarea { direction: rtl; text-align: right; }
  [data-testid="stButton"] button { border-radius: 10px; min-height: 3rem;
    text-align: right; line-height: 1.6; white-space: normal; }
  [data-testid="stButton"] button:hover { border-color: #62806c; }
  [data-testid="stMetric"] { padding: .6rem .8rem; border-radius: 10px;
    background: #f8f6f0; border: 1px solid #deded3; }
  [data-testid="stMetricValue"] { color: var(--archive-ink); }
  [data-testid="stExpander"] { border-color: #d9ddd1; }
  .archive-kicker { display: flex; align-items: center; gap: .7rem;
    color: #69766c; font-size: .85rem; margin-bottom: .7rem; }
  .archive-mark { display: inline-flex; justify-content: center; align-items: center;
    width: 2rem; height: 2rem; border: 1px solid var(--archive-gold);
    border-radius: 50%; color: var(--archive-ink); font-family: Georgia, serif; }
  @media (max-width: 640px) {
    [data-testid="stMainBlockContainer"] { padding-top: 3.5rem; }
    .archive-kicker { display: none; }
    [data-testid="stChatMessage"] { padding: .9rem; }
  }
</style>
"""


@dataclass(frozen=True)
class SetupStatus:
    ready: bool
    documents: int = 0
    chunks: int = 0
    generation_id: str = ""
    code: str | None = None
    message: str = ""


@dataclass(frozen=True)
class ChatTurn:
    question: str
    answer: Answer


def check_readiness(settings: Settings) -> SetupStatus:
    """Validate local prerequisites and persisted vectors without an API call.

    Deliberately not cached by manifest filename: changed source bytes or Chroma
    content must stop the UI even if a manifest's timestamp has not changed.
    """
    try:
        if not settings.knowledge_base_path.is_dir() or not any(settings.knowledge_base_path.glob("*.md")):
            return SetupStatus(False, code="missing_documents", message="מסמכי הארכיון חסרים. יש להכין את מאגר המסמכים לפני תחילת השיחה.")
        documents = len(load_corpus(settings.knowledge_base_path))
    except (CorpusError, OSError):
        return SetupStatus(False, code="invalid_documents", message="מאגר מסמכי הארכיון אינו תקין. יש לוודא שקיימים בדיוק 19 מסמכי המקור.")
    if not settings.has_api_key:
        return SetupStatus(False, documents=documents, code="missing_api_key", message="חסר מפתח Gemini. יש להגדיר GEMINI_API_KEY בקובץ .env המקומי ואז לרענן את מצב הארכיון.")
    try:
        manifest = validate_index(settings)
    except IndexNotReadyError:
        return SetupStatus(False, documents=documents, code="index_not_ready", message="האינדוקס טרם הושלם. יש להשלים את בניית אינדקס הארכיון לפני שליחת שאלות.")
    except (IndexingError, CorpusError, VectorStoreError, OSError):
        return SetupStatus(False, documents=documents, code="index_invalid", message="האינדקס אינו תואם למסמכים או להגדרות. יש לבדוק את האינדקס ולבנות אותו מחדש לפי הצורך.")
    return SetupStatus(True, documents=manifest["documents"], chunks=manifest["chunks"], generation_id=manifest["generation_id"])


def _text(value: str) -> None:
    # Plain text prevents model/user Markdown from loading external images or
    # injecting links/HTML. Isolate archive IDs for mixed Hebrew/LTR punctuation.
    st.text(_ALT_IDS.sub(lambda match: "\u2066" + match[0] + "\u2069", value), width="stretch", text_alignment="right")


def _sync_session(settings: Settings, status: SetupStatus) -> bool:
    st.session_state.setdefault("turns", [])
    identity = (settings, status.generation_id, status.code)
    previous = st.session_state.get("_runtime_identity")
    changed = previous is not None and previous != identity
    if previous != identity:
        st.session_state.pop("_qa_service", None)
        st.session_state["turns"] = []
        st.session_state["_runtime_identity"] = identity
    return changed


def _sidebar(settings: Settings | None, status: SetupStatus) -> None:
    with st.sidebar:
        st.header("הארכיון המלכותי")
        st.caption("מסמכי ממלכת אלתריה · חיפוש ותשובות")
        if status.ready:
            st.success("האינדקס מוכן לחיפוש")
        else:
            st.warning("נדרשת השלמת הגדרה")
        docs, chunks = st.columns(2)
        docs.metric("מסמכים", status.documents)
        chunks.metric("מקטעים", status.chunks if status.chunks else "—")
        st.caption("החיבור ל־Gemini נבדק בעת שליחת שאלה.")
        with st.expander("מודלים פעילים"):
            if settings is not None:
                st.caption("ניסוח תשובות")
                _text(settings.llm_model)
                st.caption("חיפוש במסמכים")
                _text(settings.embedding_model)
            else:
                st.caption("המודלים יוצגו לאחר תיקון ההגדרות.")
        st.divider()
        if st.button("ניקוי השיחה", key="clear_chat", disabled=not st.session_state.get("turns"), width="stretch"):
            st.session_state["turns"] = []
            st.rerun()
        st.button("רענון מצב הארכיון", key="refresh_status", width="stretch")
        st.caption("השיחה זמנית ומוגבלת לחיבור הנוכחי. נשמרות עד עשרים שאלות ותשובות אחרונות.")
        st.caption("כל שאלה נענית בפני עצמה מתוך מסמכי הארכיון.")


def _setup_help(status: SetupStatus) -> None:
    st.warning("הארכיון עדיין אינו מוכן לשאלות.")
    _text(status.message)
    with st.expander("הנחיות הפעלה"):
        if status.code in ("missing_documents", "invalid_documents"):
            st.caption("מתיקיית הפרויקט, יש להכין את 19 מסמכי המקור באמצעות הכלי:")
            st.code("python -m scripts.prepare_knowledge --source <archive-path>", language="powershell")
        if status.code in ("missing_api_key", "configuration_error"):
            st.caption("יש לעדכן את קובץ .env המקומי לפי .env.example. מפתח Gemini מוזן רק בקובץ המקומי.")
        if status.code in ("index_not_ready", "index_invalid"):
            st.caption("בנייה או המשך של בניית האינדקס מתבצעים במפורש משורת הפקודה:")
            st.code("python -m scripts.ingest --rebuild", language="powershell")
        st.caption("לאחר השלמת ההגדרה אפשר ללחוץ על רענון מצב הארכיון.")


def _render_answer(answer: Answer) -> None:
    if answer.status == "error":
        heading = {
            "service_timeout": "תם זמן ההמתנה לתשובה.",
            "connection_error": "החיבור לשירות Gemini לא הושלם.",
            "rate_limit": "הבקשה נעצרה בגלל מגבלת שימוש.",
            "access_denied": "Gemini דחה את הגישה.",
            "model_unavailable": "המודל אינו זמין בחשבון.",
            "index_not_ready": "אינדקס הארכיון טרם הושלם.",
            "index_invalid": "האינדקס דורש בדיקה.",
            "missing_api_key": "נדרש מפתח Gemini.",
        }.get(answer.error_code, "לא התקבלה תשובה מהשירות.")
        st.error(heading)
    elif answer.status == "no_answer":
        st.info("לא נמצא מידע מספיק בארכיון.")
    elif answer.status == "partial":
        st.warning("זו תשובה חלקית לפי המקטעים שאוחזרו.")
    _text(answer.text)
    # if answer.status in ("answered", "partial") and answer.sources:
    #     with st.expander(f"מקורות מהארכיון ({len(answer.sources)})"):
    #         for number, source in enumerate(answer.sources):
    #             if number:
    #                 st.divider()
    #             _text(f"{source.document_id} · {source.title}")
    #             details = " · ".join(value for value in (source.document_type, source.period) if value)
    #             if details:
    #                 _text(details)
    #             if source.section:
    #                 _text(f"סעיף: {source.section}")
    #             _text(f"״{source.quote}״")
    #             _text(f"קובץ מקור: {source.source_file}")


def _ask(settings: Settings, question: str) -> Answer:
    try:
        if "_qa_service" not in st.session_state:
            st.session_state["_qa_service"] = QAService(settings)
        return st.session_state["_qa_service"].ask(question)
    except Exception as e:
        import traceback
        print("KINGDOMAI ERROR:")
        traceback.print_exc()

        return Answer(
             text="לא ניתן להשלים את הבקשה כרגע. אפשר לנסות שוב מאוחר יותר.",
             status="error",
             error_code="service_unavailable"
    )
 

def render_app() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon="📜", layout="wide", initial_sidebar_state="expanded")
    st.markdown(_STYLES, unsafe_allow_html=True)
    st.markdown('<div class="archive-kicker"><span class="archive-mark">א</span>מסמכי הממלכה, עם המקור לצדם</div>', unsafe_allow_html=True)
    st.title(PAGE_TITLE)
    st.caption("שאלו על ההיסטוריה, שבעת בתי החותם, חוקי הכתר והאירועים — התשובות מבוססות רק על מסמכי הארכיון")
    try:
        settings = load_settings()
        status = check_readiness(settings)
    except Exception:
        settings = None
        status = SetupStatus(False, code="configuration_error", message="לא ניתן לטעון את הגדרות הארכיון. יש לבדוק את קובץ .env המקומי ואת הרשאות הגישה לקבצי הפרויקט.")
        st.session_state.pop("_qa_service", None)
        st.session_state.pop("_runtime_identity", None)
        st.session_state["turns"] = []
    if settings is not None and _sync_session(settings, status):
        st.info("הגדרות הארכיון השתנו. נפתחה שיחה חדשה.")
    _sidebar(settings, status)
    st.divider()
    if not status.ready:
        _setup_help(status)
    turns = st.session_state.get("turns", [])
    selected_question = None
    if not turns:
        st.subheader("מה תרצו למצוא בארכיון?")
        st.caption("אפשר להתחיל באחת השאלות הבאות, או לכתוב שאלה משלכם.")
        for offset in range(0, len(EXAMPLE_QUESTIONS), 2):
            columns = st.columns(2)
            for column, example in zip(columns, EXAMPLE_QUESTIONS[offset:offset + 2]):
                number = EXAMPLE_QUESTIONS.index(example)
                with column:
                    if st.button(example, key=f"example_{number}", disabled=not status.ready, width="stretch"):
                        selected_question = example
    for turn in turns:
        with st.chat_message("user", avatar="👤"):
            _text(turn.question)
        with st.chat_message("assistant", avatar="📜"):
            _render_answer(turn.answer)
    prompt = st.chat_input("מה תרצו לדעת על ממלכת אלתריה?", key="question", max_chars=MAX_QUESTION_CHARS, disabled=not status.ready)
    question = selected_question if selected_question is not None else prompt
    if question is None or not status.ready or settings is None:
        return
    try:
        validate_question(question)
    except QuestionError as exc:
        st.warning(str(exc))
        return
    with st.chat_message("user", avatar="👤"):
        _text(question)
    with st.chat_message("assistant", avatar="📜"):
        with st.spinner("מחפש בארכיון ומנסח תשובה…"):
            answer = _ask(settings, question)
        _render_answer(answer)
    st.session_state["turns"] = [*turns, ChatTurn(question, answer)][-MAX_HISTORY_TURNS:]
    st.rerun()
