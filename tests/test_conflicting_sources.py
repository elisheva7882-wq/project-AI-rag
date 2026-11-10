"""Controlled conflict/commitment responses; these do not evaluate a live LLM."""

from __future__ import annotations

from src.prompts import NO_ANSWER_TEXT
from src.qa_service import QAService

from qa_helpers import ControlledLLM, StaticRetriever, passage, response_for, settings_for


def test_initial_and_final_records_remain_separate_with_both_citations(tmp_path):
    initial = passage(9, text="הדיווח הטכני הראשוני רשם שלוש תקלות ונכתב לפני השלמת הבדיקה.", document_type="דוח ראשוני, אינו סופי", period="לפני הבדיקה הסופית")
    final = passage(10, text="לאחר הבדיקה הסופית תוקן המרשם הטכני ונרשמו בו חמש תקלות מאומתות.", document_type="דוח סופי", period="לאחר הבדיקה הסופית")
    response = {
        "status": "answered",
        "claims": [
            {"text": "בדיווח הראשוני נרשמו שלוש תקלות.", "evidence": [{"chunk_id": initial.chunk_id, "quote": initial.text}]},
            {"text": "לאחר הבדיקה הסופית נרשמו חמש תקלות מאומתות.", "evidence": [{"chunk_id": final.chunk_id, "quote": final.text}]},
        ],
    }
    service = QAService(settings_for(tmp_path), retriever=StaticRetriever((initial, final)), llm=ControlledLLM(response), offline=True)
    answer = service.ask("כמה תקלות תועדו בבדיקה הטכנית?")
    assert answer.status == "answered"
    assert {source.document_id for source in answer.sources} == {"ALT-09", "ALT-10"}
    assert "ראשוני" in answer.text and "סופית" in answer.text
    assert "שלוש" in answer.text and "חמש" in answer.text


def test_only_preliminary_source_cannot_be_presented_as_final_fact(tmp_path):
    initial = passage(9, text="הדיווח הטכני הראשוני רשם שלוש תקלות ונכתב לפני השלמת הבדיקה.", document_type="דוח ראשוני, אינו סופי")
    service = QAService(settings_for(tmp_path), retriever=StaticRetriever((initial,)), llm=ControlledLLM(response_for(initial, text="תועדו שלוש תקלות סופיות ומאומתות.")), offline=True)
    answer = service.ask("מה התוצאה הסופית של הבדיקה הטכנית?")
    assert answer.status == "no_answer"
    assert answer.text == NO_ANSWER_TEXT
    assert not answer.sources


def test_explicit_initial_report_is_marked_partial_when_final_evidence_is_absent(tmp_path):
    initial = passage(13, text="בפרוטוקול הטכני הראשוני נרשמו שלוש תוצאות שטרם אושרו בבדיקה סופית.", document_type="פרוטוקול ראשוני")
    response = response_for(initial, text="לפי הפרוטוקול הראשוני נרשמו שלוש תוצאות שטרם אושרו.")
    service = QAService(settings_for(tmp_path), retriever=StaticRetriever((initial,)), llm=ControlledLLM(response), offline=True)
    answer = service.ask("מה ידוע על התוצאות הטכניות?")
    assert answer.status == "partial"
    assert "ראשוני" in answer.text
    assert {source.document_id for source in answer.sources} == {"ALT-13"}


def test_commitment_is_reported_without_claiming_implementation(tmp_path):
    commitment = passage(15, text="מכתב הבדיקה כולל התחייבות להכין נוהל חדש, אך אינו מאשר שהנוהל נחקק או בוצע.", document_type="מכתב התחייבויות")
    response = response_for(commitment, text="במקור נרשמה התחייבות להכין נוהל חדש; אין בו אישור לחקיקה או לביצוע.", status="partial")
    service = QAService(settings_for(tmp_path), retriever=StaticRetriever((commitment,)), llm=ControlledLLM(response), offline=True)
    answer = service.ask("האם הנוהל החדש כבר בוצע?")
    assert answer.status == "partial"
    assert "התחייבות" in answer.text
    assert "אין בו אישור" in answer.text
    assert answer.sources[0].quote == commitment.text


def test_source_authority_does_not_follow_numeric_archive_id(tmp_path):
    binding = passage(2, text="הרשומה הטכנית המחייבת קובעת ששתי בדיקות נוספות נדרשות להשלמת הנוהל.", document_type="רשומה מחייבת")
    preliminary = passage(19, text="רשומת הבדיקה הראשונית הציעה בדיקה אחת בלבד לפני פרסום הנוהל המחייב.", document_type="רשומה ראשונית")
    response = {
        "status": "answered",
        "claims": [
            {"text": "בתחילה הוצעה בדיקה אחת בלבד.", "evidence": [{"chunk_id": preliminary.chunk_id, "quote": preliminary.text}]},
            {"text": "ברשומה המחייבת נקבע שנדרשות שתי בדיקות נוספות.", "evidence": [{"chunk_id": binding.chunk_id, "quote": binding.text}]},
        ],
    }
    service = QAService(settings_for(tmp_path), retriever=StaticRetriever((binding, preliminary)), llm=ControlledLLM(response), offline=True)
    answer = service.ask("כמה בדיקות נדרשות לפי הנוהל המחייב?")
    assert answer.status == "answered"
    assert {source.document_id for source in answer.sources} == {"ALT-02", "ALT-19"}
