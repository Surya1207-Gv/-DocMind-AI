import pytest
from backend.chat_engine import (
    check_conversational,
    classify_and_normalize_question,
    get_mode_temperature
)

def test_check_conversational():
    # Greetings & simple responses
    assert check_conversational("hi") is True
    assert check_conversational("hello") is True
    assert check_conversational("how are you?") is True
    assert check_conversational("thank you so much") is True
    assert check_conversational("never mind") is True
    assert check_conversational("It is out of the document") is True
    assert check_conversational("okay thanks") is True
    assert check_conversational("") is True
    
    # Factual queries
    assert check_conversational("What is Generative AI?") is False
    assert check_conversational("Tell me about the budget of 2024") is False
    assert check_conversational("Compare the candidates") is False

def test_classify_and_normalize_question():
    # Conversational greeting
    res = classify_and_normalize_question("hello there")
    assert res["classification"] == "CONVERSATIONAL"
    assert res["corrected_query"] == "hello there"
    
    # Ambiguous/short query
    res = classify_and_normalize_question("v")
    assert res["classification"] == "AMBIGUOUS"
    
    # Ambiguous predefined keywords
    res = classify_and_normalize_question("roles")
    assert res["classification"] == "AMBIGUOUS"
    
    # Factual query (note that our implementation bypasses classification directly to FACTUAL for normal questions)
    res = classify_and_normalize_question("What are the system requirements?")
    assert res["classification"] == "FACTUAL"
    assert res["corrected_query"] == "What are the system requirements?"

def test_get_mode_temperature():
    assert get_mode_temperature("qa") == 0.2
    assert get_mode_temperature("summary") == 0.3
    assert get_mode_temperature("deep") == 0.5
    assert get_mode_temperature("eli5") == 0.6
    assert get_mode_temperature("unknown_mode") == 0.2 # default fallback
