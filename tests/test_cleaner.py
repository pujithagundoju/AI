import sys
import os
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.cleaner import TextCleaner

def test_clean_text_basic():
    cleaner = TextCleaner()
    # Test lowercasing and basic split
    text = "Hello World"
    result = cleaner.clean_text(text)
    assert "hello" in result
    assert "world" in result

def test_clean_text_urls():
    cleaner = TextCleaner()
    text = "Check this out https://example.com/resume"
    result = cleaner.clean_text(text)
    assert "https" not in result
    assert "example.com" not in result

def test_clean_text_punctuation_numbers():
    cleaner = TextCleaner()
    text = "I have 5 years! of experience @ Google."
    result = cleaner.clean_text(text)
    assert "5" not in result
    assert "!" not in result
    assert "@" not in result
    # "experience" is 10 chars, > 2, not stopword.
    assert "experience" in result

def test_clean_text_stopwords_lemmatization():
    cleaner = TextCleaner()
    # "The" is stopword. "running" -> "run" / "running" (WordNet defaults to Noun lemmatization mostly if POS not tagged, but let's check basic structure).
    # "are" is stopword.
    text = "The dogs are running."
    result = cleaner.clean_text(text)
    assert "the" not in result
    assert "are" not in result
    # 'dog' should be lemmatized from 'dogs'
    assert "dog" in result
