from streamlit_app import update_sentence


def test_update_sentence_appends_after_stable_repeats() -> None:
    sentence: list[str] = []
    sentence, last_sign, stable_count = update_sentence(
        sentence=sentence,
        sign="A",
        confidence=0.9,
        threshold=0.25,
        last_sign="",
        stable_count=0,
    )
    assert sentence == []

    sentence, last_sign, stable_count = update_sentence(
        sentence=sentence,
        sign="A",
        confidence=0.9,
        threshold=0.25,
        last_sign="A",
        stable_count=1,
    )
    assert sentence == []

    sentence, last_sign, stable_count = update_sentence(
        sentence=sentence,
        sign="A",
        confidence=0.9,
        threshold=0.25,
        last_sign="A",
        stable_count=2,
    )
    assert sentence == ["A"]
