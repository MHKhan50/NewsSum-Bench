from newssum.text import numbers, repair_bbc_summary, split_sentences


def test_split_handles_abbreviations_and_quotes():
    text = 'Mr. Smith of the U.S. said it was "fine." He added that sales rose 2% to $11.1bn. Was it J. Brown? Yes!'
    assert split_sentences(text) == [
        'Mr. Smith of the U.S. said it was "fine."',
        "He added that sales rose 2% to $11.1bn.",
        "Was it J. Brown?",
        "Yes!",
    ]


def test_paragraph_breaks_end_sentences():
    assert split_sentences("Headline here\n\nFirst sentence. Second one.") == [
        "Headline here", "First sentence.", "Second one."]


def test_repair_bbc_summary_keeps_decimals_and_initialisms():
    raw = 'Sales rose to $11.1bn from $10.9bn.For the year, the U.S. firm said "no."The rest grew.'
    assert repair_bbc_summary(raw) == (
        'Sales rose to $11.1bn from $10.9bn. For the year, the U.S. firm said "no." The rest grew.')


def test_numbers_normalised():
    assert numbers("1,200 people paid £5.50 in 2005") == {"1200", "5.50", "2005"}
