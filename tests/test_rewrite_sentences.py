import pytest

from modules.sentence_rewrite.sentences import (
    SentenceSpan, first_sentence, last_sentence, replace_sentence, sentence_spans,
)


@pytest.mark.parametrize(('text', 'expected'), [
    ('First. Second. unfinished', ['First.', 'Second.']),
    ('  First.\n\tSecond.  ', ['First.', 'Second.']),
    ('Dr. Smith paid 3.14 dollars. Next', ['Dr. Smith paid 3.14 dollars.']),
    ('Visit example.com. Done.', ['Visit example.com.', 'Done.']),
    ('J. Smith uses e.g. examples. fragment', ['J. Smith uses e.g. examples.']),
    ('He said “Go.” unfinished', ['He said “Go.”']),
    ('Wait... Next fragment', ['Wait...']),
    ('Why? Really!', []),
    ('Dr.', []),
])
def test_completed_spans(text, expected):
    spans = sentence_spans(text)
    assert [span.text for span in spans] == expected
    assert all(text[span.start:span.end] == span.text for span in spans)


def test_replacement_preserves_surrounding_text_and_repeated_rewrite():
    text = '  First.\n “Last.” \tunfinished'
    changed = replace_sentence(text, last_sentence(text), '  “New.”  ')
    assert changed == '  First.\n “New.” \tunfinished'
    assert replace_sentence(changed, last_sentence(changed), 'Again.') == '  First.\n Again. \tunfinished'


@pytest.mark.parametrize('replacement', ['', 'fragment', 'One. Two.', 'One. fragment', 'Dr.'])
def test_reject_invalid_replacement(replacement):
    with pytest.raises(ValueError, match='exactly one'):
        replace_sentence('Original.', last_sentence('Original.'), replacement)


def test_stale_and_fabricated_spans_rejected():
    stale = last_sentence('Old.')
    with pytest.raises(ValueError, match='span'):
        replace_sentence('Changed.', stale, 'New.')
    with pytest.raises(ValueError, match='span'):
        replace_sentence('Original.', SentenceSpan(1, 9, 'riginal.'), 'New.')


def test_missing_period_message():
    with pytest.raises(ValueError, match='period'):
        last_sentence('unfinished')


def test_streaming_decimal_quote_and_ellipsis():
    assert first_sentence('Costs 3.') is None
    assert first_sentence('Costs 3.14') is None
    assert first_sentence('Costs 3.14 dollars. ').text == 'Costs 3.14 dollars.'
    assert first_sentence('He said “Go.') is None
    assert first_sentence('He said “Go.”') is None
    assert first_sentence('He said “Go.” ').text == 'He said “Go.”'
    assert first_sentence('Wait...') is None
    assert first_sentence('Wait... ', final=False).text == 'Wait...'
    assert first_sentence('Done.', final=True).text == 'Done.'


def test_first_and_last_differ_and_suffix_ignored():
    text = 'One. Two. unfinished'
    assert first_sentence(text).text == 'One.'
    assert last_sentence(text).text == 'Two.'
