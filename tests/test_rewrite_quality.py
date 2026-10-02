import pytest

from modules.sentence_rewrite.quality import (
    QUALITY_REASONS,
    CandidateQuality,
    assess_span,
    quality_exclusions,
)
from modules.sentence_rewrite.sentences import SentenceSpan, sentence_spans


def assessments(text, cleanup='conservative'):
    spans = sentence_spans(text)
    return [assess_span(text, span,
                        spans[index - 1] if index else None,
                        spans[index + 1] if index + 1 < len(spans) else None,
                        cleanup=cleanup)
            for index, span in enumerate(spans)]


@pytest.mark.parametrize('text', ['Yes.', 'Certainly.', '42.', '3.14.', 'Ні.',
                                 'نعم.', '世界.', '雨が降っていて庭の花が濡れています.'])
def test_short_numeric_and_multilingual_sentences_remain_eligible(text):
    quality = assessments(text, cleanup='scanned_book')[0]
    assert not quality_exclusions(quality)


def test_unicode_word_runs_keep_marks_and_internal_joiners():
    text = "Cafe\u0301 can't well-being देवनागरी 世界 42."
    quality = assessments(text)[0]
    assert quality.word_count == 6
    assert quality.letter_count > 20


def test_nabokov_boundary_flags_both_sides_without_changing_text():
    text = 'It 5 s the coldest and meanest in. the whole house. Another valid sentence.'
    qualities = assessments(text, cleanup='scanned_book')
    assert quality_exclusions(qualities[0]) == ('suspicious_boundary', 'dangling_function_boundary', 'ocr_quote_digit')
    assert quality_exclusions(qualities[1]) == ('suspicious_boundary', 'dangling_function_boundary')
    assert not quality_exclusions(qualities[2])
    assert not any(quality_exclusions(item) for item in assessments(text))
    assert text == 'It 5 s the coldest and meanest in. the whole house. Another valid sentence.'


@pytest.mark.parametrize('text', [
    'it was the coldest room in. the whole house.',
    'A perfectly complete sentence.\n\nthe whole house.',
    'A perfectly complete sentence.\r\n\r\nthe whole house.',
    'A perfectly complete sentence.\u2029the whole house.',
    'A perfectly complete sentence.\fthe whole house.',
    'A deliberately hesitant sentence... the whole house.',
    'He said a complete sentence." the whole house.',
    'He said a complete sentence. "the whole house.',
    'I ran. on and on.',
    'هذه جملة عربية كاملة. هذه جملة أخرى.',
    'これは句読点を持つ完全な文です. これは別の文です.',
])
def test_boundary_heuristic_does_not_reject_intentional_forms(text):
    assert not any(quality_exclusions(item) for item in assessments(text, cleanup='scanned_book'))


def test_boundary_context_is_occurrence_specific():
    text = 'It was the coldest room in. the whole house.\n\nthe whole house.'
    qualities = assessments(text, cleanup='scanned_book')
    assert 'suspicious_boundary' in qualities[1].flags
    assert 'suspicious_boundary' not in qualities[2].flags


@pytest.mark.parametrize(('text', 'reason'), [
    ('The damaged \ufffd word.', 'replacement_character'),
    ('The damaged \x00 word.', 'control_character'),
    ('The damaged \ud800 word.', 'control_character'),
])
def test_obvious_character_damage_is_reported_and_can_be_retained(text, reason):
    quality = assess_span(text, SentenceSpan(0, len(text), text))
    assert reason in quality_exclusions(quality)
    assert quality_exclusions(quality, policy='off') == ()
    assert reason in QUALITY_REASONS


def test_symbol_noise_only_in_scan_mode_and_no_general_symbol_penalty():
    text = '|||__|¦¦ text.'
    assert quality_exclusions(assessments(text, cleanup='scanned_book')[0]) == ('symbol_noise',)
    assert not quality_exclusions(assessments(text)[0])
    for clean in ('The price is $100.', 'The sum is x + y = 42.', '😀😀😀😀😀😀.', '42.'):
        assert not quality_exclusions(assessments(clean, cleanup='scanned_book')[0])


def test_whitespace_and_script_joiners_are_not_control_damage():
    text = 'A wrapped\tline\nwith\rspaces and نامه\u200cای.'
    quality = assess_span(text, SentenceSpan(0, len(text), text))
    assert 'control_character' not in quality.flags


def test_short_flag_is_descriptive_and_bad_policy_fails():
    quality = CandidateQuality(1, 3, ('short_reference',))
    assert quality_exclusions(quality) == ()
    with pytest.raises(ValueError, match='quality policy'):
        quality_exclusions(quality, policy='invented')


@pytest.mark.parametrize('prefix', ['If.', 'in.', 'of.', 'to.'])
def test_scanned_dangling_function_boundary_flags_both_sides(prefix):
    text = prefix + ' understand that I expected a clear explanation of the situation.'
    qualities = assessments(text, cleanup='scanned_book')
    assert len(qualities) == 2
    assert all('dangling_function_boundary' in quality_exclusions(quality) for quality in qualities)
    assert not any(quality_exclusions(item) for item in assessments(text))


def test_dangling_final_word_is_detected_inside_a_longer_passage():
    # Original test prose reproducing the observed OCR structure, not a corpus
    # quotation: the spurious "If." is attached to a much longer left span.
    text = ('He explained the plan again, but she kept interrupting and must If. '
            'understand that the arrangement depended on a clear account of the situation.')
    qualities = assessments(text, cleanup='scanned_book')
    assert len(qualities) == 2
    assert all('dangling_function_boundary' in quality_exclusions(quality) for quality in qualities)
    assert not any(quality_exclusions(item) for item in assessments(text))


@pytest.mark.parametrize('text', [
    'Yes. understand that we can continue.',
    'Certainly. understand that we can continue.',
    'If.\n\nunderstand that we can continue.',
    'If... understand that we can continue.',
    '"If." understand that we can continue.',
    'If. "understand that we can continue.',
    'In. yes.',
    'If. Understand that we can continue.',
    'He looked within. understand that we can continue.',
    'The number is 15. understand that we can continue.',
    'He came in. Yes, we can continue.',
    'He came in.\n\nyes, we can continue.',
])
def test_dangling_rule_preserves_short_dialogue_and_explicit_boundaries(text):
    assert not any(quality_exclusions(item) for item in assessments(text, cleanup='scanned_book'))


@pytest.mark.parametrize('text', [
    'It 5 s the coldest and meanest room.',
    '‘It 5 s the coldest room.',
    'He 5 s beginning to understand.',
    '‘Hi! 5 she said, and we went inside.',
    '"Why? 5 he asked, waiting for an answer.',
    '‘You fool, 9 she said, turning toward the window.',
    '"Really? 0 he replied, studying the note.',
])
def test_scanned_english_quote_digit_patterns_are_quarantined_without_repair(text):
    quality = assess_span(text, SentenceSpan(0, len(text), text), cleanup='scanned_book')
    assert 'ocr_quote_digit' in quality_exclusions(quality)
    assert not quality_exclusions(assess_span(text, SentenceSpan(0, len(text), text)))
    assert not quality_exclusions(quality, policy='off')


@pytest.mark.parametrize('text', [
    'He bought 5 apples.',
    'The result decreased by 15 percent.',
    'Give it 5 s to settle.',
    'It takes 5 s to settle.',
    '"Hi!" she said, and we went inside.',
    '"There were 5 apples," she said.',
    '“There were 15 percent fewer apples,” he replied.',
    '"At five, 15 she said, I shall come back.',
    '"At five, 9 apples were already on the table.',
    '"The amount was 9," she said.',
    'قالت إن لديها 5 تفاحات.',
    '她说有5个苹果.',
])
def test_quote_digit_rule_preserves_numbers_and_intact_dialogue(text):
    assert not any(quality_exclusions(item) for item in assessments(text, cleanup='scanned_book'))


@pytest.mark.parametrize('text', ['. .', '* .', '!!!.', '... .'])
def test_punctuation_fragments_are_identified(text):
    quality = assess_span(text, SentenceSpan(0, len(text), text))
    assert quality_exclusions(quality) == ('punctuation_fragment',)
    assert not quality_exclusions(quality, policy='off')


@pytest.mark.parametrize('text', ['3 + 4 = 7.', '$5.', '£.', '+.', '😀.', '∑.'])
def test_meaningful_symbols_and_math_are_not_punctuation_fragments(text):
    quality = assess_span(text, SentenceSpan(0, len(text), text), cleanup='scanned_book')
    assert not quality_exclusions(quality)


@pytest.mark.parametrize('separator', ['\n\n', '\r\n\r\n', '\u2029', '\f'])
def test_scanned_leading_digit_separated_from_prose_is_flagged(separator):
    text = '9' + separator + 'He reached toward the shelf.'
    quality = assess_span(text, SentenceSpan(0, len(text), text), cleanup='scanned_book')
    assert quality_exclusions(quality) == ('leading_digit_paragraph',)
    assert not quality_exclusions(assess_span(text, SentenceSpan(0, len(text), text)))
    assert not quality_exclusions(quality, policy='off')


@pytest.mark.parametrize('text', [
    '9 apples were on the shelf.',
    '15 percent of the apples were missing.',
    '9 He reached toward the shelf.',
    '9\nHe reached toward the shelf.',
    '15\n\nHe reached toward the shelf.',
    '9\n\n42.',
    '9\n\n£5.',
    '9\n\nTitle.',
])
def test_leading_digit_rule_does_not_treat_ordinary_numbers_as_ocr(text):
    quality = assess_span(text, SentenceSpan(0, len(text), text), cleanup='scanned_book')
    assert not quality_exclusions(quality)


def test_scanned_heuristics_report_ambiguous_numbered_prose_and_phrasal_verbs():
    # These intentional constructions resemble OCR damage.  The decision is
    # inspectable and opt-out, not a claim of proven corruption or a repair.
    text = 'He came in. yes, he walked through the room.'
    qualities = assessments(text, cleanup='scanned_book')
    assert all('dangling_function_boundary' in quality.flags for quality in qualities)
    assert all(not quality_exclusions(quality, policy='off') for quality in qualities)
    numbered = '9\n\nThis section describes the arrangement.'
    quality = assess_span(numbered, SentenceSpan(0, len(numbered), numbered), cleanup='scanned_book')
    assert 'leading_digit_paragraph' in quality.flags
    assert 'numbered heading' in QUALITY_REASONS['leading_digit_paragraph']
    assert not quality_exclusions(quality, policy='off')
