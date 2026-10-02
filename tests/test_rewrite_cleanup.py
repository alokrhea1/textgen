from array import array

from modules.sentence_rewrite.cleanup import clean_text


def test_conservative_preserves_paragraphs_and_original_offsets():
    original = '  First\twrapped\r\n sentence.\r\n\r\nSecond para.'
    result = clean_text(original)
    assert result.text == ' First wrapped sentence.\n\nSecond para.'
    start = result.text.index('First')
    end = result.text.index('.') + 1
    raw_start, raw_end = result.source_span(start, end)
    assert original[raw_start:raw_end] == 'First\twrapped\r\n sentence.'
    assert isinstance(result.offsets, array)
    assert result.offsets.itemsize == 4


def test_soft_hyphen_join_and_optional_compound_join():
    original = 'A dis\u00ad\n connected word and well-\nformed phrase.'
    result = clean_text(original)
    assert result.text == 'A disconnected word and well- formed phrase.'
    assert clean_text(original, join_hyphenated_lines=True).text == 'A disconnected word and wellformed phrase.'
    assert result.source_span(0, len(result.text)) == (0, len(original))


def test_scanned_furniture_only_opt_in_and_none_unchanged():
    original = '17\nCHAPTER IV\nA com¬\nplete sentence.\n\n42\nAnother sentence.'
    assert clean_text(original, 'none').text == original
    conservative = clean_text(original)
    assert '17' in conservative.text and '42' in conservative.text
    scanned = clean_text(original, 'scanned_book')
    assert scanned.text == 'A complete sentence.\n\nAnother sentence.'
    assert scanned.report['counts']['furniture_lines'] == 3
    assert 'meaningful' in scanned.report['warning']


def test_unicode_offsets_are_characters_and_bom_is_accounted():
    original = '\ufeffÉlan\t世界\r\nends.'
    result = clean_text(original)
    assert result.text == 'Élan 世界 ends.'
    assert result.source_span(0, len(result.text)) == (1, len(original))


def test_reports_sample_categories_once_with_context():
    original = ('Repeated wrap\r\n' * 20) + '\r\n17\r\nA complete sentence.'
    result = clean_text(original, 'scanned_book')
    snippets = result.report['snippets']
    assert len({item['kind'] for item in snippets}) == len(snippets)
    furniture = next(item for item in snippets if item['kind'] == 'furniture_lines')
    assert '17' in furniture['before']
    assert 'A complete' in furniture['after']
    assert all(len(item['before']) <= 160 and len(item['after']) <= 160 for item in snippets)


def test_unicode_paragraph_line_and_page_separators_preserve_offsets():
    original = 'First\u2028wrapped sentence.\u2029Second sentence.\fThird sentence.'
    result = clean_text(original)
    assert result.text == 'First wrapped sentence.\n\nSecond sentence.\n\nThird sentence.'
    assert result.report['counts']['paragraph_separators'] == 2
    assert result.report['counts']['line_separators'] == 1
    start = result.text.index('Third')
    assert result.source_span(start, len(result.text)) == (original.index('Third'), len(original))
    assert clean_text(original, 'none').text == original
