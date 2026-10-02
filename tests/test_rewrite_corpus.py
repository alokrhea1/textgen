from types import SimpleNamespace

import numpy as np
import pytest

from modules.sentence_rewrite.corpus import CorpusConfig, CorpusIndex, CorpusQualityError


class FakeEncoder:
    config = SimpleNamespace(max_tokens=256, batch_size=2, device='cpu')
    identity = {'model': 'fake', 'revision': '1'}

    def __init__(self):
        self.encoded = 0
        self.fail = False

    def load(self, progress):
        pass

    def token_count(self, text):
        return len(text.split())

    def encode_documents(self, texts):
        if self.fail:
            raise RuntimeError('backend unavailable')
        self.encoded += len(texts)
        return [self.encode_sentence(text) for text in texts]

    def encode_sentence(self, text):
        return np.array([[1., 0.]], dtype=np.float32)

    encode_query = encode_sentence


def test_deduplicated_provenance_and_warm_index(tmp_path):
    source = tmp_path / 'sources'
    source.mkdir()
    (source / 'a.txt').write_text('A complete sentence.', encoding='utf-8')
    (source / 'b.txt').write_text('A complete sentence.', encoding='utf-8')
    encoder = FakeEncoder()
    config = CorpusConfig(str(source), max_sentences=1)
    index = CorpusIndex(tmp_path / 'cache')
    result = index.build(config, encoder)
    assert result['candidates'] == 1
    assert result['occurrences'] == 2
    assert index.build(config, encoder) == result
    assert encoder.encoded == 1
    hits = index.search('Another sentence.', encoder)
    assert len(hits[0].provenance) == 2


def test_content_hash_invalidation_and_failed_build_preserves_old(tmp_path):
    source = tmp_path / 'a.txt'
    source.write_text('First complete sentence.', encoding='utf-8')
    encoder = FakeEncoder()
    config = CorpusConfig(str(source), max_sentences=1)
    index = CorpusIndex(tmp_path / 'cache')
    before = index.build(config, encoder)
    source.write_text('Other complete sentence.', encoding='utf-8')
    with pytest.raises(ValueError, match='changed'):
        index.assert_current(config, encoder)
    encoder.fail = True
    with pytest.raises(RuntimeError, match='unavailable'):
        index.build(config, encoder)
    assert index.manifest == before
    encoder.fail = False
    index.build(config, encoder)
    index.assert_current(config, encoder)


def test_generator_token_count_controls_exact_length(tmp_path):
    source = tmp_path / 'a.txt'
    source.write_text('A short sentence. Another longer complete sentence.', encoding='utf-8')
    encoder = FakeEncoder()
    index = CorpusIndex(tmp_path / 'cache')
    index.build(CorpusConfig(str(source), max_sentences=2), encoder)
    counts = {'query': 9, 'A short sentence.': 9,
              'Another longer complete sentence.': 10,
              'A short sentence. Another longer complete sentence.': 19}
    hits = index.search('query', encoder, length_mode='tokens', token_tolerance=0,
                        token_counter=counts.__getitem__)
    assert [hit.text for hit in hits] == ['A short sentence.']
    with pytest.raises(ValueError, match='No corpus candidates'):
        index.search('query', encoder, length_mode='tokens', token_tolerance=0,
                     token_counter=lambda text: 7 if text == 'query' else 8)


def test_long_sentence_fails_and_symlink_rejected(tmp_path):
    source = tmp_path / 'a.txt'
    source.write_text('One two three four.', encoding='utf-8')
    encoder = FakeEncoder()
    encoder.config = SimpleNamespace(max_tokens=2, batch_size=2)
    index = CorpusIndex(tmp_path / 'cache')
    with pytest.raises(ValueError, match='exceeds embedding token limit'):
        index.build(CorpusConfig(str(source)), encoder)
    assert not index.ready
    link = tmp_path / 'link.txt'
    link.symlink_to(source)
    with pytest.raises(ValueError, match='Symlink'):
        index.build(CorpusConfig(str(link)), encoder)


def test_shared_path_lock_and_malformed_cache_force_recovery(tmp_path):
    source = tmp_path / 'a.txt'
    source.write_text('A complete sentence.', encoding='utf-8')
    index = CorpusIndex(tmp_path / 'cache')
    other = CorpusIndex(tmp_path / 'cache')
    assert index._lock is other._lock
    index.cache_dir.mkdir()
    index.path.write_bytes(b'corrupt sqlite')
    with pytest.raises(ValueError, match='corrupt'):
        index.build(CorpusConfig(str(source)), FakeEncoder())
    index.build(CorpusConfig(str(source)), FakeEncoder(), force=True)
    assert other.ready


def test_change_during_encoding_never_publishes(tmp_path):
    source = tmp_path / 'a.txt'
    source.write_text('A complete sentence.', encoding='utf-8')
    encoder = FakeEncoder()
    original = encoder.encode_documents

    def mutate(texts):
        result = original(texts)
        source.write_text('Changed complete sentence.', encoding='utf-8')
        return result

    encoder.encode_documents = mutate
    index = CorpusIndex(tmp_path / 'cache')
    with pytest.raises(ValueError, match='changed during indexing'):
        index.build(CorpusConfig(str(source)), encoder)
    assert not index.ready


def test_nuance_reranking_preserves_raw_scores_and_warns_short_pool(tmp_path):
    source = tmp_path / 'a.txt'
    source.write_text('First sentence. Second sentence.', encoding='utf-8')
    encoder = FakeEncoder()
    index = CorpusIndex(tmp_path / 'cache')
    index.build(CorpusConfig(str(source), max_sentences=1), encoder)

    class Reranker:
        def score(self, query, texts, **kwargs):
            return [0.9 if text == 'Second sentence.' else 0.1 for text in texts]

    messages = []
    hits = index.search('query', encoder, top_k=3, reranker=Reranker(),
                        progress=messages.append)
    assert hits[0].text == 'Second sentence.'
    assert hits[0].score == hits[0].nuance_score == 0.9
    assert hits[0].late_score == hits[1].late_score
    assert 'Only 2 matching candidates' in messages[-1]


def test_cancelled_search_and_invalid_reranker(tmp_path):
    import threading
    source = tmp_path / 'a.txt'
    source.write_text('First sentence.', encoding='utf-8')
    encoder = FakeEncoder()
    index = CorpusIndex(tmp_path / 'cache')
    index.build(CorpusConfig(str(source)), encoder)
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(ValueError, match='cancelled'):
        index.search('query', encoder, cancel_event=cancelled)

    class InvalidReranker:
        def score(self, query, texts, **kwargs):
            return [float('nan')]

    with pytest.raises(ValueError, match='invalid candidate scores'):
        index.search('query', encoder, reranker=InvalidReranker())


def test_mixed_embedding_dimensions_never_publish(tmp_path):
    source = tmp_path / 'a.txt'
    source.write_text('First sentence. Second sentence.', encoding='utf-8')
    encoder = FakeEncoder()
    encoder.encode_documents = lambda texts: [
        np.ones((1, dimension), dtype=np.float32) for dimension in (2, 3)
    ]
    index = CorpusIndex(tmp_path / 'cache')
    with pytest.raises(ValueError, match='inconsistent matrix dimensions'):
        index.build(CorpusConfig(str(source), max_sentences=1), encoder)
    assert not index.ready


def test_matrix_budget_fails_before_publish(tmp_path, monkeypatch):
    from modules.sentence_rewrite import corpus
    monkeypatch.setattr(corpus, 'MAX_INDEX_BYTES', 4)
    source = tmp_path / 'a.txt'
    source.write_text('First sentence.', encoding='utf-8')
    index = CorpusIndex(tmp_path / 'cache')
    with pytest.raises(ValueError, match='index budget'):
        index.build(CorpusConfig(str(source)), FakeEncoder())
    assert not index.ready


def test_token_count_cache_key_and_new_generation_invalidation(tmp_path):
    source = tmp_path / 'a.txt'
    source.write_text('First sentence.', encoding='utf-8')
    encoder = FakeEncoder()
    index = CorpusIndex(tmp_path / 'cache')
    config = CorpusConfig(str(source), max_sentences=1)
    index.build(config, encoder)
    calls = []

    def counter(text):
        calls.append(text)
        return 2

    for _ in range(2):
        index.search('query', encoder, length_mode='tokens', token_counter=counter,
                     token_counter_key=('model', 1))
    assert calls.count('First sentence.') == 1
    index.search('query', encoder, length_mode='tokens', token_counter=counter,
                 token_counter_key=('model', 2))
    assert calls.count('First sentence.') == 2
    index.build(config, encoder, force=True)
    index.search('query', encoder, length_mode='tokens', token_counter=counter,
                 token_counter_key=('model', 2))
    assert calls.count('First sentence.') == 3
    reopened = CorpusIndex(tmp_path / 'cache')
    reopened.search('query', encoder, length_mode='tokens', token_counter=counter,
                    token_counter_key=('model', 2))
    assert calls.count('First sentence.') == 4


def test_batched_scoring_preserves_late_matrix_ranking(tmp_path):
    source = tmp_path / 'a.txt'
    source.write_text('Different meaning. Same meaning.', encoding='utf-8')
    encoder = FakeEncoder()

    def encode(text):
        return np.array([[0., 1.]] if text == 'Different meaning.' else [[1., 0.]], dtype=np.float32)

    encoder.encode_sentence = encode
    encoder.encode_query = encode
    index = CorpusIndex(tmp_path / 'cache')
    index.build(CorpusConfig(str(source), max_sentences=1), encoder)
    hits = index.search('query', encoder, top_k=2)
    assert hits[0].text == 'Same meaning.'
    assert hits[0].late_score > hits[1].late_score


def test_nuance_components_retained_and_validated(tmp_path):
    source = tmp_path / 'a.txt'
    source.write_text('First sentence. Second sentence.', encoding='utf-8')
    encoder = FakeEncoder()
    index = CorpusIndex(tmp_path / 'cache')
    index.build(CorpusConfig(str(source), max_sentences=1), encoder)

    class DetailedReranker:
        def score_details(self, query, texts, cancel_event=None):
            return [dict(score=0.6, semantic_score=0.7, entailment_score=0.2,
                         contradiction_score=0.6) for text in texts]

        def score(self, *args, **kwargs):
            raise AssertionError('Detailed interface should take precedence')

    hits = index.search('query', encoder, reranker=DetailedReranker(), top_k=2)
    assert hits[0].score == hits[0].nuance_score == 0.6
    assert hits[0].semantic_score == 0.7
    assert hits[0].entailment_score == 0.2
    assert hits[0].contradiction_score == 0.6

    class InvalidDetails:
        def score_details(self, query, texts, **kwargs):
            return [dict(score=0.6, semantic_score=float('nan'),
                         entailment_score=0.2, contradiction_score=0.6) for text in texts]

    with pytest.raises(ValueError, match='invalid score components'):
        index.search('query', encoder, reranker=InvalidDetails())


def test_cleanup_deduplicates_with_raw_offsets_and_invalidates_options(tmp_path):
    source = tmp_path / 'sources'
    source.mkdir()
    wrapped = 'First\twrapped\n sentence.'
    (source / 'a.txt').write_text(wrapped, encoding='utf-8')
    (source / 'b.txt').write_text('First wrapped sentence.', encoding='utf-8')
    index = CorpusIndex(tmp_path / 'cache')
    encoder = FakeEncoder()
    config = CorpusConfig(str(source), max_sentences=1)
    manifest = index.build(config, encoder)
    assert manifest['candidates'] == 1
    assert manifest['cleanup']['files_changed'] == 1
    assert manifest['cleanup']['samples']
    hit = index.search('query', encoder)[0]
    assert hit.text == 'First wrapped sentence.'
    assert hit.provenance[0][1:] == (0, len(wrapped))
    with pytest.raises(ValueError, match='changed'):
        index.assert_current(CorpusConfig(str(source), max_sentences=1, cleanup='none'), encoder)
    assert (source / 'a.txt').read_text(encoding='utf-8') == wrapped


def test_long_sentence_error_reports_original_offset_after_cleanup(tmp_path):
    source = tmp_path / 'a.txt'
    original = '17\n\nOne two three four.'
    source.write_text(original, encoding='utf-8')
    encoder = FakeEncoder()
    encoder.config = SimpleNamespace(max_tokens=2, batch_size=2, device='cpu')
    index = CorpusIndex(tmp_path / 'cache')
    with pytest.raises(ValueError, match=f'offset {original.index("One")} '):
        index.build(CorpusConfig(str(source), cleanup='scanned_book'), encoder)


def test_diversity_changes_selection_using_casefolded_word_overlap(tmp_path):
    source = tmp_path / 'a.txt'
    source.write_text('Quiet rain falls. QUIET rain falls softly. Bright sunlight returns.', encoding='utf-8')
    encoder = FakeEncoder()
    index = CorpusIndex(tmp_path / 'cache')
    index.build(CorpusConfig(str(source), max_sentences=1), encoder)

    class Reranker:
        def score(self, query, texts, **kwargs):
            scores = {'Quiet rain falls.': .9, 'QUIET rain falls softly.': .85,
                      'Bright sunlight returns.': .8}
            return [scores[text] for text in texts]

    ordinary = index.search('query', encoder, top_k=2, reranker=Reranker())
    diverse = index.search('query', encoder, top_k=2, reranker=Reranker(), diversity=.2)
    assert [hit.text for hit in ordinary] == ['Quiet rain falls.', 'QUIET rain falls softly.']
    assert [hit.text for hit in diverse] == ['Quiet rain falls.', 'Bright sunlight returns.']
    assert [hit.score for hit in diverse] == pytest.approx([.9, .8])


def test_quality_exclusions_have_raw_audit_and_windows_never_bridge(tmp_path):
    import json
    import sqlite3

    source = tmp_path / 'a.txt'
    original = 'First\tvalid\n sentence. Broken \ufffd material. Last valid sentence. Another valid ending.'
    source.write_text(original, encoding='utf-8')
    encoder = FakeEncoder()
    encoded = []
    original_encode = encoder.encode_documents

    def record(texts):
        encoded.extend(texts)
        return original_encode(texts)

    encoder.encode_documents = record
    index = CorpusIndex(tmp_path / 'cache')
    result = index.build(CorpusConfig(str(source), max_sentences=2), encoder)
    assert result['quality']['analyzed_spans'] == 4
    assert result['quality']['excluded_spans'] == 1
    assert result['quality']['excluded_windows'] == 3
    assert result['quality']['excluded_reasons'] == {'replacement_character': 1}
    assert set(encoded) == {'First valid sentence.', 'Last valid sentence.', 'Another valid ending.',
                            'Last valid sentence. Another valid ending.'}
    sample = next(sample for sample in result['quality']['samples'] if sample['reasons'])
    assert original[sample['start']:sample['end']] == 'Broken \ufffd material.'
    assert sample['context_before'] == 'First valid sentence.'
    assert 'Last valid sentence.' in sample['context_after']
    with sqlite3.connect(index.path) as db:
        audit = db.execute('SELECT source,start,end,text,flags,reasons FROM excluded_spans').fetchone()
        assert audit[:4] == (str(source), sample['start'], sample['end'], sample['text'])
        assert 'replacement_character' in json.loads(audit[4])
        assert json.loads(audit[5]) == ['replacement_character']
        assert db.execute('SELECT COUNT(*) FROM excluded_spans').fetchone()[0] == 1
    assert source.read_text(encoding='utf-8') == original


def test_contextual_ocr_exclusion_does_not_poison_valid_duplicate(tmp_path):
    source = tmp_path / 'sources'
    source.mkdir()
    (source / 'a.txt').write_text('It 5 s the coldest and meanest in. the whole house.', encoding='utf-8')
    valid = source / 'b.txt'
    valid.write_text('the whole house.', encoding='utf-8')
    index = CorpusIndex(tmp_path / 'cache')
    encoder = FakeEncoder()
    result = index.build(CorpusConfig(str(source), max_sentences=2, cleanup='scanned_book'), encoder)
    assert result['quality']['excluded_reasons']['suspicious_boundary'] == 2
    assert result['quality']['excluded_windows'] == 3
    hit, = index.search('A different query.', encoder)
    assert hit.text == 'the whole house.'
    assert hit.provenance == ((str(valid), 0, len(hit.text)),)
    assert hit.quality_flags == ('short_reference',)
    assert hit.word_count == hit.content_token_count == 3


def test_all_excluded_failure_carries_bounded_report_and_preserves_cache(tmp_path):
    source = tmp_path / 'a.txt'
    source.write_text('A usable sentence.', encoding='utf-8')
    config = CorpusConfig(str(source), max_sentences=1)
    index = CorpusIndex(tmp_path / 'cache')
    encoder = FakeEncoder()
    before = index.build(config, encoder)
    before_bytes = index.path.read_bytes()
    source.write_text(' '.join(f'Damaged \ufffd sentence number {number}.' for number in range(12)), encoding='utf-8')
    with pytest.raises(CorpusQualityError, match='All complete corpus sentence spans') as error:
        index.build(config, encoder)
    assert error.value.quality_report['excluded_spans'] == 12
    assert len(error.value.quality_report['samples']) == 10
    assert encoder.encoded == 1
    assert index.path.read_bytes() == before_bytes
    assert index.manifest == before
    assert not list(index.cache_dir.glob('.building-*'))


def test_bad_overlimit_span_is_excluded_before_token_check(tmp_path):
    source = tmp_path / 'a.txt'
    source.write_text('Broken \ufffd material that is much too long. Still usable.', encoding='utf-8')
    encoder = FakeEncoder()
    encoder.config = SimpleNamespace(max_tokens=2, batch_size=2, device='cpu')
    index = CorpusIndex(tmp_path / 'cache')
    result = index.build(CorpusConfig(str(source), max_sentences=2), encoder)
    assert result['candidates'] == 1
    assert result['quality']['excluded_spans'] == 1
    assert index.search('query', encoder)[0].text == 'Still usable.'
    with pytest.raises(ValueError, match='exceeds embedding token limit'):
        index.build(CorpusConfig(str(source), quality_policy='off'), encoder)


def test_quality_policy_off_retains_flagged_text_and_invalidates_cache(tmp_path):
    source = tmp_path / 'a.txt'
    source.write_text('Broken \ufffd material. A usable sentence.', encoding='utf-8')
    encoder = FakeEncoder()
    index = CorpusIndex(tmp_path / 'cache')
    balanced = CorpusConfig(str(source), max_sentences=1)
    before = index.build(balanced, encoder)
    unfiltered = CorpusConfig(str(source), max_sentences=1, quality_policy='off')
    with pytest.raises(ValueError, match='changed'):
        index.assert_current(unfiltered, encoder)
    after = index.build(unfiltered, encoder)
    assert before['generation'] != after['generation']
    assert after['candidates'] == 2
    assert after['quality']['excluded_spans'] == 0
    assert after['quality']['flag_counts']['replacement_character'] == 1
    assert any('replacement_character' in sample['flags'] and not sample['reasons']
               for sample in after['quality']['samples'])
    flagged = next(hit for hit in index.search('query', encoder) if '\ufffd' in hit.text)
    assert 'replacement_character' in flagged.quality_flags
    with pytest.raises(ValueError, match='quality policy'):
        index.build(CorpusConfig(str(source), quality_policy='invented'), encoder)


def test_old_schema_rebuilds_normally_and_failure_keeps_old_file(tmp_path):
    import json
    import sqlite3

    source = tmp_path / 'a.txt'
    source.write_text('A usable sentence.', encoding='utf-8')
    index = CorpusIndex(tmp_path / 'cache')
    encoder = FakeEncoder()
    config = CorpusConfig(str(source), max_sentences=1)
    previous = index.build(config, encoder)
    previous['signature']['schema'] = 1
    previous['signature'].pop('quality_version')
    previous['signature']['config'].pop('quality_policy')
    with sqlite3.connect(index.path) as db:
        db.execute('UPDATE metadata SET value=?', (json.dumps(previous),))
    old_bytes = index.path.read_bytes()
    with pytest.raises(ValueError, match='older schema.*rebuild'):
        index.search('query', encoder)
    encoder.fail = True
    with pytest.raises(RuntimeError, match='unavailable'):
        index.build(config, encoder)
    assert index.path.read_bytes() == old_bytes
    encoder.fail = False
    result = index.build(config, encoder)
    assert result['signature']['schema'] == 2
    assert result['signature']['config']['quality_policy'] == 'balanced'
    assert result['quality']['analyzed_spans'] == 1


def test_content_length_filter_precedes_pool_and_excludes_special_token_overhead(tmp_path):
    source = tmp_path / 'a.txt'
    source.write_text('Certainly. A more substantial example.', encoding='utf-8')
    encoder = FakeEncoder()
    encoder.token_count = lambda text: len(text.split()) + 8
    index = CorpusIndex(tmp_path / 'cache')
    index.build(CorpusConfig(str(source), max_sentences=1), encoder)

    class Reranker:
        def score(self, query, texts, **kwargs):
            assert texts == ['A more substantial example.']
            return [0.7]

    hits = index.search('One two three four five six.', encoder, top_k=1,
                        reranker=Reranker(), rerank_pool=1, min_length_ratio=0.5)
    assert hits[0].text == 'A more substantial example.'
    assert hits[0].content_token_count == 4
    assert hits[0].token_count == 12
    # Legitimate short references remain available for short queries.
    assert index.search('Indeed.', encoder, top_k=1, min_length_ratio=0.5)[0].text == 'Certainly.'
    # Token mode uses the generator's exact lengths, ignoring this sentence-mode filter.
    native = index.search('One two three four five six.', encoder, top_k=1,
                          length_mode='tokens', min_length_ratio=1, token_tolerance=0,
                          token_counter=lambda text: 23 if text != 'A more substantial example.' else 24)
    assert native[0].text == 'Certainly.'
    assert native[0].token_count == 23


def test_semantic_threshold_filters_without_backfilling_and_preserves_components(tmp_path):
    source = tmp_path / 'a.txt'
    source.write_text('First sentence. Second sentence. Third sentence.', encoding='utf-8')
    encoder = FakeEncoder()
    index = CorpusIndex(tmp_path / 'cache')
    index.build(CorpusConfig(str(source), max_sentences=1), encoder)

    class Reranker:
        def score_details(self, query, texts, **kwargs):
            scores = {'First sentence.': (0.95, 0.1), 'Second sentence.': (0.6, 0.3),
                      'Third sentence.': (0.5, 0.8)}
            return [dict(score=scores[text][0], semantic_score=scores[text][1],
                         entailment_score=0.4, contradiction_score=0.2) for text in texts]

    messages = []
    hits = index.search('query', encoder, top_k=3, reranker=Reranker(), min_semantic_score=0.3,
                        progress=messages.append, diversity=0.5)
    assert [hit.text for hit in hits] == ['Second sentence.', 'Third sentence.']
    assert hits[0].score == hits[0].nuance_score == 0.6
    assert hits[0].semantic_score == 0.3
    assert 'Only 2 matching candidates' in messages[-1]
    with pytest.raises(ValueError, match='No retrieved references meet.*Reduce.*expand the corpus'):
        index.search('query', encoder, reranker=Reranker(), min_semantic_score=0.9)
    # Explicitly disabling acceptance returns the original ranking unchanged.
    assert index.search('query', encoder, reranker=Reranker())[0].text == 'First sentence.'


@pytest.mark.parametrize('reranker', [None, SimpleNamespace(score=lambda *args, **kwargs: [0.7])])
def test_semantic_threshold_requires_component_scores(tmp_path, reranker):
    with pytest.raises(ValueError, match='requires nuance reranking'):
        CorpusIndex(tmp_path).search('query', FakeEncoder(), reranker=reranker, min_semantic_score=0.3)


@pytest.mark.parametrize('option', ['min_length_ratio', 'min_semantic_score', 'max_contradiction_score'])
@pytest.mark.parametrize('value', [-0.1, 1.1, float('nan')])
def test_invalid_quality_retrieval_settings_rejected(tmp_path, option, value):
    with pytest.raises(ValueError, match='between zero and one'):
        CorpusIndex(tmp_path).search('query', FakeEncoder(), **{option: value})


@pytest.mark.parametrize('max_sentences,expected_error', [(1, 'assessed sentence span'), (2, 'considered sentence window')])
def test_quality_audit_budgets_include_excluded_spans_and_preserve_cache(tmp_path, monkeypatch, max_sentences, expected_error):
    from modules.sentence_rewrite import corpus

    source = tmp_path / 'a.txt'
    source.write_text('A usable sentence.', encoding='utf-8')
    encoder = FakeEncoder()
    index = CorpusIndex(tmp_path / 'cache')
    config = CorpusConfig(str(source), max_sentences=max_sentences)
    before = index.build(config, encoder)
    before_bytes = index.path.read_bytes()
    monkeypatch.setattr(corpus, 'MAX_WINDOWS', 3)
    source.write_text('Broken \ufffd first. Broken \ufffd second. Broken \ufffd third. Broken \ufffd fourth.', encoding='utf-8')
    with pytest.raises(CorpusQualityError, match=expected_error) as error:
        index.build(config, encoder)
    assert error.value.quality_report['analyzed_spans'] <= 3
    assert index.path.read_bytes() == before_bytes
    assert index.manifest == before
    assert encoder.encoded == 1
    assert not list(index.cache_dir.glob('.building-*'))


def test_contradiction_cutoff_blocks_wrong_roles_despite_high_semantic_score(tmp_path):
    source = tmp_path / 'a.txt'
    source.write_text('The dog chased the man. The man pursued the dog.', encoding='utf-8')
    encoder = FakeEncoder()
    index = CorpusIndex(tmp_path / 'cache')
    index.build(CorpusConfig(str(source), max_sentences=1), encoder)

    class Reranker:
        def score_details(self, query, texts, **kwargs):
            return [dict(score=0.9 if text.startswith('The dog') else 0.8, semantic_score=0.8,
                         entailment_score=0.5, contradiction_score=0.99 if text.startswith('The dog') else 0.8)
                    for text in texts]

    query = 'A man chased a dog.'
    default = index.search(query, encoder, reranker=Reranker(), min_semantic_score=0.3)
    assert default[0].text == 'The dog chased the man.'
    guarded = index.search(query, encoder, reranker=Reranker(), min_semantic_score=0.3, max_contradiction_score=0.8)
    assert [hit.text for hit in guarded] == ['The man pursued the dog.']
    assert guarded[0].score == guarded[0].semantic_score == guarded[0].contradiction_score == 0.8
    with pytest.raises(ValueError, match='No retrieved references meet.*maximum contradiction score'):
        index.search(query, encoder, reranker=Reranker(), max_contradiction_score=0.7)


@pytest.mark.parametrize('reranker', [None, SimpleNamespace(score=lambda *args, **kwargs: [0.7])])
def test_contradiction_threshold_requires_component_scores(tmp_path, reranker):
    with pytest.raises(ValueError, match='requires nuance reranking'):
        CorpusIndex(tmp_path).search('query', FakeEncoder(), reranker=reranker, max_contradiction_score=0.8)
