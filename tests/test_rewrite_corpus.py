from types import SimpleNamespace

import numpy as np
import pytest

from modules.sentence_rewrite.corpus import CorpusConfig, CorpusIndex


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
