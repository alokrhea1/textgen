"""Controlled continuation/rewrite contracts without loading generation models."""

import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from modules.sentence_rewrite import automatic
from modules.sentence_rewrite.engine import RewriteProgress, _replacement_body


@pytest.fixture
def native(monkeypatch):
    import modules

    shared = ModuleType('modules.shared')
    shared.stop_everything = False
    shared.model, shared.tokenizer = object(), object()
    shared.bos_token, shared.eos_token = '<bos>', '<eos>'
    shared.is_seq2seq = False
    generation = ModuleType('modules.text_generation')
    generation.get_encoded_length = lambda text: len(text.split()) + 2
    generation.get_rewrite_prompt_length = lambda prompt, state: len(prompt)
    generation.get_max_prompt_length = lambda state: state['truncation_length'] - state['max_new_tokens']
    generation.drafts = [[' New draft. lookahead'], [' Another draft. lookahead']]
    generation.calls, generation.closed = [], []
    generation.hook = lambda text: text
    generation.terminal = False

    def generate(prompt, state, **kwargs):
        shared.stop_everything = False
        generation.calls.append((prompt, state, kwargs))
        index = len(generation.calls) - 1
        replies = generation.drafts[index]
        last = ''
        try:
            for last in replies:
                state['_notebook_auto_generation_info']['raw_reply'] = last
                if generation.terminal:
                    state['_notebook_auto_generation_info']['terminal'] = True
                    break
                if state['_notebook_auto_stop_predicate'](last):
                    state['_notebook_auto_generation_info']['boundary'] = True
                    break
                yield last
            yield generation.hook(last)
        finally:
            generation.closed.append(index)

    generation.generate_reply = generate
    models = ModuleType('modules.models')
    models.load_model_if_idle_unloaded = lambda: None
    utils = ModuleType('modules.utils')
    utils.check_model_loaded = lambda: (True, None)
    for name, module in [('shared', shared), ('text_generation', generation), ('models', models), ('utils', utils)]:
        monkeypatch.setitem(sys.modules, 'modules.' + name, module)
        monkeypatch.setattr(modules, name, module, raising=False)
    rewrites = []
    rewrite_closed = []
    replacement = {'text': 'Accepted replacement.'}

    def rewrite(document, hits, state, **kwargs):
        rewrites.append((document, hits, state, kwargs))
        try:
            yield RewriteProgress('Provisional fragment', 'Generating replacement')
            yield RewriteProgress(replacement['text'], 'Replacement ready', done=True)
        finally:
            rewrite_closed.append(True)

    monkeypatch.setattr(automatic, 'generate_rewrite', rewrite)
    return SimpleNamespace(shared=shared, generation=generation, models=models, utils=utils,
                           rewrites=rewrites, rewrite_closed=rewrite_closed, replacement=replacement)


def settings(**changes):
    value = dict(max_new_tokens=100, truncation_length=10000, auto_max_new_tokens=True, stream=False,
                 skip_special_tokens=True, temperature=.73, instruction_template_str='')
    value.update(changes)
    return value


def references(text='A reference sentence.'):
    return [SimpleNamespace(text=text, source='corpus.txt', start=0, end=len(text))]


def accepted(events):
    return [event for event in events if event.phase == 'accepted']


def selected_template(monkeypatch, tail=''):
    chat = ModuleType('modules.chat')
    captured = []

    class Template:
        def render(self, **kwargs):
            captured.append(kwargs)
            return '<template>' + kwargs['messages'][0]['content'] + tail

    chat.get_compiled_template = lambda template: Template()
    monkeypatch.setitem(sys.modules, 'modules.chat', chat)
    return captured


def test_selected_template_requests_only_continuation_and_keeps_actual_source(native, monkeypatch):
    captured = selected_template(monkeypatch)
    native.generation.drafts = [[' reluctantly stayed. more']]
    retrieved = []
    events = list(automatic.generate_automatic('Old sentence. She', settings(instruction_template_str='selected'),
                                             lambda text: retrieved.append(text) or references(), max_sentences=1))
    prompt, draft_state, _ = native.generation.calls[0]
    assert prompt.startswith('<template>')
    request = captured[0]['messages'][0]['content']
    assert 'The beginning of your answer has already been supplied' in request
    assert 'continue it with only the missing text' in request
    assert '[Exact unfinished sentence prefix]' not in request
    assert 'headings' in request
    assert request.endswith('[Notebook text]\nOld sentence. She')
    assert captured[0]['add_generation_prompt'] is True
    assert captured[0]['bos_token'] == '<bos>'
    assert retrieved == ['She reluctantly stayed.']
    assert '[Notebook' not in retrieved[0] and '<template>' not in events[-1].document
    assert events[-1].document == 'Old sentence. Accepted replacement.'
    assert draft_state['_rewrite_generation_guard'] is True
    assert draft_state['_rewrite_stop_predicate'] is None
    assert draft_state['_rewrite_prompt_kind'] == 'Automatic continuation'
    assert prompt.endswith('She')
    assert draft_state['_notebook_auto_prefill_suffix'] == 'She'


def test_template_thinking_flags_and_rendered_tail_drive_reasoning_extraction(native, monkeypatch):
    captured = selected_template(monkeypatch, tail='<think>\n')
    native.generation.drafts = [['Compare options. ', 'Compare options. </think>She reluctantly stayed. more']]
    events = list(automatic.generate_automatic('Old sentence. She',
                                             settings(instruction_template_str='selected', enable_thinking=False,
                                                      reasoning_effort='low', preserve_thinking=True),
                                             lambda _: references(), max_sentences=1))
    assert accepted(events)[0].target == 'She reluctantly stayed.'
    assert captured[0]['enable_thinking'] is False
    assert captured[0]['thinking'] is False
    assert captured[0]['thinking_budget'] == 0
    assert captured[0]['reasoning_effort'] == 'low'
    assert captured[0]['preserve_thinking'] is True


def test_unchecked_template_path_retains_raw_prompt(native, monkeypatch):
    captured = selected_template(monkeypatch)
    list(automatic.generate_automatic('Old sentence.', settings(instruction_template_str='selected'),
                                     lambda _: references(), use_template=False, max_sentences=1))
    assert native.generation.calls[0][0] == 'Old sentence.'
    assert '_rewrite_generation_guard' not in native.generation.calls[0][1]
    assert not captured


def test_prepare_continuation_keeps_unfinished_sentence_and_trims_old_context(native, monkeypatch):
    selected_template(monkeypatch)
    source = 'Earlier sentence. ' * 500 + ' An unfinished current sentence'
    minimal = automatic.prepare_continuation(' An unfinished current sentence', settings(instruction_template_str='selected'))
    options = settings(instruction_template_str='selected', truncation_length=minimal.token_count + 100 + 80)
    plan = automatic.prepare_continuation(source, options)
    assert plan.templated and plan.context_trimmed
    assert ' An unfinished current sentence' in plan.prompt
    assert plan.token_count <= options['truncation_length'] - options['max_new_tokens']
    assert 'Finish the notebook sentence' in plan.prompt and '[Notebook text]' in plan.prompt


def test_prepare_continuation_keeps_latest_complete_sentence(native, monkeypatch):
    selected_template(monkeypatch)
    source = 'Earlier sentence. ' * 500 + 'Last completed sentence.'
    minimal = automatic.prepare_continuation('Last completed sentence.', settings(instruction_template_str='selected'))
    options = settings(instruction_template_str='selected', truncation_length=minimal.token_count + 100 + 80)
    plan = automatic.prepare_continuation(source, options)
    assert plan.context_trimmed and 'Last completed sentence.\n' in plan.prompt
    assert plan.token_count <= options['truncation_length'] - options['max_new_tokens']


def test_template_minimal_request_cannot_fit_fails_before_native_generation(native, monkeypatch):
    selected_template(monkeypatch)
    with pytest.raises(ValueError, match='automatic continuation request and current sentence'):
        list(automatic.generate_automatic('An unfinished sentence',
                                         settings(instruction_template_str='selected', truncation_length=120),
                                         lambda _: references()))
    assert not native.generation.calls and not native.rewrites


def test_template_uses_native_generation_state_in_token_counter(native, monkeypatch):
    selected_template(monkeypatch)
    measured = []

    def count(prompt, state):
        measured.append(state)
        return len(prompt)

    native.generation.get_rewrite_prompt_length = count
    list(automatic.generate_automatic('Old sentence.', settings(instruction_template_str='selected', add_bos_token=False),
                                     lambda _: references(), max_sentences=1))
    assert measured and all(state['add_bos_token'] is False for state in measured)
    assert all(state['max_new_tokens'] == 100 for state in measured)
    assert all(state['_notebook_auto_generation'] is True for state in measured)


def test_templated_context_trim_survives_final_progress(native, monkeypatch):
    selected_template(monkeypatch)
    source = 'Earlier sentence. ' * 500 + 'Last completed sentence.'
    minimal = automatic.prepare_continuation('Last completed sentence.', settings(instruction_template_str='selected'))
    options = settings(instruction_template_str='selected', truncation_length=minimal.token_count + 100 + 80)
    events = list(automatic.generate_automatic(source, options, lambda _: references(), max_sentences=1))
    assert accepted(events)[0].context_trimmed and events[-1].context_trimmed
    assert events[-1].document.startswith(source)


def test_changed_exact_unfinished_prefix_fails_loudly_before_retrieval(native, monkeypatch):
    selected_template(monkeypatch, tail='<think>\n')
    native.generation.drafts = [['</think>He reluctantly stayed. more']]
    retrieved = []
    with pytest.raises(ValueError, match='exact unfinished Notebook sentence prefix'):
        list(automatic.generate_automatic('Old sentence. She ', settings(instruction_template_str='selected'),
                                         lambda text: retrieved.append(text) or references(), max_sentences=1))
    assert not retrieved and not native.rewrites and native.generation.closed == [0]


def test_partial_word_can_repeat_letters_without_echo_rejection(native, monkeypatch):
    selected_template(monkeypatch)
    native.generation.drafts = [['ha rang out. more']]
    events = list(automatic.generate_automatic('Old sentence. ha', settings(instruction_template_str='selected'),
                                             lambda _: references(), max_sentences=1))
    assert accepted(events)[0].target == 'haha rang out.'


def test_contract_inserts_word_boundary_from_full_completed_sentence(native, monkeypatch):
    selected_template(monkeypatch)
    native.generation.drafts = [[' the terms of the arrangement. more']]
    source = 'Old sentence. She reluctantly agreed to'
    events = list(automatic.generate_automatic(source, settings(instruction_template_str='selected'),
                                             lambda _: references(), max_sentences=1))
    assert accepted(events)[0].target == 'She reluctantly agreed to the terms of the arrangement.'
    assert 'tothe' not in accepted(events)[0].target
    assert native.generation.calls[0][0].endswith('She reluctantly agreed to')
    assert events[-1].document == 'Old sentence. Accepted replacement.'


def test_contract_completes_partial_word_without_inserting_a_space(native, monkeypatch):
    selected_template(monkeypatch)
    native.generation.drafts = [['lo world. more']]
    events = list(automatic.generate_automatic('Old sentence. hel', settings(instruction_template_str='selected'),
                                             lambda _: references(), max_sentences=1))
    assert accepted(events)[0].target == 'hello world.'
    assert native.rewrites[0][0] == 'Old sentence. hello world.'


def test_contract_waits_for_partial_streamed_prefix_before_retrieval(native, monkeypatch):
    selected_template(monkeypatch, tail='<think>\n')
    native.generation.drafts = [['</think>Sh', '</think>She reluctantly agreed', '</think>She reluctantly agreed to stay. more']]
    retrieved = []
    events = list(automatic.generate_automatic('Old sentence. She reluctantly agreed to',
                                             settings(instruction_template_str='selected'),
                                             lambda text: retrieved.append(text) or references(), max_sentences=1))
    assert retrieved == ['She reluctantly agreed to stay.']
    assert len([event for event in events if event.phase == 'drafting']) >= 3
    assert events[-1].done and native.generation.closed == [0]


def test_contract_preserves_original_leading_internal_and_trailing_spacing(native, monkeypatch):
    captured = selected_template(monkeypatch)
    source = '  Old sentence.\n\t She  agreed to  '
    native.generation.drafts = [['stay. more']]
    events = list(automatic.generate_automatic(source, settings(instruction_template_str='selected'),
                                             lambda _: references(), max_sentences=1))
    assert accepted(events)[0].target == 'She  agreed to  stay.'
    assert native.rewrites[0][0] == '  Old sentence.\n\t She  agreed to  stay.'
    assert events[-1].document == '  Old sentence.\n\t Accepted replacement.'
    assert native.generation.calls[0][0].endswith('She  agreed to  ')


@pytest.mark.parametrize('reply', ['agreed to stay. ', 'She agreed to stay. ', 'She  agreed to stay. '])
def test_contract_missing_or_normalized_typed_prefix_fails(native, monkeypatch, reply):
    selected_template(monkeypatch, tail='<think>\n')
    native.generation.drafts = [['</think>' + reply]]
    retrieved = []
    with pytest.raises(ValueError, match='exact unfinished Notebook sentence prefix'):
        list(automatic.generate_automatic('Old sentence. She  agreed to  ', settings(instruction_template_str='selected'),
                                         lambda text: retrieved.append(text) or references(), max_sentences=1))
    assert not retrieved and not native.rewrites and native.generation.closed == [0]


def test_contract_output_hook_can_repair_divergent_streamed_prefix(native, monkeypatch):
    selected_template(monkeypatch, tail='<think>\n')
    native.generation.drafts = [['</think>A wrong raw answer. ']]
    native.generation.hook = lambda _: '</think>She reluctantly agreed to stay. '
    events = list(automatic.generate_automatic('Old sentence. She reluctantly agreed to',
                                             settings(instruction_template_str='selected'),
                                             lambda _: references(), max_sentences=1))
    assert accepted(events)[0].target == 'She reluctantly agreed to stay.'
    assert native.generation.closed == [0]


def test_contract_output_hook_breaking_prefix_fails_after_native_closure(native, monkeypatch):
    selected_template(monkeypatch, tail='<think>\n')
    native.generation.drafts = [['</think>She reluctantly agreed to stay. more']]
    native.generation.hook = lambda _: '</think>He reluctantly agreed to stay. '
    retrieved = []
    with pytest.raises(ValueError, match='exact unfinished Notebook sentence prefix'):
        list(automatic.generate_automatic('Old sentence. She reluctantly agreed to',
                                         settings(instruction_template_str='selected'),
                                         lambda text: retrieved.append(text) or references(), max_sentences=1))
    assert not retrieved and native.generation.closed == [0]


def test_contract_copies_prefix_tokens_against_shared_draft_allowance(native, monkeypatch):
    selected_template(monkeypatch, tail='<think>\n')
    native.generation.drafts = [['</think>She reluctantly agreed to stay. '], ['</think>Second sentence. ']]
    list(automatic.generate_automatic('Old sentence. She reluctantly agreed to',
                                     settings(instruction_template_str='selected', max_new_tokens=10),
                                     lambda _: references(), max_sentences=2))
    assert native.generation.calls[1][1]['max_new_tokens'] == 5


def test_contract_plan_retains_exact_unfinished_prefix_in_minimal_context(native, monkeypatch):
    selected_template(monkeypatch)
    source = 'Earlier sentence. ' * 500 + '\n She  agreed to  '
    minimum = automatic.prepare_continuation('\n She  agreed to  ', settings(instruction_template_str='selected'))
    plan = automatic.prepare_continuation(source, settings(instruction_template_str='selected',
                                                          truncation_length=minimum.token_count + 100 + 20))
    assert plan.context_trimmed and plan.prefilled_prefix == 'She  agreed to  '
    assert plan.response_prefix is None
    assert plan.prompt.endswith('She  agreed to  ')
    assert '\n She  agreed to  ' in plan.prompt


def test_prefill_is_counted_in_native_prompt_budget(native, monkeypatch):
    selected_template(monkeypatch)
    source = 'Old sentence. She reluctantly agreed to'
    plan = automatic.prepare_continuation(source, settings(instruction_template_str='selected'))
    assert plan.prefilled_prefix == 'She reluctantly agreed to'
    assert plan.response_prefix is None and plan.reasoning_prefix == ''
    assert plan.prompt.endswith(plan.prefilled_prefix)
    assert plan.token_count == len(plan.prompt)
    budget_without_prefill = plan.token_count - len(plan.prefilled_prefix)
    # Even with old context removed, the mandatory prefill cannot be discarded
    # to satisfy a smaller prompt budget.
    minimum = automatic.prepare_continuation(plan.prefilled_prefix, settings(instruction_template_str='selected'))
    with pytest.raises(ValueError, match='automatic continuation request and current sentence'):
        automatic.prepare_continuation(source, settings(instruction_template_str='selected',
                                                       truncation_length=100 + minimum.token_count - 1))
    assert budget_without_prefill < plan.token_count


def test_active_reasoning_template_retains_full_copy_fallback(native, monkeypatch):
    selected_template(monkeypatch, tail='<think>\n')
    plan = automatic.prepare_continuation('Old sentence. She reluctantly agreed to',
                                         settings(instruction_template_str='selected'))
    assert plan.response_prefix == 'She reluctantly agreed to'
    assert plan.prefilled_prefix == '' and plan.reasoning_prefix == '<think>'
    assert plan.prompt.endswith('<think>\n')
    assert '[Exact unfinished sentence prefix]\nShe reluctantly agreed to\n' in plan.prompt


def test_typed_reasoning_control_uses_full_copy_without_prefill(native, monkeypatch):
    selected_template(monkeypatch)
    plan = automatic.prepare_continuation('Old sentence. <think>\n', settings(instruction_template_str='selected'))
    assert plan.response_prefix == '<think>\n'
    assert plan.prefilled_prefix == ''
    assert not plan.prompt.endswith('<think>\n')


def test_ordinary_html_prefix_can_be_prefilled(native, monkeypatch):
    selected_template(monkeypatch)
    plan = automatic.prepare_continuation('Old sentence. A <em>clear</em> example',
                                         settings(instruction_template_str='selected'))
    assert plan.prefilled_prefix == 'A <em>clear</em> example'
    assert plan.response_prefix is None


def test_prefill_suffix_is_literal_after_final_output_hook(native, monkeypatch):
    selected_template(monkeypatch)
    native.generation.drafts = [[' a rough finish. more']]
    native.generation.hook = lambda _: ' a final finish. '
    events = list(automatic.generate_automatic('Old sentence. She described', settings(instruction_template_str='selected'),
                                             lambda _: references(), max_sentences=1))
    assert accepted(events)[0].target == 'She described a final finish.'
    assert native.generation.closed == [0]


def test_prefill_does_not_strip_quotes_or_infer_spaces(native, monkeypatch):
    selected_template(monkeypatch)
    native.generation.drafts = [['"go home." more']]
    events = list(automatic.generate_automatic('Old sentence. She said ', settings(instruction_template_str='selected'),
                                             lambda _: references(), max_sentences=1))
    assert accepted(events)[0].target == 'She said "go home."'


def test_final_multiword_prefill_echo_refuses_after_native_closure(native, monkeypatch):
    selected_template(monkeypatch)
    native.generation.drafts = [['She reluctantly agreed to stay. more']]
    retrieved = []
    with pytest.raises(ValueError, match='repeated the supplied unfinished Notebook prefix'):
        list(automatic.generate_automatic('Old sentence. She reluctantly agreed to', settings(instruction_template_str='selected'),
                                         lambda text: retrieved.append(text) or references(), max_sentences=1))
    assert not retrieved and native.generation.closed == [0]


def test_final_output_hook_can_repair_prefill_echo(native, monkeypatch):
    selected_template(monkeypatch)
    native.generation.drafts = [['She reluctantly agreed to stay. more']]
    native.generation.hook = lambda _: ' stay. '
    events = list(automatic.generate_automatic('Old sentence. She reluctantly agreed to', settings(instruction_template_str='selected'),
                                             lambda _: references(), max_sentences=1))
    assert accepted(events)[0].target == 'She reluctantly agreed to stay.'


@pytest.mark.parametrize('marker', ['', '<eos>'])
def test_templated_sentence_eos_ends_pass_but_allows_next_sentence(native, monkeypatch, marker):
    selected_template(monkeypatch)
    native.generation.drafts = [[f' First draft.{marker}'], [f' Second draft.{marker}']]
    events = list(automatic.generate_automatic('Old sentence.', settings(instruction_template_str='selected'),
                                             lambda _: references(), max_sentences=2))
    assert len(native.generation.calls) == 2
    assert events[-1].done and events[-1].completed == 2
    assert [event.target for event in accepted(events)] == ['First draft.', 'Second draft.']


def test_templated_custom_stop_still_ends_whole_operation(native, monkeypatch):
    selected_template(monkeypatch)
    native.generation.terminal = True
    native.generation.drafts = [[' First draft. ']]
    events = list(automatic.generate_automatic('Old sentence.', settings(instruction_template_str='selected'),
                                             lambda _: references()))
    assert len(native.generation.calls) == 1 and events[-1].completed == 1


def test_templated_incomplete_eos_preserves_prior_acceptance_and_fails(native, monkeypatch):
    selected_template(monkeypatch)
    native.generation.drafts = [[' First draft.'], [' incomplete']]
    events = []
    with pytest.raises(ValueError, match='new complete'):
        for event in automatic.generate_automatic('Old sentence.', settings(instruction_template_str='selected'), lambda _: references()):
            events.append(event)
    assert accepted(events)[-1].document == 'Old sentence. Accepted replacement.'
    assert events[-1].document == accepted(events)[-1].document
    assert len(native.generation.closed) == 2


def test_missing_template_keeps_raw_natural_eos_semantics(native):
    native.generation.drafts = [[' First draft.']]
    events = list(automatic.generate_automatic('Old sentence.', settings(), lambda _: references(), use_template=True))
    assert len(native.generation.calls) == 1 and events[-1].completed == 1


def test_multiple_sentences_retrieve_each_and_only_commit_replacements(native):
    retrieved = []

    def retrieve(sentence):
        assert len(native.generation.closed) == len(native.generation.calls)
        retrieved.append(sentence)
        return references()

    events = list(automatic.generate_automatic('Original sentence.', settings(), retrieve, max_sentences=2))
    assert retrieved == ['New draft.', 'Another draft.']
    assert [event.document for event in accepted(events)] == [
        'Original sentence. Accepted replacement.',
        'Original sentence. Accepted replacement. Accepted replacement.',
    ]
    assert events[-1].done and events[-1].completed == 2
    assert not any('draft' in event.document or 'lookahead' in event.document or 'Provisional' in event.document for event in events)
    assert len(native.rewrite_closed) == 2


def test_draft_chunk_lookahead_is_discarded_and_charged(native):
    native.generation.drafts = [[' New draft. Discarded second sentence. More words.'], [' Next draft. extra']]
    events = list(automatic.generate_automatic('Old sentence.', settings(max_new_tokens=12), lambda _: references(), max_sentences=2))
    assert accepted(events)[0].target == 'New draft.'
    assert native.generation.calls[1][1]['max_new_tokens'] == 5
    assert all(call[2]['max_new_tokens'] == 12 for call in native.rewrites)
    assert 'Discarded' not in events[-1].document


def test_budget_exhaustion_stops_without_another_draft(native):
    native.generation.drafts = [[' First sentence. discarded lookahead words']]
    events = list(automatic.generate_automatic('', settings(max_new_tokens=1), lambda _: references()))
    assert len(native.generation.calls) == 1
    assert events[-1].done and events[-1].completed == 1


def test_minimum_one_token_charge_prevents_infinite_empty_tokenizer(native):
    native.generation.get_encoded_length = lambda _: 2
    events = list(automatic.generate_automatic('', settings(max_new_tokens=2), lambda _: references(), max_sentences=10))
    assert len(native.generation.calls) == 2
    assert events[-1].completed == 2


def test_unfinished_source_sentence_completed_then_whole_span_rewritten(native):
    native.generation.drafts = [[' reluctantly agreed. lookahead']]
    events = list(automatic.generate_automatic('  Old sentence.\n She', settings(), lambda _: references(), max_sentences=1))
    assert accepted(events)[0].target == 'She reluctantly agreed.'
    assert events[-1].document == '  Old sentence.\n Accepted replacement.'


def test_unfinished_word_continuation_not_split(native):
    native.generation.drafts = [['lo world. lookahead']]
    events = list(automatic.generate_automatic('Old sentence. hel', settings(), lambda _: references(), max_sentences=1))
    assert accepted(events)[0].target == 'hello world.'
    assert events[-1].document == 'Old sentence. Accepted replacement.'


def test_completed_source_without_backend_spacing_gets_separator(native):
    native.generation.drafts = [['New draft. lookahead']]
    events = list(automatic.generate_automatic('Old sentence.', settings(), lambda _: references(), max_sentences=1))
    assert accepted(events)[0].target == 'New draft.'
    assert events[-1].document == 'Old sentence. Accepted replacement.'


def test_abbreviation_and_old_sentences_not_used_as_boundaries(native):
    native.generation.drafts = [[' Mr. Smith', ' Mr. Smith stayed. more']]
    retrieved = []
    events = list(automatic.generate_automatic('A previous sentence. Another previous sentence. ', settings(),
                                             lambda sentence: retrieved.append(sentence) or references(), max_sentences=1))
    assert retrieved == ['Mr. Smith stayed.']
    assert events[-1].document == 'A previous sentence. Another previous sentence.  Accepted replacement.'


@pytest.mark.parametrize('reply', ['', ' incomplete', ' Dr.'])
def test_no_complete_new_sentence_fails_without_retrieval(native, reply):
    native.generation.drafts = [[reply]]
    retrieved = []
    with pytest.raises(ValueError, match='new complete'):
        list(automatic.generate_automatic('An older complete sentence. ', settings(),
                                        lambda text: retrieved.append(text) or references()))
    assert not retrieved and not native.rewrites
    assert native.generation.closed == [0]


def test_natural_eos_at_unconfirmed_period_rewrites_once_then_finishes(native):
    native.generation.drafts = [[' Final draft.']]
    events = list(automatic.generate_automatic('Old sentence.', settings(), lambda _: references()))
    assert events[-1].done and events[-1].completed == 1
    assert len(native.generation.calls) == 1


def test_custom_stop_with_complete_sentence_is_terminal(native):
    native.generation.terminal = True
    native.generation.drafts = [[' Final draft. ']]
    events = list(automatic.generate_automatic('Old sentence.', settings(), lambda _: references()))
    assert events[-1].done and len(native.generation.calls) == 1


def test_explicit_eos_marker_is_terminal_even_with_confirmed_boundary(native):
    native.generation.drafts = [[' Final draft. <eos>']]
    events = list(automatic.generate_automatic('', settings(), lambda _: references()))
    assert len(native.generation.calls) == 1 and events[-1].done


def test_output_extension_is_consumed_and_revalidated(native):
    native.generation.hook = lambda _: ' Altered final sentence. lookahead'
    retrieved = []
    events = list(automatic.generate_automatic('', settings(), lambda text: retrieved.append(text) or references(), max_sentences=1))
    assert retrieved == ['Altered final sentence.']
    assert accepted(events)[0].target == retrieved[0]
    assert native.generation.closed == [0]


def test_output_extension_removing_completion_fails_before_retrieval(native):
    native.generation.hook = lambda _: ' incomplete final'
    retrieved = []
    with pytest.raises(ValueError, match='new complete'):
        list(automatic.generate_automatic('', settings(), lambda text: retrieved.append(text) or references()))
    assert not retrieved and native.generation.closed == [0]


def test_output_extension_cannot_hide_raw_lookahead_budget(native):
    native.generation.drafts = [[' New draft. discarded words words words words words']]
    native.generation.hook = lambda _: ' Final sentence. '
    events = list(automatic.generate_automatic('', settings(max_new_tokens=3), lambda _: references()))
    assert len(native.generation.calls) == 1 and events[-1].completed == 1


def test_reasoning_and_html_extraction(native):
    native.generation.drafts = [['<think>Consider several sentences.</think>A <em>clear</em> draft. more']]
    events = list(automatic.generate_automatic('Old sentence.', settings(), lambda _: references(), max_sentences=1))
    assert accepted(events)[0].target == 'A <em>clear</em> draft.'
    assert '<think>' not in events[-1].document


def test_reasoning_extraction_keeps_unfinished_prefix_spacing(native):
    native.generation.drafts = [['<think>Options.</think> reluctantly stayed. more']]
    events = list(automatic.generate_automatic('Old sentence. She', settings(), lambda _: references(), max_sentences=1))
    assert accepted(events)[0].target == 'She reluctantly stayed.'


def test_prompt_tail_reasoning_marker_seeds_extractor(native):
    native.generation.drafts = [['Several options. ', 'Several options. </think>New draft. more']]
    events = list(automatic.generate_automatic('Old sentence. <think>\n', settings(), lambda _: references(), max_sentences=1))
    assert accepted(events)[0].target == '<think>\nNew draft.'
    assert 'Several options' not in events[-1].document


def test_preserve_leading_helper_is_compatible_for_manual_rewrite(native):
    assert _replacement_body('  A sentence. ') == 'A sentence. '
    assert _replacement_body('  A sentence. ', preserve_leading=True) == '  A sentence. '
    assert _replacement_body('<|im_start|>assistant\n A sentence.<|im_end|>', preserve_leading=True).startswith('\n A sentence.')
    assert _replacement_body('<think>Options.</think>  A sentence.', preserve_leading=True) == '  A sentence.'


def test_sampling_and_caller_state_preserved(native):
    cancel = threading.Event()
    settings_value = settings(stop_event=cancel, keep={'value': 1})
    original = dict(settings_value)
    list(automatic.generate_automatic('Old sentence.', settings_value, lambda _: references(), guidance='Use precise verbs.',
                                     use_template=False, max_sentences=1))
    assert settings_value == original
    draft_state = native.generation.calls[0][1]
    assert draft_state['temperature'] == .73
    assert draft_state['stream'] and not draft_state['auto_max_new_tokens']
    assert not draft_state['skip_special_tokens']
    assert draft_state['_notebook_auto_generation'] is True
    assert '_rewrite_generation_guard' not in draft_state
    assert native.rewrites[0][2]['max_new_tokens'] == 100
    assert native.rewrites[0][3]['guidance'] == 'Use precise verbs.'
    assert native.rewrites[0][3]['use_template'] is False
    assert draft_state['stop_event'] is native.rewrites[0][2]['stop_event'] is cancel


def test_context_trim_warning_survives_accepted_and_final_events(native, monkeypatch):
    def rewrite(*args, **kwargs):
        yield RewriteProgress('Accepted replacement.', 'Replacement ready', done=True,
                              plan=SimpleNamespace(context_trimmed=True))

    monkeypatch.setattr(automatic, 'generate_rewrite', rewrite)
    events = list(automatic.generate_automatic('Old sentence.', settings(), lambda _: references(), max_sentences=1))
    assert accepted(events)[0].context_trimmed
    assert events[-1].context_trimmed


@pytest.mark.parametrize('phase', ['drafting', 'retrieving', 'rewriting', 'accepted'])
def test_cancellation_on_every_phase_preserves_accepted_only(native, phase):
    cancel = threading.Event()
    iterator = automatic.generate_automatic('Old sentence.', settings(), lambda _: references(), cancel_event=cancel)
    seen = []
    with pytest.raises(InterruptedError):
        for progress in iterator:
            seen.append(progress)
            if progress.phase == phase:
                cancel.set()
    if phase != 'accepted':
        assert all(event.document == 'Old sentence.' for event in seen)
    else:
        assert seen[-1].document == 'Old sentence. Accepted replacement.'
    assert len(native.generation.closed) == len(native.generation.calls)
    assert len(native.rewrite_closed) == len(native.rewrites)


def test_cancellation_during_retrieval_stops_before_rewrite(native):
    cancel = threading.Event()

    def retrieve(_):
        cancel.set()
        return references()

    with pytest.raises(InterruptedError):
        list(automatic.generate_automatic('', settings(), retrieve, cancel_event=cancel))
    assert not native.rewrites and native.generation.closed == [0]


def test_cancellation_before_start_does_not_load_or_generate(native):
    cancel = threading.Event()
    cancel.set()
    native.models.load_model_if_idle_unloaded = lambda: pytest.fail('Loaded after cancellation')
    with pytest.raises(InterruptedError):
        list(automatic.generate_automatic('', settings(), lambda _: references(), cancel_event=cancel))
    assert not native.generation.calls


def test_fresh_event_can_retry_after_stale_global_stop(native):
    native.shared.stop_everything = True
    events = list(automatic.generate_automatic('', settings(), lambda _: references(), cancel_event=threading.Event(), max_sentences=1))
    assert events[-1].done


def test_global_stop_after_start_is_respected(native):
    iterator = automatic.generate_automatic('', settings(), lambda _: references(), cancel_event=threading.Event())
    next(iterator)
    next(iterator)
    native.shared.stop_everything = True
    with pytest.raises(InterruptedError):
        next(iterator)
    assert native.generation.closed == [0]


@pytest.mark.parametrize('attribute', ['model', 'tokenizer'])
@pytest.mark.parametrize('phase', ['drafting', 'retrieving', 'rewriting', 'accepted'])
def test_model_or_tokenizer_change_in_any_phase_fails_without_committing_draft(native, attribute, phase):
    iterator = automatic.generate_automatic('Old sentence.', settings(), lambda _: references())
    seen = []
    with pytest.raises(ValueError, match='model or tokenizer changed'):
        for progress in iterator:
            seen.append(progress)
            if progress.phase == phase:
                setattr(native.shared, attribute, object())
    if phase != 'accepted':
        assert all(event.document == 'Old sentence.' for event in seen)
    assert len(native.generation.closed) == len(native.generation.calls)


def test_empty_retrieval_fails_loudly_and_keeps_prior_acceptance(native):
    calls = 0

    def retrieve(_):
        nonlocal calls
        calls += 1
        return references() if calls == 1 else []

    events = []
    with pytest.raises(ValueError, match='No qualifying references'):
        for event in automatic.generate_automatic('Old sentence.', settings(), retrieve):
            events.append(event)
    assert accepted(events)[-1].document == 'Old sentence. Accepted replacement.'
    assert events[-1].document == accepted(events)[-1].document
    assert len(native.rewrites) == 1


def test_native_exception_propagates_and_closes(native):
    def generate(*args, **kwargs):
        try:
            raise RuntimeError('backend failed')
            yield ''
        finally:
            native.generation.closed.append(0)

    native.generation.generate_reply = generate
    with pytest.raises(RuntimeError, match='backend failed'):
        list(automatic.generate_automatic('', settings(), lambda _: references()))
    assert native.generation.closed == [0]


def test_closing_controller_closes_active_native(native):
    native.generation.drafts = [[' unfinished', ' New draft. more']]
    iterator = automatic.generate_automatic('', settings(), lambda _: references())
    next(iterator)
    next(iterator)
    iterator.close()
    assert native.generation.closed == [0]


def test_closing_controller_closes_active_rewrite(native):
    iterator = automatic.generate_automatic('', settings(), lambda _: references())
    for event in iterator:
        if event.phase == 'rewriting':
            break
    iterator.close()
    assert native.rewrite_closed == [True]


@pytest.mark.parametrize('value', [0, -1, None, float('inf'), 1.5, True])
def test_finite_positive_draft_budget_required(native, value):
    with pytest.raises(ValueError, match='positive finite'):
        list(automatic.generate_automatic('', settings(max_new_tokens=value), lambda _: references()))
    assert not native.generation.calls


@pytest.mark.parametrize('value', [0, -1, None, 1.5, True])
def test_finite_positive_sentence_limit_required(native, value):
    with pytest.raises(ValueError, match='positive sentence limit'):
        list(automatic.generate_automatic('', settings(), lambda _: references(), max_sentences=value))


def test_seq2seq_rejected_before_generation(native):
    native.shared.is_seq2seq = True
    with pytest.raises(ValueError, match='decoder-only'):
        list(automatic.generate_automatic('', settings(), lambda _: references()))
    assert not native.generation.calls


def test_missing_model_rejected_before_generation(native):
    native.utils.check_model_loaded = lambda: (False, 'No model loaded')
    with pytest.raises(ValueError, match='No model loaded'):
        list(automatic.generate_automatic('', settings(), lambda _: references()))
    assert not native.generation.calls
