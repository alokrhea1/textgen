"""Opt-in automatic-rewrite workflow against an already running scratch server.

Client and server must share the corpus filesystem. This changes Notebook text,
sampling settings, and layout. Use the shipped synthetic fixture, a scratch user
data directory, and a loaded decoder-only model. This verifies wiring and safety,
not literary retrieval quality; later runs explicitly relax semantic acceptance.
Run with Playwright and Chromium installed. The lead agent alone runs this test.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


AUTO = 'Automatically retrieve and rewrite generated sentences'
LIMIT = 'number input for Maximum sentences per automatic generation'
PREFIX = 'She reluctantly agreed to'
INVALID_SEED = 'This manual seed is deliberately incomplete'
PREFIX_QUERY_RUNS = frozenset({
    'later_generate', 'regenerate', 'stop_retry', 'single_native_submit',
    'two_column_generate', 'two_column_native_submit', 'two_column_continue',
})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:7860')
    parser.add_argument('--corpus', type=Path, default=Path(__file__).resolve().parents[1] / 'fixtures' / 'rewrite_reference_sentences.txt')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--output-dir', type=Path, default=Path('rewrite-automatic-browser-artifacts'))
    args = parser.parse_args()
    args.corpus = args.corpus.expanduser().resolve(strict=True)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        parser.error('--output-dir must be empty; do not overwrite previous evidence')
    results = {'url': args.url, 'corpus': str(args.corpus), 'device': args.device,
               'corpus_sha256': hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
               'generation_settings': {'seed': args.seed, 'temperature': args.temperature,
                                       'max_new_tokens': 150, 'auto_max_new_tokens': False,
                                       'stream': True, 'thinking': False},
               'textbox_submit_key': 'Shift+Enter (native Gradio multiline submission)',
               'runs': {}}
    page = None
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, args=['--no-sandbox'])
            page = browser.new_page(viewport={'width': 1500, 'height': 1100})
            page.goto(args.url, wait_until='domcontentloaded')

            def tab(name):
                page.get_by_role('tab', name=name, exact=True).click()

            def number(name, value):
                item = page.get_by_role('spinbutton', name=name, exact=True)
                item.fill(str(value))
                item.press('Tab')

            def parameters(rate=0):
                tab('Parameters')
                number('Seed (-1 for random)', args.seed)
                number('number input for temperature', args.temperature)
                number('number input for max_new_tokens', 150)
                number('number input for Maximum tokens/second', rate)
                page.get_by_role('checkbox', name='auto_max_new_tokens', exact=True).uncheck()
                page.get_by_role('checkbox', name='Activate text streaming', exact=True).check()
                tab('Notebook')
                tab('Rewrite')

            def layout(two_columns):
                tab('Session')
                page.get_by_role('checkbox', name='Show two columns in the Notebook tab', exact=True).set_checked(two_columns)
                tab('Notebook')
                tab('Rewrite')

            layout(False)
            parameters()
            auto = page.get_by_role('checkbox', name=AUTO, exact=True)
            status = page.get_by_role('textbox', name='Status', exact=True)
            preview = page.get_by_role('textbox', name=re.compile(r'^Automatic generation preview'))
            evidence = page.get_by_role('textbox', name='Retrieved references and sources', exact=True)
            seed = page.get_by_role('textbox', name=re.compile(r'^Optional seed sentence'))
            review = page.get_by_role('checkbox', name='Review before applying', exact=True)
            generate = page.get_by_role('button', name='Generate', exact=True)
            raw = page.locator('#textbox-notebook textarea')

            def set_source(source, two_columns=False, output=''):
                tab('Raw')
                if two_columns:
                    page.get_by_role('textbox', name='Input', exact=True).fill(source)
                    page.locator('#textbox-default textarea').fill(output)
                else:
                    raw.fill(source)
                tab('Rewrite')

            def ready():
                expect(generate).to_be_visible(timeout=240000)
                expect(generate).to_be_enabled(timeout=10000)
                expect(auto).to_be_enabled(timeout=10000)

            def start(button='Generate', unchanged=PREFIX, two_columns=False):
                if button == 'Shift+Enter':
                    tab('Raw')
                    entry = page.get_by_role('textbox', name='Input', exact=True) if two_columns else raw
                    entry.press('Shift+Enter')
                    tab('Rewrite')
                else:
                    page.get_by_role('button', name=button, exact=True).click()
                expect(auto).to_be_disabled(timeout=15000)
                expect(page.get_by_role('spinbutton', name=LIMIT, exact=True)).to_be_disabled()
                text = page.locator('#textbox-default textarea') if two_columns else raw
                expect(text).to_have_value(unchanged)
                expect(text).to_be_disabled()
                if two_columns:
                    expect(page.get_by_role('textbox', name='Input', exact=True)).to_be_disabled()
                    expect(page.get_by_role('button', name='Continue', exact=True)).to_be_disabled()
                else:
                    expect(page.get_by_role('button', name='Regenerate', exact=True)).to_be_disabled()

            def record(name, maximum, two_columns=False, allow_partial_failure=False, require_count=None):
                ready()
                final_status = status.input_value()
                applied = re.search(r'Applied (\d+) completed rewrite\(s\)', final_status)
                assert applied, final_status
                count = int(applied.group(1))
                partial_failure = final_status.startswith('AUTOMATIC GENERATION FAILED:')
                text = page.locator('#textbox-default textarea') if two_columns else raw
                after = text.input_value()
                references = evidence.input_value()
                replacement = page.get_by_role('textbox', name='Proposed replacement', exact=True).input_value().strip()
                query = page.get_by_role('textbox', name='Sentence used for retrieval', exact=True).input_value()
                # Save observations before assertions so a one-sentence EOS or
                # draft-budget stop remains available for lead investigation.
                results['runs'][name] = {'status': final_status, 'accepted_sentences': count,
                                         'after': after, 'preview': preview.input_value(),
                                         'query': query, 'references': references,
                                         'replacement': replacement, 'partial_failure': partial_failure,
                                         'required_count': require_count}
                assert 1 <= count <= maximum, final_status
                if require_count is not None:
                    assert count == require_count, final_status
                if partial_failure:
                    assert allow_partial_failure, final_status
                    assert re.match(r'^AUTOMATIC GENERATION FAILED: No (?:corpus candidates|retrieved references)', final_status), final_status
                else:
                    assert final_status.startswith('Automatic generation complete:'), final_status
                assert after and re.search(r'\.[\s\'"”’)]*$', after), after
                assert preview.input_value() == after
                if not partial_failure:
                    assert str(args.corpus) in references, references
                    assert replacement and replacement in after, (replacement, after)
                if name in PREFIX_QUERY_RUNS:
                    # PREFIX ends with a complete word. The full-prefix draft
                    # contract must retain the separator before its continuation.
                    assert re.match(re.escape(PREFIX) + r'\s', query), query
                expect(auto).to_be_checked()
                expect(review).to_be_checked()
                expect(seed).to_have_value(INVALID_SEED)
                expect(page.get_by_role('button', name='Apply rewrite', exact=True)).to_be_disabled()
                results['runs'][name]['passed'] = True
                print(f'{name}: {count} completed rewrites', flush=True)
                return after

            def build():
                page.locator('span:visible').filter(has_text=re.compile(r'^Corpus and embedding settings$')).click()
                device = page.get_by_role('listbox', name='Embedding device', exact=True)
                device.fill(args.device)
                device.press('Enter')
                quality = page.get_by_role('listbox', name='Ingestion quality screening', exact=True)
                quality.fill('balanced')
                quality.press('Enter')
                page.get_by_role('textbox', name=re.compile(r'^Local .txt files or directories')).fill(str(args.corpus))
                page.get_by_role('button', name='Build corpus / retry', exact=True).click()
                expect(status).to_have_value(re.compile(r'^Corpus ready:'), timeout=120000)
                assert 'balanced' in page.get_by_role('textbox', name='Ingestion quality report', exact=True).input_value()
                page.get_by_role('checkbox', name='Enable model thinking for this rewrite', exact=True).uncheck()
                page.get_by_role('checkbox', name='Rerank for meaning and nuance', exact=True).check()
                seed.fill(INVALID_SEED)
                review.check()
                auto.check()

            # A checked box must never bypass a missing index by generating normally.
            auto.check()
            number(LIMIT, 2)
            set_source(PREFIX)
            generate.click()
            expect(status).to_have_value(re.compile(r'^AUTOMATIC GENERATION FAILED: Build the corpus first\.'), timeout=30000)
            ready()
            expect(raw).to_have_value(PREFIX)
            expect(evidence).to_have_value('')
            results['no_index_no_generation'] = status.input_value()
            build()

            # First run uses shipped acceptance defaults. A legitimate refusal is
            # recorded and retried with permissive settings for routing coverage.
            number('number input for Minimum reference length relative to target', 0.5)
            number('number input for Minimum semantic similarity', 0.3)
            number('number input for Maximum contradiction score', 0.8)
            start()
            ready()
            results['default_acceptance_status'] = status.input_value()
            results['default_acceptance_evidence'] = {
                'query': page.get_by_role('textbox', name='Sentence used for retrieval', exact=True).input_value(),
                'replacement': page.get_by_role('textbox', name='Proposed replacement', exact=True).input_value(),
                'references': evidence.input_value(),
                'preview': preview.input_value(),
            }
            default_query = results['default_acceptance_evidence']['query']
            if default_query.startswith(PREFIX):
                assert re.match(re.escape(PREFIX) + r'\s', default_query), default_query
            if 'Applied ' in status.input_value():
                record('default_acceptance', 2, allow_partial_failure=True)
            else:
                assert re.match(r'^AUTOMATIC GENERATION FAILED: No (?:corpus candidates|retrieved references)', status.input_value()), status.input_value()
                expect(raw).to_have_value(PREFIX)

            # Intentionally impossible STS floor proves no weak-reference fallback.
            number('number input for Minimum reference length relative to target', 0)
            number('number input for Minimum semantic similarity', 1)
            set_source('The satellite measured plasma density during a solar eruption.')
            start(unchanged='The satellite measured plasma density during a solar eruption.')
            expect(status).to_have_value(re.compile(r'^AUTOMATIC GENERATION FAILED: No retrieved references'), timeout=240000)
            ready()
            expect(raw).to_have_value('The satellite measured plasma density during a solar eruption.')
            results['strict_acceptance_refusal'] = status.input_value()

            number('number input for Minimum reference length relative to target', 0)
            number('number input for Minimum semantic similarity', 0)
            number('number input for Maximum contradiction score', 1)
            results['mechanical_run_acceptance'] = {'min_length_ratio': 0, 'min_semantic_score': 0,
                                                    'max_contradiction_score': 1, 'nuance': True}
            set_source(PREFIX)
            start()
            record('single_generate', 2, require_count=2)
            page.get_by_role('button', name='Undo rewrite', exact=True).click()
            expect(status).to_have_value('Rewrite undone.', timeout=20000)
            expect(raw).to_have_value(PREFIX)
            results['whole_run_undo'] = True
            number(LIMIT, 1)
            start()
            record('later_generate', 1)
            start('Regenerate')
            record('regenerate', 1)

            # Slow drafting leaves a reliable window to verify both Stop routes.
            parameters(rate=1)
            for button, name in [('Stop', 'native_stop'), ('Stop generation', 'rewrite_stop')]:
                set_source(PREFIX)
                start()
                expect(status).to_have_value(re.compile(r'^Drafting sentence 1'), timeout=30000)
                page.get_by_role('button', name=button, exact=True).click()
                expect(status).to_have_value(re.compile(r'^Stopped automatic generation\.'), timeout=60000)
                ready()
                expect(raw).to_have_value(PREFIX)
                results[name] = {'status': status.input_value(), 'source_preserved': True}
            parameters(rate=0)
            start()
            record('stop_retry', 1)
            # Native Gradio multiline Textbox.submit uses Shift+Enter; plain
            # Enter inserts a newline. Exercise that submission key while
            # retaining the one-sentence cap and mechanical settings.
            number(LIMIT, 1)
            set_source(PREFIX)
            start('Shift+Enter')
            record('single_native_submit', 1)
            auto.uncheck()
            set_source(PREFIX)
            generate.click()
            expect(raw).not_to_have_value(PREFIX, timeout=60000)
            ready()
            results['ordinary_generate_disabled'] = raw.input_value()

            layout(True)
            build()
            number(LIMIT, 1)
            number('number input for Minimum reference length relative to target', 0)
            number('number input for Minimum semantic similarity', 0)
            number('number input for Maximum contradiction score', 1)
            set_source(PREFIX, two_columns=True)
            start(unchanged='', two_columns=True)
            generated = record('two_column_generate', 1, two_columns=True)
            expect(page.get_by_role('textbox', name='Input', exact=True)).to_have_value(PREFIX)
            page.get_by_role('button', name='Undo rewrite', exact=True).click()
            expect(status).to_have_value('Rewrite undone.', timeout=20000)
            expect(page.locator('#textbox-default textarea')).to_have_value('')
            results['two_column_undo'] = True
            set_source(PREFIX, two_columns=True)
            start('Shift+Enter', unchanged='', two_columns=True)
            record('two_column_native_submit', 1, two_columns=True)
            expect(page.get_by_role('textbox', name='Input', exact=True)).to_have_value(PREFIX)
            # Continue must use the existing output, not the unrelated left input.
            output_source = 'The station had fallen silent. She reluctantly agreed to'
            left_source = 'This unrelated left input must not supply the continuation.'
            set_source(left_source, two_columns=True, output=output_source)
            start('Continue', unchanged=output_source, two_columns=True)
            continued = record('two_column_continue', 1, two_columns=True)
            assert continued.startswith('The station had fallen silent. '), continued
            assert not continued.startswith(left_source), continued
            expect(page.get_by_role('textbox', name='Input', exact=True)).to_have_value(left_source)
            # Empty output is still Continue's source. It must not silently use
            # the left input. The initial staged preview establishes the source
            # before any new sentence is accepted, then the normal count/evidence
            # checks validate the finite empty-prompt continuation.
            set_source(left_source, two_columns=True, output='')
            start('Continue', unchanged='', two_columns=True)
            expect(preview).to_have_value('')
            empty_continued = record('two_column_empty_continue', 1, two_columns=True)
            assert not empty_continued.startswith(left_source), empty_continued
            expect(page.get_by_role('textbox', name='Input', exact=True)).to_have_value(left_source)
            auto.uncheck()
            set_source(PREFIX, two_columns=True, output=generated)
            generate.click()
            expect(page.locator('#textbox-default textarea')).not_to_have_value(generated, timeout=60000)
            ready()
            results['two_column_ordinary_generate_disabled'] = page.locator('#textbox-default textarea').input_value()
            page.screenshot(path=str(args.output_dir / 'automatic-workflow.png'), full_page=True)
            results['passed'] = True
            print('Automatic browser workflow passed', flush=True)
            browser.close()
    except Exception as exc:
        results['error'] = f'{type(exc).__name__}: {exc}'
        if page is not None and not page.is_closed():
            try:
                page.screenshot(path=str(args.output_dir / 'automatic-failure.png'), full_page=True)
            except Exception:
                pass
        raise
    finally:
        (args.output_dir / 'browser-results.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
