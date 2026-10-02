"""Read-only comparison of this fork with an explicitly fetched upstream ref.

Uses only Python's standard library and Git. No fetch, merge, checkout, install,
metadata update, or model loading occurs. Exit 0 means the report was produced,
not that the candidate is compatible or safe to release.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


class ReportError(Exception):
    pass


def run_git(repo, *args):
    env = os.environ.copy()
    # Defense for versions that support it; partial clones are also rejected
    # explicitly before reading objects, including on older Git versions.
    env['GIT_NO_LAZY_FETCH'] = '1'
    env['GIT_TERMINAL_PROMPT'] = '0'
    return subprocess.run(
        ['git', '--no-optional-locks', '-c', 'core.fsmonitor=false', '-C', str(repo), *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env,
    )


def git(repo, *args):
    result = run_git(repo, *args)
    if result.returncode:
        detail = result.stderr.decode('utf-8', errors='replace').strip()
        raise ReportError(detail or f'git {args[0]} failed ({result.returncode})')
    return result.stdout


def commit(repo, ref):
    return git(repo, 'rev-parse', '--verify', '--end-of-options',
               ref + '^{commit}').decode('ascii').strip()


def is_ancestor(repo, ancestor, descendant):
    result = run_git(repo, 'merge-base', '--is-ancestor', ancestor, descendant)
    if result.returncode not in (0, 1):
        raise ReportError(result.stderr.decode('utf-8', errors='replace').strip())
    return result.returncode == 0


def changed_paths(repo, before, after):
    # Treat renames as delete/add so either old or new path can be an overlap.
    raw = git(repo, 'diff', '--no-ext-diff', '--no-textconv', '--name-only',
              '--no-renames', '-z', before, after, '--')
    return sorted(path.decode('utf-8', errors='surrogateescape')
                  for path in raw.split(b'\0') if path)


def require_full_clone(repo):
    partial = run_git(repo, 'config', '--get', 'extensions.partialclone')
    promisor = run_git(repo, 'config', '--type=bool', '--get-regexp', r'^remote\..*\.promisor$')
    for result in (partial, promisor):
        if result.returncode not in (0, 1):
            raise ReportError(result.stderr.decode('utf-8', errors='replace').strip())
    if partial.returncode == 0 or any(line.rsplit(None, 1)[-1:] == [b'true']
                                     for line in promisor.stdout.splitlines()):
        raise ReportError('Partial/promisor clones are unsupported: Git may fetch missing objects implicitly. Use a full clone.')


def report(repo, baseline, target_ref):
    require_full_clone(repo)
    baseline, head, target = (commit(repo, ref) for ref in (baseline, 'HEAD', target_ref))
    for label, candidate in [('HEAD', head), ('upstream target', target)]:
        if not is_ancestor(repo, baseline, candidate):
            raise ReportError(
                f'The recorded upstream baseline is not an ancestor of {label}. '
                'Check the selected ref, sync metadata, and whether this is a shallow clone; '
                'do not advance the baseline to suppress this error.'
            )
    incoming = changed_paths(repo, baseline, target)
    fork = changed_paths(repo, baseline, head)
    dirty = bool(git(repo, 'status', '--porcelain=v1', '-z', '--untracked-files=normal'))
    return {
        'baseline': baseline,
        'head': head,
        'target_ref': target_ref,
        'target': target,
        'target_already_in_head': is_ancestor(repo, target, head),
        'working_tree_dirty': dirty,
        'upstream_commits_since_baseline': int(git(repo, 'rev-list', '--count', f'{baseline}..{target}')),
        'incoming_paths': incoming,
        'fork_paths': fork,
        'overlap_paths': sorted(set(incoming).intersection(fork)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', nargs='?', default='refs/remotes/upstream/main',
                        help='Already fetched ref/commit (default: local upstream/main, possibly stale).')
    parser.add_argument('--json', action='store_true', help='Emit a machine-readable report.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        state = json.loads((root / 'docs/upstream-sync.json').read_text(encoding='utf-8'))
        if state['schema_version'] != 1:
            raise ReportError('Unsupported upstream-sync.json schema_version')
        baseline = state['last_integrated_commit']
        if not isinstance(baseline, str) or len(baseline) != 40 or any(c not in '0123456789abcdef' for c in baseline):
            raise ReportError('last_integrated_commit must be a full lowercase Git commit SHA')
        result = report(root, baseline, args.target)
    except (ReportError, OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'Cannot compare upstream: {exc}\nFetch the desired ref explicitly; see docs/UPSTREAM-MAINTENANCE.md.\n')
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"Recorded upstream baseline: {result['baseline']}")
    print(f"Fork HEAD:                  {result['head']}")
    print(f"Candidate ({args.target}): {result['target']}")
    print('Local refs only; this command has not checked for new releases or fetched anything.')
    if result['working_tree_dirty']:
        print('Working tree has uncommitted changes; this report compares committed HEAD only.')
    if result['target_already_in_head']:
        print('The candidate is already in HEAD history; confirm whether the sync record is current.')
    print(f"Upstream commits since baseline: {result['upstream_commits_since_baseline']}")
    for label, paths in [('Incoming changed paths', result['incoming_paths']),
                         ('Fork delta paths', result['fork_paths']),
                         ('Paths changed on both sides', result['overlap_paths'])]:
        print(f'\n{label} ({len(paths)}):')
        for path in paths:
            print('  ' + json.dumps(path, ensure_ascii=True))
    print('\nOverlaps are review targets, not predicted conflicts. Review cross-file API changes too.')


if __name__ == '__main__':
    main()
