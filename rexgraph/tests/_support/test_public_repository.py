"""Check publication rules against prose, real Git history and source assets."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('public_repository', Path(__file__).with_name('check_public_repository.py'))
repository = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repository)


class PublicRepository(unittest.TestCase):
    def test_private_documents_and_generated_files_reject(self):
        for name in ['AGENTS.md', 'internal/report.json', 'build/cache.o', 'capture.zip']:
            with self.subTest(name=name):
                self.assertIsNotNone(repository.reason(name))

    def test_reference_guides_frontends_and_native_fixtures_remain_source(self):
        for name in ['rexgraph/REFERENCE.md', 'agent/frontend/index.html',
                     'bindings/cpp/tests/fixtures/example.rfcxs']:
            with self.subTest(name=name):
                self.assertIsNone(repository.reason(name))

    def test_workspace_paths_and_private_links_reject(self):
        for text in ['/home/' + 'example/project', '../' + 'internal/notes.md']:
            self.assertTrue(repository.content_reasons('README.md', text.encode()))

    def test_decorative_and_promotional_prose_reject(self):
        for text in ['# Guide\n' + '-' * 7, 'We lever' + 'age a game-' + 'changing system']:
            self.assertTrue(repository.content_reasons('README.md', text.encode()))

    def test_mathematical_terms_and_boundary_members_remain_precise(self):
        text = "Foster's identity relates leverage scores to rank.\nUse B1.T and the B1.npy archive."
        self.assertEqual(repository.content_reasons('README.md', text.encode()), [])

    def test_python_and_cython_comments_are_checked(self):
        text = '# ' + 'C' + '03' + ' update\nx = 1\n'
        for suffix in ['.py', '.pyx', '.pxd']:
            self.assertTrue(repository.content_reasons('example' + suffix, text.encode()))

    def test_executable_strings_do_not_become_prose(self):
        text = 'text = "a game-' + 'changing example"\n'
        self.assertEqual(repository.content_reasons('example.py', text.encode()), [])

    def test_commit_structure_and_trailers(self):
        message = 'storage: preserve native source ownership\n\nRetain the actual source through replay.\n\nTests: 8 total.\n'
        self.assertEqual(repository.message_reasons(message), [])
        self.assertTrue(repository.message_reasons(message.replace('Tests: 8 total.', 'Validation passed.')))
        self.assertTrue(repository.message_reasons(message + '\nSigned-' + 'off-by: Example\n'))

    def test_attribution_rejects(self):
        text = 'Co-' + 'authored-by: Cod' + 'ex <example@example.com>'
        self.assertTrue(repository.content_reasons('README.md', text.encode()))

    def test_historical_blob_is_checked_after_current_file_is_clean(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def git(*args):
                return subprocess.check_output(['git', '-c', 'user.name=Repository Test',
                    '-c', 'user.email=test@example.com', *args], cwd=root, stderr=subprocess.DEVNULL)
            git('init', '-b', 'main')
            (root / 'README.md').write_text('../' + 'internal/notes.md\n')
            git('add', '--', 'README.md')
            git('commit', '-m', 'docs: describe query inputs\n\nTests: 1 total.')
            first = git('rev-parse', 'HEAD').decode().strip()
            (root / 'README.md').write_text('# Queries\n\nRetain original sources.\n')
            git('add', '--', 'README.md')
            git('commit', '-m', 'docs: retain query sources\n\nTests: 1 total.')
            with patch.object(repository, 'ROOT', root):
                self.assertTrue(list(repository.history_errors(repository.history_commits())))
                self.assertEqual(list(repository.history_errors(repository.history_commits(base=first))), [])
                self.assertEqual(repository.history_commits(event={'before': first}, event_name='push'),
                                 repository.history_commits(base=first))

    def _rewritten_history(self, root):
        def git(*args, input=None):
            return subprocess.check_output(['git', '-c', 'user.name=Repository Test',
                '-c', 'user.email=test@example.com', *args], cwd=root, input=input,
                stderr=subprocess.DEVNULL).decode().strip()
        git('init', '-b', 'main')
        (root / 'README.md').write_text('# Queries\n\nRetain original sources.\n')
        git('add', '--', 'README.md')
        git('commit', '-m', 'docs: describe query inputs\n\nTests: 1 total.')
        base = git('rev-parse', 'HEAD')
        (root / 'README.md').write_text('../' + 'internal/notes.md\n')
        git('add', '--', 'README.md')
        git('commit', '-m', 'docs: describe query storage\n\nTests: 1 total.')
        removed = git('rev-parse', 'HEAD')
        rewritten = git('commit-tree', base + '^{tree}', '-p', base,
                        input=b'docs: retain complete query sources\n\nTests: 1 total.\n')
        git('update-ref', 'refs/heads/main', rewritten, removed)
        return git, base, removed, rewritten

    def test_forced_push_checks_new_history_without_removed_commits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, removed, rewritten = self._rewritten_history(root)
            with patch.object(repository, 'ROOT', root):
                event = {'before': removed, 'forced': True}
                commits = repository.history_commits(event=event, event_name='push')
                self.assertEqual(commits, [rewritten])
                self.assertEqual(list(repository.history_errors(commits)), [])
                with self.assertRaises(ValueError):
                    repository.history_commits(base=removed)
                with self.assertRaises(ValueError):
                    repository.history_commits(event={'before': removed}, event_name='push')

    def test_forced_push_checks_each_new_commit_after_a_rewrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git, _, removed, rewritten = self._rewritten_history(root)
            (root / 'README.md').write_text('../' + 'internal/notes.md\n')
            git('add', '--', 'README.md')
            git('commit', '-m', 'docs: describe source storage\n\nTests: 1 total.')
            intermediate = git('rev-parse', 'HEAD')
            (root / 'README.md').write_text('# Queries\n\nRetain complete sources.\n')
            git('add', '--', 'README.md')
            git('commit', '-m', 'docs: retain query ownership\n\nTests: 1 total.')
            current = git('rev-parse', 'HEAD')
            with patch.object(repository, 'ROOT', root):
                event = {'before': removed, 'forced': True}
                commits = repository.history_commits(event=event, event_name='push')
                self.assertEqual(commits, [rewritten, intermediate, current])
                errors = list(repository.history_errors(commits))
                self.assertTrue(any(intermediate in error for error in errors))
                self.assertFalse(any(current in error for error in errors))
