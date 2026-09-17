import tempfile
import unittest
from pathlib import Path

from tiling_template.discovery import (
    AssetDiscoverer,
    AssetSpec,
    GlobMatcher,
    RegexMatcher,
)
from tiling_template.records import AssetRef


class TestMatchers(unittest.TestCase):
    def test_glob_matcher(self):
        matcher = GlobMatcher('*.csv')

        self.assertTrue(matcher.matches('file.csv'))
        self.assertFalse(matcher.matches('file.tif'))

    def test_regex_matcher(self):
        matcher = RegexMatcher(r'.*\.csv')

        self.assertTrue(matcher.matches('file.csv'))
        self.assertFalse(matcher.matches('file.tif'))

    def test_regex_matcher_rejects_invalid_pattern(self):
        with self.assertRaisesRegex(ValueError, 'Invalid regex pattern'):
            RegexMatcher('*.csv')


class TestAssetSpec(unittest.TestCase):
    def test_creates_selected_matcher(self):
        glob_spec = AssetSpec(
            spec_id='glob_csv',
            root=Path('/not/read/during/spec/construction'),
            role='input',
            matcher_type='glob',
            modality='csv',
            pattern='*.csv',
            match_scope='basename',
        )
        regex_spec = AssetSpec(
            spec_id='regex_csv',
            root=Path('/not/read/during/spec/construction'),
            role='label',
            matcher_type='regex',
            modality='csv',
            pattern=r'.*\.csv',
            match_scope='basename',
        )

        self.assertIsInstance(glob_spec.matcher, GlobMatcher)
        self.assertIsInstance(regex_spec.matcher, RegexMatcher)

    def test_rejects_invalid_matcher_type(self):
        with self.assertRaisesRegex(ValueError, 'matcher type'):
            AssetSpec(
                spec_id='invalid',
                root=Path('.'),
                role='input',
                matcher_type='invalid',  # type: ignore[arg-type]
                modality='csv',
                pattern='*.csv',
                match_scope='basename',
            )

    def test_rejects_invalid_match_scope(self):
        with self.assertRaisesRegex(ValueError, 'match scope'):
            AssetSpec(
                spec_id='invalid',
                root=Path('.'),
                role='input',
                matcher_type='glob',
                modality='csv',
                pattern='*.csv',
                match_scope='invalid',  # type: ignore[arg-type]
            )

    def test_rejects_negative_max_depth(self):
        with self.assertRaisesRegex(ValueError, 'max_depth'):
            AssetSpec(
                spec_id='invalid',
                root=Path('.'),
                role='input',
                matcher_type='glob',
                modality='csv',
                pattern='*.csv',
                match_scope='basename',
                max_depth=-1,
            )


class TestAssetDiscoverer(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

        (self.root / 'b.csv').write_text('bbb')
        (self.root / 'a.csv').write_text('a')
        (self.root / 'ignore.txt').write_text('ignored')
        (self.root / '.hidden.csv').write_text('hidden')

        (self.root / 'nested').mkdir()
        (self.root / 'nested' / 'c.csv').write_text('cc')

        (self.root / 'nested' / 'deeper').mkdir()
        (self.root / 'nested' / 'deeper' / 'd.csv').write_text('dddd')

        (self.root / '.hidden_directory').mkdir()
        (self.root / '.hidden_directory' / 'secret.csv').write_text('secret')

    def tearDown(self):
        self.temporary_directory.cleanup()

    def make_spec(self, **overrides):
        values = {
            'spec_id': 'csv_input',
            'root': self.root,
            'role': 'input',
            'matcher_type': 'glob',
            'modality': 'csv',
            'pattern': '*.csv',
            'match_scope': 'basename',
            'recursive': True,
        }
        values.update(overrides)
        return AssetSpec(**values)

    def test_nonrecursive_discovery_is_sorted(self):
        discoverer = AssetDiscoverer((self.make_spec(recursive=False),))

        results = discoverer.discover()

        self.assertEqual(
            [result.relative_path.as_posix() for result in results],
            ['a.csv', 'b.csv'],
        )

    def test_recursive_discovery_and_asset_metadata(self):
        discoverer = AssetDiscoverer((self.make_spec(),))

        results = discoverer.discover()

        self.assertEqual(
            [result.relative_path.as_posix() for result in results],
            ['a.csv', 'b.csv', 'nested/c.csv', 'nested/deeper/d.csv'],
        )
        first = results[0]
        self.assertIsInstance(first, AssetRef)
        self.assertTrue(first.path.is_absolute())
        self.assertEqual(first.spec_id, 'csv_input')
        self.assertEqual(first.role, 'input')
        self.assertEqual(first.modality, 'csv')
        self.assertEqual(first.size_bytes, 1)
        self.assertIsInstance(first.modified_time_ns, int)

    def test_max_depth_includes_files_at_depth_limit(self):
        discoverer = AssetDiscoverer((self.make_spec(max_depth=1),))

        results = discoverer.discover()

        self.assertEqual(
            [result.relative_path.as_posix() for result in results],
            ['a.csv', 'b.csv', 'nested/c.csv'],
        )

    def test_exclusion_patterns_prune_directories(self):
        (self.root / 'excluded_directory').mkdir()
        (self.root / 'excluded_directory' / 'excluded.csv').write_text(
            'excluded'
        )
        discoverer = AssetDiscoverer((
            self.make_spec(exclude_patterns=('excluded_directory',)),
        ))

        results = discoverer.discover()

        self.assertNotIn(
            'excluded_directory/excluded.csv',
            [result.relative_path.as_posix() for result in results],
        )

    def test_relative_path_scope(self):
        discoverer = AssetDiscoverer((
            self.make_spec(
                pattern='nested/*.csv',
                match_scope='relative_path',
            ),
        ))

        results = discoverer.discover()

        self.assertEqual(
            [result.relative_path.as_posix() for result in results],
            ['nested/c.csv', 'nested/deeper/d.csv'],
        )

    def test_stem_scope(self):
        discoverer = AssetDiscoverer((
            self.make_spec(pattern='a', match_scope='stem'),
        ))

        results = discoverer.discover()

        self.assertEqual(
            [result.relative_path.as_posix() for result in results],
            ['a.csv'],
        )

    def test_regex_discovery(self):
        discoverer = AssetDiscoverer((
            self.make_spec(
                matcher_type='regex',
                pattern=r'.*\.csv',
                recursive=False,
            ),
        ))

        results = discoverer.discover()

        self.assertEqual(
            [result.relative_path.as_posix() for result in results],
            ['a.csv', 'b.csv'],
        )

    def test_rejects_missing_root(self):
        discoverer = AssetDiscoverer((
            self.make_spec(root=self.root / 'missing'),
        ))

        with self.assertRaisesRegex(ValueError, 'does not exist'):
            discoverer.discover()

    def test_rejects_file_root(self):
        file_root = self.root / 'a.csv'
        discoverer = AssetDiscoverer((self.make_spec(root=file_root),))

        with self.assertRaisesRegex(ValueError, 'not a directory'):
            discoverer.discover()

    def test_rejects_duplicate_spec_ids(self):
        spec = self.make_spec()

        with self.assertRaisesRegex(ValueError, 'spec_id values must be unique'):
            AssetDiscoverer((spec, spec))

    def test_rejects_asset_matched_by_multiple_specs(self):
        first = self.make_spec(spec_id='first')
        second = self.make_spec(spec_id='second')
        discoverer = AssetDiscoverer((first, second))

        with self.assertRaisesRegex(ValueError, 'matched multiple specs'):
            discoverer.discover()


if __name__ == '__main__':
    unittest.main()
