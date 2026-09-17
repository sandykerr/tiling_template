import unittest
from pathlib import Path

from tiling_template.association import (
    AssociationStrategy,
    AssociationValidationError,
    AssetAssociator,
    CompositeAssociationStrategy,
    DirectoryAssociationStrategy,
    FilenameAssociationStrategy,
)
from tiling_template.configs import ModalityAssociationConfig
from tiling_template.records import AssetRef, SourceRecord


def make_asset(
    relative_path: str,
    *,
    role: str = 'input',
    modality: str = 'optical',
    spec_id: str = 'assets',
) -> AssetRef:
    relative = Path(relative_path)
    return AssetRef(
        path=Path('/dataset') / relative,
        relative_path=relative,
        spec_id=spec_id,
        role=role,
        modality=modality,
        size_bytes=1,
        modified_time_ns=1,
    )


class TestFilenameAssociationStrategy(unittest.TestCase):
    def test_stem_with_prefix_and_suffix(self):
        strategy = FilenameAssociationStrategy(
            match_type='stem',
            prefix_to_remove='input_',
            suffix_to_remove='_image',
        )

        key = strategy.key_for(make_asset('input_scene_01_image.tif'))

        self.assertEqual(key, ('scene_01',))

    def test_basename_preserves_extension(self):
        strategy = FilenameAssociationStrategy(match_type='basename')

        self.assertEqual(
            strategy.key_for(make_asset('scene_01.tif')),
            ('scene_01.tif',),
        )

    def test_missing_configured_affix_is_rejected(self):
        strategy = FilenameAssociationStrategy(
            match_type='stem',
            suffix_to_remove='_label',
        )

        with self.assertRaisesRegex(ValueError, 'does not end with'):
            strategy.key_for(make_asset('scene_01.tif'))

    def test_empty_normalized_key_is_rejected(self):
        strategy = FilenameAssociationStrategy(
            match_type='stem',
            suffix_to_remove='scene_01',
        )

        with self.assertRaisesRegex(ValueError, 'empty key'):
            strategy.key_for(make_asset('scene_01.tif'))


class TestDirectoryAssociationStrategy(unittest.TestCase):
    def test_immediate_parent(self):
        strategy = DirectoryAssociationStrategy('immediate_parent')

        self.assertEqual(
            strategy.key_for(make_asset('region/scene/file.tif')),
            ('scene',),
        )

    def test_selected_parent_indices_preserve_order(self):
        strategy = DirectoryAssociationStrategy(
            'parent_indices',
            parent_indices=(0, -1),
        )

        self.assertEqual(
            strategy.key_for(make_asset('region/year/scene/file.tif')),
            ('region', 'scene'),
        )

    def test_out_of_bounds_parent_index_is_rejected(self):
        strategy = DirectoryAssociationStrategy(
            'parent_indices',
            parent_indices=(2,),
        )

        with self.assertRaisesRegex(ValueError, 'out of bounds'):
            strategy.key_for(make_asset('region/file.tif'))

    def test_root_level_asset_is_rejected(self):
        strategy = DirectoryAssociationStrategy('immediate_parent')

        with self.assertRaisesRegex(ValueError, 'no relative parent'):
            strategy.key_for(make_asset('file.tif'))

    def test_parent_indices_are_required(self):
        with self.assertRaisesRegex(ValueError, 'requires at least one index'):
            DirectoryAssociationStrategy('parent_indices')


class TestCompositeAssociationStrategy(unittest.TestCase):
    def test_flattens_component_keys_in_strategy_order(self):
        strategy = CompositeAssociationStrategy((
            DirectoryAssociationStrategy('immediate_parent'),
            FilenameAssociationStrategy(match_type='stem'),
        ))

        self.assertEqual(
            strategy.key_for(make_asset('region/scene/file.tif')),
            ('scene', 'file'),
        )

    def test_empty_composite_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'at least one strategy'):
            CompositeAssociationStrategy(())


class TestModalityAssociationConfig(unittest.TestCase):
    def test_optional_single_asset(self):
        config = ModalityAssociationConfig(
            role='label',
            modality='mask',
            minimum_count=0,
            maximum_count=1,
        )

        self.assertEqual(config.minimum_count, 0)
        self.assertEqual(config.maximum_count, 1)

    def test_unlimited_maximum(self):
        config = ModalityAssociationConfig('input', 'temporal')

        self.assertIsNone(config.maximum_count)

    def test_can_prohibit_a_modality(self):
        config = ModalityAssociationConfig(
            role='label',
            modality='mask',
            minimum_count=0,
            maximum_count=0,
        )

        self.assertEqual(config.maximum_count, 0)

    def test_invalid_count_range_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'less than minimum'):
            ModalityAssociationConfig(
                role='input',
                modality='optical',
                minimum_count=2,
                maximum_count=1,
            )


class InvalidAssociationStrategy(AssociationStrategy):
    def key_for(self, asset: AssetRef):
        return ''


class TestAssetAssociator(unittest.TestCase):
    def setUp(self):
        self.assets = (
            make_asset('scene_b/input.tif'),
            make_asset(
                'scene_b/label.tif',
                role='label',
                modality='mask',
            ),
            make_asset('scene_a/input.tif'),
            make_asset(
                'scene_a/label.tif',
                role='label',
                modality='mask',
            ),
        )
        self.requirements = (
            ModalityAssociationConfig('input', 'optical', 1, 1),
            ModalityAssociationConfig('label', 'mask', 1, 1),
        )
        self.strategy = DirectoryAssociationStrategy('immediate_parent')

    def test_groups_valid_sources_deterministically(self):
        associator = AssetAssociator(
            self.assets,
            self.strategy,
            self.requirements,
        )

        records = associator.associate_assets()

        self.assertTrue(all(isinstance(record, SourceRecord) for record in records))
        self.assertEqual(
            [record.association_key for record in records],
            [('scene_a',), ('scene_b',)],
        )
        self.assertEqual(
            [record.source_id for record in records],
            ['scene_a', 'scene_b'],
        )
        self.assertEqual(
            [(asset.role, asset.modality) for asset in records[0].assets],
            [('input', 'optical'), ('label', 'mask')],
        )

    def test_reports_all_invalid_groups(self):
        input_assets = tuple(
            asset for asset in self.assets if asset.role == 'input'
        )
        associator = AssetAssociator(
            input_assets,
            self.strategy,
            self.requirements,
        )

        with self.assertRaises(AssociationValidationError) as context:
            associator.associate_assets()

        self.assertEqual(len(context.exception.issues), 2)
        self.assertTrue(all('label/mask' in issue for issue in context.exception.issues))

    def test_rejects_too_many_assets(self):
        duplicate_modality = make_asset('scene_a/second_input.tif')
        associator = AssetAssociator(
            (*self.assets, duplicate_modality),
            self.strategy,
            self.requirements,
        )

        with self.assertRaisesRegex(AssociationValidationError, 'at most 1'):
            associator.associate_assets()

    def test_optional_missing_modality_is_valid(self):
        requirements = (
            self.requirements[0],
            ModalityAssociationConfig('qa', 'cloud', 0, 1),
        )
        input_assets = tuple(
            asset for asset in self.assets if asset.role == 'input'
        )
        associator = AssetAssociator(
            input_assets,
            self.strategy,
            requirements,
        )

        self.assertEqual(len(associator.associate_assets()), 2)

    def test_can_reject_unconfigured_modalities(self):
        associator = AssetAssociator(
            self.assets,
            self.strategy,
            (self.requirements[0],),
            reject_unconfigured=True,
        )

        with self.assertRaisesRegex(
            AssociationValidationError,
            'unconfigured label/mask',
        ):
            associator.associate_assets()

    def test_rejects_duplicate_modality_configs(self):
        with self.assertRaisesRegex(ValueError, 'unique role/modality pairs'):
            AssetAssociator(
                self.assets,
                self.strategy,
                (self.requirements[0], self.requirements[0]),
            )

    def test_rejects_duplicate_asset_paths(self):
        with self.assertRaisesRegex(ValueError, 'duplicate paths'):
            AssetAssociator(
                (self.assets[0], self.assets[0]),
                self.strategy,
                self.requirements,
            )

    def test_rejects_invalid_strategy_key(self):
        associator = AssetAssociator(
            self.assets,
            InvalidAssociationStrategy(),
            self.requirements,
        )

        with self.assertRaisesRegex(ValueError, 'invalid key'):
            associator.associate_assets()

    def test_custom_source_id_factory(self):
        associator = AssetAssociator(
            self.assets,
            self.strategy,
            self.requirements,
            source_id_factory=lambda key: f'source-{key[0]}',
        )

        self.assertEqual(
            [record.source_id for record in associator.associate_assets()],
            ['source-scene_a', 'source-scene_b'],
        )

    def test_rejects_source_id_collisions(self):
        associator = AssetAssociator(
            self.assets,
            self.strategy,
            self.requirements,
            source_id_factory=lambda key: 'same-source',
        )

        with self.assertRaisesRegex(ValueError, 'multiple keys'):
            associator.associate_assets()


if __name__ == '__main__':
    unittest.main()
