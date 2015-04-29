from codebase.data.dbd4.feature_extractors.b_value_extractor import BValueExtractor
from codebase.data.dbd4.feature_extractors.d1_plain_shape_distribution import D1PlainShapeDistributionExtractor
from codebase.data.dbd4.feature_extractors.d1_sureface_shape_distribution import D1SurfaceShapeDistributionExtractor
from codebase.data.dbd4.feature_extractors.d2_category_shape_distribution import D2CategoryShapeDistributionExtractor
from codebase.data.dbd4.feature_extractors.d2_plain_shape_distribution import D2PlainShapeDistributionExtractor
from codebase.data.dbd4.feature_extractors.d2_sureface_shape_distribution import D2SurfaceShapeDistributionExtractor
from codebase.data.dbd4.feature_extractors.half_sphere_exposure_extractor import HalfSphereExposureExtractor
from codebase.data.dbd4.feature_extractors.profile_extractor import ProfileExtractor
from codebase.data.dbd4.feature_extractors.protrusion_index_extractor import ProtrusionIndexExtractor
from codebase.data.dbd4.feature_extractors.residue_depth_extractor import ResidueDepthExtractor
from codebase.data.dbd4.feature_extractors.stride_secondary_srtucture_extractor import StrideSecondaryStructureExtractor
from codebase.pairpred.model.enums import Features

__author__ = 'basir'


class DBD4FeatureExtractor():
    def __init__(self):
        pass

    @staticmethod
    def extract(features, database, **kwargs):
        computed_features = set()
        secondary_features = {Features.SECONDARY_STRUCTURE, Features.RELATIVE_ACCESSIBLE_SURFACE_AREA,
                              Features.ACCESSIBLE_SURFACE_AREA, Features.PHI, Features.PSI}
        profile_features = {Features.SEQUENCE_PROFILE, Features.POSITION_SPECIFIC_SCORING_MATRIX,
                            Features.POSITION_SPECIFIC_FREQUENCY_MATRIX,
                            Features.WINDOWED_POSITION_SPECIFIC_SCORING_MATRIX,
                            Features.WINDOWED_POSITION_SPECIFIC_FREQUENCY_MATRIX}
        for feature in features:
            if feature in computed_features:
                continue
            computed_features.add(feature)
            if feature in profile_features:
                ProfileExtractor(database).extract_feature()
                computed_features = computed_features.union(profile_features)
            elif feature == Features.D1_PLAIN_SHAPE_DISTRIBUTION:
                D1PlainShapeDistributionExtractor(database, **kwargs).extract_feature()
            elif feature == Features.D2_PLAIN_SHAPE_DISTRIBUTION:
                D2PlainShapeDistributionExtractor(database, **kwargs).extract_feature()
            elif feature == Features.D2_SURFACE_SHAPE_DISTRIBUTION:
                D2SurfaceShapeDistributionExtractor(database, **kwargs).extract_feature()
            elif feature == Features.D1_SURFACE_SHAPE_DISTRIBUTION:
                D1SurfaceShapeDistributionExtractor(database, **kwargs).extract_feature()
            elif feature == Features.D2_CATEGORY_SHAPE_DISTRIBUTION:
                D2CategoryShapeDistributionExtractor(database, **kwargs).extract_feature()
            elif feature == Features.RESIDUE_DEPTH:
                ResidueDepthExtractor(database, **kwargs).extract_feature()
            elif feature == Features.HALF_SPHERE_EXPOSURE:
                HalfSphereExposureExtractor(database, **kwargs).extract_feature()
            elif feature == Features.B_VALUE:
                BValueExtractor(database).extract_feature()
            elif feature == Features.PROTRUSION_INDEX:
                ProtrusionIndexExtractor(database).extract_feature()
            elif feature in secondary_features:
                StrideSecondaryStructureExtractor(database).extract_feature()
                computed_features = computed_features.union(secondary_features)
