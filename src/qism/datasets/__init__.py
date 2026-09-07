"""Dataset generation, Parquet writing, and provenance manifests.

Note: ``qism.datasets.generator`` (which imports ``qism.simulation.runner``)
is intentionally not imported here to keep the package graph acyclic; import
it explicitly when needed.
"""

from qism.datasets.manifest import (
    build_manifest,
    git_commit,
    package_versions,
    write_manifest_atomic,
)
from qism.datasets.writer import DatasetWriter

__all__ = [
    "DatasetWriter",
    "build_manifest",
    "git_commit",
    "package_versions",
    "write_manifest_atomic",
]
