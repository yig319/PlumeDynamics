"""Build entry point for PlumeDynamics.

Most package metadata lives in ``setup.cfg``. Runtime dependencies are read
from ``requirements.txt`` to match the AFM-tools publishing workflow.
"""

from pathlib import Path

from setuptools import setup


def _read_requirements(path: str) -> list[str]:
    """Read non-comment requirement lines from a local requirements file."""

    req_path = Path(__file__).parent / path
    requirements: list[str] = []
    for line in req_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r"):
            raise ValueError("Nested requirements files are not supported for install_requires")
        requirements.append(line)
    return requirements


if __name__ == "__main__":
    try:
        setup(
            install_requires=_read_requirements("requirements.txt"),
            use_scm_version={"version_scheme": "no-guess-dev"},
        )
    except Exception:
        print(
            "\n\nAn error occurred while building the project. Please ensure you "
            "have current build tooling with:\n"
            "   pip install -U setuptools setuptools_scm wheel\n\n"
        )
        raise
