from __future__ import annotations

import importlib.metadata
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

project = "Kahlus / NeuroTwin"
author = "Kahlus contributors"
copyright = "2026, Kahlus contributors"
try:
    release = version = importlib.metadata.version("neurotwin")
except importlib.metadata.PackageNotFoundError:
    release = version = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "numpydoc",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
master_doc = "index"
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**/.ipynb_checkpoints",
    "research/*.tex",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "substitution",
]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_extra_path = ["CNAME", ".nojekyll"] if (ROOT / "docs" / "CNAME").exists() else [".nojekyll"]
html_css_files = ["custom.css"]
html_title = "Kahlus / NeuroTwin Research Docs"
html_theme_options = {
    "github_url": "https://github.com/ptlnextdoor/Kahlus-V1",
    "show_toc_level": 2,
    "navbar_align": "left",
    "logo": {"text": "Kahlus / NeuroTwin"},
}

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"plot_.*\.py",
    "download_all_examples": False,
    "within_subsection_order": "FileNameSortKey",
    "remove_config_comments": True,
}

autosummary_generate = True
numpydoc_show_class_members = False
napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "mne": ("https://mne.tools/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# Legacy Markdown pages contain fenced `math`/`txt` blocks that are rendered as code.
suppress_warnings = ["misc.highlighting_failure"]
