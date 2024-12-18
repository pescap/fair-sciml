import os
import sys

# Recursively add all subdirectories in 'src' to the Python path
src_path = os.path.abspath("../src")
for root, dirs, _ in os.walk(src_path):
    sys.path.insert(0, root)

# -- Project information -----------------------------------------------------
project = "fair-sciml"
copyright = "2024, Eduardo Alvarez, Paul Escapil, Adolfo Parra"
author = "Eduardo Alvarez, Paul Escapil, Adolfo Parra"
release = "1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

autosummary_generate = True  # Automatically generate summary tables
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": False,
    "show-inheritance": True,
}
