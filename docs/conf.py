import os
import sys

# Add source directory to the Python path
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
project = "fair-sciml"
copyright = "2024"
author = "Eduardo Alvarez, Paul Escapil, Adolfo Parra"
release = "1.0"

# -- General configuration ---------------------------------------------------
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon"]

# Specify the master doc (entry point of your documentation)
master_doc = "index"

# Options for HTML output
html_theme = "sphinx_rtd_theme"
