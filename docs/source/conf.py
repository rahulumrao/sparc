# Configuration file for the Sphinx documentation builder.
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Ensure sparc package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '_ext'))  # Custom extensions

from sparc_lexer import SparcTemplateLexer


def setup(app):
    app.add_lexer('sparc-template', SparcTemplateLexer)

project = 'SPARC'
copyright = '2024'
author = 'Rahul Verma'

# The full version, including alpha/beta/rc tags
release = '0.1.0'

# Add any Sphinx extension module names here
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme',
    'sphinx_autodoc_typehints',
    'nbsphinx',
]

# Mock optional heavy dependencies so autodoc can import modules without them
autodoc_mock_imports = [
    'chemiscope',
    'deepmd',
    'dpdata',
    'lammps',
    'plumed',
    'nglview',
]

# # Optional
# source_suffix = {
#     '.rst': 'restructuredtext',
#     '.ipynb': 'nbsphinx',
# }

# Add type hint descriptions
autodoc_typehints = "description"

# Add any paths that contain templates here
templates_path = ['_templates']

# List of patterns to exclude
exclude_patterns = ['_build', 'Thumbs.db']

# The theme to use for HTML and HTML Help pages
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files
html_static_path = ['_static', '../_static']
html_css_files = ['custom.css']
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "titles_only": False,
}

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'ase': ('https://wiki.fysik.dtu.dk/ase/', None),
}
