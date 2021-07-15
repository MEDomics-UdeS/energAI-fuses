# -- Path setup --------------------------------------------------------------#
import os
import sys

sys.path.insert(0, os.path.abspath('../..'))    # Project $WORKDIR

# Adding every modules path for autodoc 
sys.path.append(os.path.abspath('../../src/coco'))
sys.path.append(os.path.abspath('../../src/data'))
sys.path.append(os.path.abspath('../../src/detr'))
sys.path.append(os.path.abspath('../../src/gui'))
sys.path.append(os.path.abspath('../../src/gui/modules'))
sys.path.append(os.path.abspath('../../src/models'))
sys.path.append(os.path.abspath('../../src/utils'))
sys.path.append(os.path.abspath('../../src/visualization'))


# -- Project information -----------------------------------------------------

project = 'EnergAI-fuses'
copyright = '2021, Simon Giard-Leroux, Guillaume Cléroux, Shreyas Sunil Kulkarni, Martin Vallières'
author = 'Simon Giard-Leroux, Guillaume Cléroux, Shreyas Sunil Kulkarni, Martin Vallières'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.autosummary']
autosummary_generate = True     # Turning on auto-summary for recursive generation

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '_templates']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
