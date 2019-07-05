import os
import sys

import sphinx_rtd_theme
sys.path.insert(0, os.path.abspath('_ext'))
# from recommonmark.parser import CommonMarkParser

sys.path.insert(0, os.path.abspath('..'))
sys.path.append(os.path.dirname(__file__))
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "readthedocs.settings.dev")

# from django.conf import settings

# import django
# django.setup()


sys.path.append(os.path.abspath('_ext'))
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
]
templates_path = ['_templates']

edit_on_github_project = 'machinelearningmindset/machine-learning-course'

edit_on_github_branch = 'master'

source_suffix = ['.rst', '.md']
# source_parsers = {
#     '.md': CommonMarkParser,
# }

master_doc = 'index'
project = u'Machine-Learning-Course'
copyright = u'2019, Amirsina Torfi'
author = u'Amirsina Torfi \n Brendan Sherman, James E Hopkins, Zac Smith, Eric Wynn'
version = '1.0'
release = '1.0'
exclude_patterns = ['_build']
default_role = 'obj'
pygments_style = 'sphinx'
# intersphinx_mapping = {
#     'TensorFlow-World': ('https://github.com/astorfi/TensorFlow-World', None),
# }
htmlhelp_basename = 'ReadTheDocsdoc'
latex_documents = [
    (master_doc, 'Machine-Learning-Course.tex', u'Machine-Learning-Course Documentation',
     u'Amirsina Torfi \n Brendan Sherman, James E Hopkins, Zac Smith, Eric Wynn', 'manual'),
]

latex_logo = 'logo/pythonml.pdf'
latex_additional_files = ['flaskstyle.sty', 'logo/pythonml.pdf']

man_pages = [
    (master_doc, 'Machine-Learning-Course', u'Machine-Learning-Course Documentation',
     [author], 1)
]
exclude_patterns = [
    # 'api' # needed for ``make gettext`` to not die.
]

language = 'en'

locale_dirs = [
    'locale/',
]
gettext_compact = False

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_logo = 'logo/pythonml.png'
html_theme_options = {
    'logo_only': True,
    'display_version': False,
}

github_url='https://github.com/machinelearningmindset/machine-learning-course'

html_context = {
"display_github": True, # Add 'Edit on Github' link instead of 'View page source'
"last_updated": True,
"commit": False,
 'github_url': 'https://github.com/machinelearningmindset/machine-learning-course'
}


#def setup(app):
#     app.add_stylesheet('custom.css')
