"""
Custom Pygments lexer for SPARC DFT calculator template files.

Covers VASP INCAR, CP2K, ORCA, xTB, Quantum ESPRESSO, and Gaussian
input formats. Register in conf.py via the setup() hook.
"""
import re

from pygments.lexer import RegexLexer, bygroups
from pygments.token import Comment, Keyword, Name, Operator, String, Text, Number


class SparcTemplateLexer(RegexLexer):
    """
    Highlights common patterns across all SPARC template file formats:

    - Lines starting with # or ! -> Comment (blue)
    - &SECTION / &END SECTION, bare / -> Keyword
    - %pal, %scf, *xyz (ORCA directives) -> Name.Decorator
    - key = value -> Name.Attribute + Operator + String
    - Numbers -> Number
    """
    name = 'SparcTemplate'
    aliases = ['sparc-template']
    flags = re.MULTILINE

    tokens = {
        'root': [
            # Comment lines — # or ! at start of line (with optional whitespace)
            (r'^[ \t]*[#!][^\n]*', Comment.Single),
            # Fortran/CP2K/QE section markers: &WORD, &END WORD, bare /
            (r'&(?:END\s+)?\w+', Keyword),
            (r'^\s*/$', Keyword),
            # ORCA block directives (%pal, %scf) and coord marker (*xyz)
            (r'%\w+|\*\w+', Name.Decorator),
            # key = value
            (r'([ \t]*[\w]+)([ \t]*=[ \t]*)([^\n]*)',
             bygroups(Name.Attribute, Operator, String)),
            # Bare numbers (QE data lines, ATOMIC_SPECIES masses, k-point grids)
            (r'\b\d+\.?\d*(?:[dDeE][+-]?\d+)?\b', Number),
            # Whitespace
            (r'\s+', Text),
            # Anything else
            (r'[^\s]', Text),
        ]
    }
