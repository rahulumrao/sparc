"""
Pygments lexer for workflow-annotated SPARC ``input.yaml`` examples.

Stage banner comments (``# [Stage N — …]``) receive distinct token types so
Sphinx can color-code sections on the workflow overview page.
"""
from pygments.lexer import inherit
from pygments.lexers.data import YamlLexer
from pygments.token import Comment, Generic, Keyword


class SparcWorkflowYamlLexer(YamlLexer):
    """YAML with SPARC workflow stage banner highlighting."""

    name = "SPARC Workflow YAML"
    aliases = ["sparc-workflow-yaml"]
    filenames = []

    tokens = {
        "root": [
            (r"^# \[Stage 1[^\n]*", Comment.Preproc),
            (r"^# \[Stage 2[^\n]*", Comment.Special),
            (r"^# \[Stage 3[^\n]*", Generic.Heading),
            (r"^# \[Active Learning\][^\n]*", Keyword.Type),
            (r"^# \[Optional\][^\n]*", Comment.Single),
            inherit,
        ],
    }
