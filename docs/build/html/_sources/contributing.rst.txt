Contributing
============

Development Installation
------------------------

.. code-block:: bash

   git clone https://github.com/rahulumrao/sparc.git
   cd sparc
   pip install -e ".[dev]"

This installs SPARC in editable mode so source changes take effect immediately.


Code Style
----------

- Follow **PEP 8**. Run ``flake8 sparc/`` before committing.
- Docstrings: **NumPy style** (compatible with ``sphinx.ext.napoleon``).

  .. code-block:: python

     def my_function(param1, param2):
         """Short one-line summary.

         Parameters
         ----------
         param1 : int
             Description of param1.
         param2 : str
             Description of param2.

         Returns
         -------
         bool
             Description of return value.
         """

- Type hints are encouraged for public API functions.


Building the Documentation
--------------------------

.. code-block:: bash

   pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints nbsphinx
   cd docs
   make html

Output goes to ``docs/build/html/``. Open ``index.html`` in a browser to preview.

Check for warnings:

.. code-block:: bash

   make html 2>&1 | grep -E "WARNING|ERROR"


Running the Tests
-----------------

.. code-block:: bash

   pytest docs/tests/ -v


Pull Request Checklist
----------------------

Before opening a PR:

- [ ] All existing tests pass (``pytest docs/tests/ -v``)
- [ ] New functionality has tests
- [ ] Docstrings updated for any changed public API
- [ ] RST docs updated if user-facing behaviour changed
- [ ] Documentation builds without new warnings: ``make -C docs html 2>&1 | grep WARNING``
- [ ] YAML examples in docs match the current ``input.yaml`` schema


Reporting Issues
----------------

Open an issue at https://github.com/rahulumrao/sparc/issues. Include:

- SPARC version (``sparc --version``)
- Python and DeePMD-kit versions
- Minimal reproducing ``input.yaml``
- Full traceback from ``sparc.log``
