## Code Standards
==============

Before pushing code to the development branch, please make sure your changes respect the following code standards.

## PEP 8
-----
Run your code through pylint to check that you're in compliance with [PEP 8](https://www.python.org/dev/peps/pep-0008/):
```bash
pylint <file name>
```

## Docstrings
----------
All new modules, classes, and methods should have [Sphinx style docstrings](https://pythonhosted.org/an_example_pypi_project/sphinx.html) describing what the code does and what its inputs are. These docstrings are used to automatically generate SPARC's documentation, so please make sure they're clear and descriptive.

## Tests
-----
New features must be accompanied by unit and integration tests written using [pytest](https://docs.pytest.org/en/latest/). This helps ensure the code works properly and makes the code as a whole easier to maintain.


## General workflow
----------------

Development should follow this pattern:

1. Create an issue on Github proposing a new example and make sure to describe what you want to contribute.
2. Fork the repo and create your branch from `main`.


## Pushing changes from a forked repo
----------------------------------

1. Fork the [SPARC repository](https://github.com/rahulumrao/sparc).
2. Set SPARC as an upstream remote
```bash
git remote add upstream https://github.com/rahulumrao/sparc
```
Before branching off, make sure that your forked copy of `main` branch is up to date
```bash
git fetch upstream
git merge upstream/main
```
If for some reason there were changes made on the `main` branch of your fork that you want to discard, you can force a reset:

```bash
git reset --hard upstream/master
```

3. Create a new branch with a name that describes the specific feature you want to work on:

```bash
git checkout -b <feature branch name>
```

4. While you're working, commit your changes periodically, and when you're done, push your branch to your fork:

```bash
git push -u origin <feature branch name>
```

5. When you go to Github, you'll now see an option to open a Pull Request for the topic branch you just pushed. Write a helpful description of the changes you made, and then create the Pull Request.
