# Publishing to PyPI

1. Ensure version is updated in `pyproject.toml`.
2. Build the distribution:
   ```bash
   python -m pip install --upgrade build twine
   python -m build
   ```
   This creates `dist/robbuffet-*.tar.gz` and `dist/robbuffet-*.whl`.
3. Upload to PyPI:
   ```bash
   twine upload dist/*
   ```
4. Verify install:
   ```bash
   pip install --no-cache-dir avocet-cp
   ```

For TestPyPI, replace the upload command with:
```
twine upload --repository testpypi dist/*
```
