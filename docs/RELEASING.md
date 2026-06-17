# Releasing openfusion

Pushing a `v*` tag runs `.github/workflows/release.yml`, which:

1. builds the wheel + sdist,
2. builds and pushes the Docker image to GHCR (`ghcr.io/<owner>/openfusion`),
3. publishes to PyPI **if** the `PYPI_API_TOKEN` secret is set (skipped otherwise),
4. creates a GitHub Release with the wheel attached and auto-generated notes.

## One-time setup: PyPI publishing

1. **Check the name is free.** Visit <https://pypi.org/project/openfusion/>. If it's taken, pick a
   new distribution name (e.g. `openfusion-proxy`) and update `name = "..."` in `pyproject.toml`
   plus the install commands in the README. (The `openfusion` *command* can stay the same.)
2. **Create a PyPI account** and an **API token** at <https://pypi.org/manage/account/token/>.
   Scope it to the whole account for the first publish (you can re-scope to the project afterward).
3. **Add the token as a repo secret:** GitHub → repo **Settings → Secrets and variables → Actions →
   New repository secret** → name `PYPI_API_TOKEN`, value `pypi-…`.

> More secure alternative: [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) via
> OIDC (no token). If you use it, drop the `password:` line from the publish step and configure the
> publisher on PyPI for this repo + `Release` workflow.

## Cut a release

```bash
# from an up-to-date main
git checkout main && git pull
git tag -a v0.1.0 -m "openfusion v0.1.0"
git push origin v0.1.0
```

Then watch **Actions → Release**. When it's green you'll have:

- `pip install openfusion` / `uvx openfusion` (if the PyPI token was set),
- `docker run -p 8000:8000 ghcr.io/<owner>/openfusion:v0.1.0`,
- a GitHub Release at `…/releases/tag/v0.1.0`.

## Bumping versions

Update `version` in `pyproject.toml` and add a dated section to `CHANGELOG.md`, then tag
`vX.Y.Z` to match. (PyPI rejects re-uploading an existing version, so bump before re-tagging.)
