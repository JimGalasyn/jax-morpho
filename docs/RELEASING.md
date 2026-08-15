# Releasing

> **Cross-repo release failure modes live in the `publish-release` skill**
> (`claude-shared/skills/publish-release`). It carries what has bitten across all of
> these repos — green workflows that published nothing, PyPI propagation lying in both
> directions, the DOI commit that the *next* release's changelog check always flags, and
> the one-time setup steps that cannot be undone. **This file keeps what is specific to
> this repo**; the two are deliberately not copies of each other, because duplicated
> process drifts.

1. Bump `version` in `pyproject.toml`, `src/jax_morpho/__init__.py`, and
   `CITATION.cff` (keep them in sync).
2. Commit, tag `vX.Y.Z`, push the tag.
3. Publish a GitHub Release for the tag — this triggers
   `.github/workflows/publish-pypi.yml` (OIDC trusted publishing to PyPI).

## First publish (one-time)

- Add a **pending publisher** on pypi.org (Project `jax-morpho`, Owner
  `JimGalasyn`, Repo `jax-morpho`, Workflow `publish-pypi.yml`, Environment
  `pypi`), and create a GitHub Environment named `pypi`.
- For the DOI badge, connect the repo to Zenodo before the first release so it
  mints a concept DOI; add it to `README.md`, `CITATION.cff`, `.zenodo.json`.
- Add the repo to Codecov and set the `CODECOV_TOKEN` secret.
