# Applying the round-2 review response

`git push` and the GitHub API were both refused for this session (git proxy 403;
`create_branch` → *"Resource not accessible by integration"*), so the six commits
are delivered as a git bundle instead of a pushed branch. The bundle is a
complete, verified pack — applying it gives you the identical commits.

```sh
git bundle verify review-round-2.bundle          # base is 6424c81 (main)
git fetch review-round-2.bundle review-round-2:review-round-2
git switch review-round-2
```

Then push and open the PR from a machine with write access:

```sh
git push -u origin review-round-2
```

## Build

There is no venv or TeX in a fresh clone. To reproduce:

```sh
python -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
./build_derivation.sh        # pdflatex x4 + bibtex x2; reports the three counts
.venv/bin/python -m pytest -q # 68 passed
```

`build_derivation.sh` runs **bibtex twice** on purpose: `derivationNotes.bib`
stores footnote text containing `\cite`, so citations inside notes only reach the
`.aux` after the `.bbl` has been typeset once. With a single pass
`Kraichnan:1959` is undefined.

## One thing needs an author

The LISA noise-model reference could not be given a `bib.bib` entry: egress to
`inspirehep.net` and `arxiv.org` is blocked by network policy on the machine this
ran on, so it could not be verified against a primary source, and the house rule
is not to guess bibliography data. The model is written out in full in
`Notebooks/k_break_frequency.py` and a `\why{}` note in §`sec:k-break-hz` marks
the spot. It is the Robson–Cornish–Liu / LISA SciRD analytic fit.
