# TimeGym publication strategy

Updated 13 November 2026. The author chose a new manuscript for the redesigned package and, on 6 November 2026, to post it as version 2 of [arXiv:2105.01404](https://arxiv.org/abs/2105.01404) rather than as a new submission. The replacement has not been posted yet, and the manuscript has not been submitted to a venue. The package is on PyPI (1.4.0).

**Plan decided on 13 November 2026.** Version 2 will be posted with the corrections and additions of that date: a capability research question, the statement of which results hold by construction, the exploratory answer-key alarm and the explanations of the random-walk and trend results (recorded under protocol 2.1 in `experiments/results-v2-followup/`), the suite-level global test and the paired answer-key check (protocol 2.2, in `experiments/results-v2-addendum/`), and the added related work. A study of generated mutants of the lag-Ridge code and a study of one nonlinear model, where the shift relations hold only approximately, are kept for the peer-reviewed version; that version also recasts the Limitations as threats to validity (construct, internal, external, conclusion). The target is an empirical software-engineering venue, with a submission of the package to the Journal of Open Source Software.

## Posting version 2

1. Pick the commit to post, run `python paper/build_standalone.py --check`, compile `paper/timegym.tex` as described in the [build guide](README.md), and record the commit hash.
2. On the arXiv user page, choose **Replace** for 2105.01404. Do not start a new submission.
3. Upload `paper/timegym.tex` alone. It contains the figures (TikZ) and the bibliography, so arXiv needs no other file. Check arXiv's compiled preview against the local PDF.
4. Metadata: title "TimeGym: What Can Your Forecaster Learn, and What Is Broken?" (without the `\newline`); the abstract pasted from the manuscript, which is plain text and must stay under arXiv's 1,920 characters; the same author; primary category cs.LG. A cs.SE cross-list fits the verification framing.
5. Comments: say what changed, for example "v2: rewritten for TimeGym 1.4 (https://github.com/diogoseca/timegym), with corrected reference semantics, redesigned software and two new controlled studies; v1 (2021) described the original prototype", followed by the page and figure counts.
6. License: keep CC BY 4.0, the license of version 1.
7. Once version 2 is announced, update `CITATION.cff` in one pull request: the reference takes the new title, year 2026 and the `v2` URL, and keeps the same DOI. Update the README sentence and the citation label in `tools/sync_citation.py` in the same pull request. The arXiv DOI has no version, so from announcement onwards it resolves to version 2 and must not stay paired with the 2021 title. [arXiv DOIs](https://info.arxiv.org/help/doi.html).
8. When a venue publishes the paper, add its DOI and journal reference to the arXiv record, which needs no new version, and cite the venue DOI from then on.

The [claim ledger](claims.md) links each claim to its evidence and exclusions; check the abstract against it before posting.
