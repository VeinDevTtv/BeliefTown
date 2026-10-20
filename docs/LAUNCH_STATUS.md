# Launch verification

Verified locally on Windows, Python 3.13.1, October 19, 2026.

- Restored the 127-test regression suite, covering provenance, belief updates, constraints, bounded search, tampered audits, replay, CLI, notebooks, and browser service behavior. All 127 passed.
- Restored CI for Linux Python 3.10/3.12/3.13 and Windows 3.13, with notebook execution on Linux 3.13. Install development checks with `python -m pip install -e ".[all,dev]"`.
- All four research notebooks executed with zero errors. The checked-in executed notebooks remain the release reference outputs; this run did not retune published checkpoints or benchmark data.
- A new clean virtual environment installed the editable core package with NumPy only. Doctor verified the canonical scene. Solve, verify, and replay passed in the working environment.
- The genuine 28-second launch recording uses the local Python HTTP backend and the current UI, with burned-in captions. It shows the one-message solution and matched replay. It is separate from the older 60-second sequence of still captures.
- Public recorded showcase: https://www.aitbourich.com/belieftown/demo/
- Article: https://www.aitbourich.com/blog/belieftown-npcs-can-be-wrong
- The article includes app imagery, a video, local theme-aware diagrams, native math, explicit bounded-minimum wording, benchmark latency and small OOD sample caveats, and links to the source research artifacts.
- Article HTML serves its own title, description, canonical URL, 1200 by 630 share image, and summary_large_image metadata without JavaScript.
- Public article, showcase, image, and MP4 returned HTTP 200. At 390px the article had no page overflow; complex figures scroll inside the page. Desktop and mobile responsive layouts were inspected through the shared Chromium preview. Physical-phone, Firefox, Safari, and LinkedIn preview checks remain unverified.

## Before posting from your account

1. Open LinkedIn Post Inspector and inspect the article URL to refresh and review the cached share preview.
2. Watch the MP4 on your phone and check captions and readability. Upload `artifacts/walkthrough/BeliefTown_Launch.mp4` directly to LinkedIn.
3. Pin the repository on your GitHub profile if desired.
4. Review the launch text in `docs/LINKEDIN_POST.md`, add the video, and publish from your account.

The v1.0.0 release assets are historical snapshots and have not been silently replaced. For this launch use the current main commit and the new launch recording. Expanded scaling experiments, a new independent split, game-engine integration, and user studies remain future work. The published benchmark is a finite synthetic evaluation, not a game usability study or universal correctness proof.
