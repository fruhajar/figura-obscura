# Bundled-component license texts

This directory ships verbatim inside every installer and archive: `stage.sh`
copies its contents to `licenses/` in the staged tree. It is tracked in git
**only** because of this file — git does not track empty directories, so
without it a fresh clone would not have the directory at all, and that is
precisely how the release before this one came to have no license text in it.

The texts themselves are not committed, because none of them may be reproduced
from memory: a license must be byte-exact. Fetch them on a connected machine
before building a release that bundles ffmpeg.

| File | When it is required | Source |
|---|---|---|
| `LGPL-2.1.txt` | Bundling an LGPL ffmpeg (`stage.sh --ffmpeg DIR`, the default and recommended path) | <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt> |
| `GPL-2.0.txt` | Bundling a GPL ffmpeg (`stage.sh --ffmpeg DIR --allow-gpl`) | <https://www.gnu.org/licenses/old-licenses/gpl-2.0.txt> |
| `GPL-3.0.txt` | Only if that GPL ffmpeg was configured `--enable-version3` | <https://www.gnu.org/licenses/gpl-3.0.txt> |

```sh
curl -fsSLO --output-dir packaging/common/licenses \
    https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
```

`stage.sh` now **fails** rather than warns when the text for the ffmpeg you are
bundling is absent. Shipping an LGPL or GPL binary without its license text is a
distribution-compliance failure, not a cosmetic omission, and it is invisible in
the finished artifact — so it has to be caught at build time.

Not bundling ffmpeg at all (omit `--ffmpeg`) carries no obligation whatsoever
and needs nothing in this directory. See `../THIRD-PARTY.md` for the full
reasoning, including why ffmpeg's copyleft does not reach Figura Obscura's own
source.
