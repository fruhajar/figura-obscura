# Releasing Figura Obscura

Releases are **tag-driven GitHub releases**, built by CI on all three platforms:

```sh
git push origin v0.5.1
```

`.github/workflows/release.yml` builds Windows, Linux and macOS on their own
runners and opens a **draft** release with the artifacts attached. Publishing is
a manual click, on purpose: the installers are unsigned, and a draft is the last
point at which that is still a private problem.

No storefront is involved. An earlier plan to sell on itch.io was dropped, and
`--version`, the About page, the release notes and this document are the whole of
the distribution story.

---

## 0. One-time setup

### Pin the model checksums

The registry ships with four empty `sha256` fields
(`crates/ob-core/src/registry.rs`); only `nudenet-320n` is pinned. An unpinned
model means a corrupted or substituted download is caught only by the ONNX header
sniff, not by a digest. That is a supply-chain property of shipping a downloader,
not a storefront requirement, so it stays on this list:

```sh
cargo run --release -p ob-cli -- setup --all
# each download prints:  sha256 = <digest>
```

Paste each digest into its entry's `sha256`, then confirm:

```sh
cargo run --release -p ob-cli -- models verify nudenet-320n   # "checksum OK"
```

> Never pin a digest computed inside the dev container. GitHub's release-asset
> host answers there with a sign-in page, and HuggingFace's LFS CDN does not
> resolve, so the digest would be the hash of an HTML document. See
> [`HOST-BUILD.md`](HOST-BUILD.md).

### Decide how ffmpeg reaches the user

Obscura spawns ffmpeg as a child process and never links libav, so **FFmpeg's
licence does not reach Figura Obscura's own code** in any of these options. Pick
one:

1. **Don't bundle it.** Omit `--ffmpeg` from the build. No obligations at all;
   the app finds a user-installed ffmpeg and shows an install command when it
   cannot. Images work with no ffmpeg present; only video needs it. **This is
   what CI does**, so it is what every release so far has shipped.
2. **Bundle an LGPL build.** Lightest obligations of the bundling options.
3. **Bundle a GPL build** with `--allow-gpl`. Also fine, but you must publish the
   corresponding FFmpeg source for that exact build.

A `--enable-nonfree` build is refused unconditionally: nobody may redistribute
it. `packaging/common/check-ffmpeg-licence.sh` enforces all of this, and
`build.ps1` reimplements the same check in PowerShell.

| Platform | Where to get an LGPL build |
|---|---|
| Windows | <https://github.com/BtbN/FFmpeg-Builds> — an asset with `lgpl` in the name |
| macOS | build it: `./configure --disable-gpl --disable-nonfree` |
| Linux | most distro builds are GPL; build LGPL, or omit `--ffmpeg` and let the tarball use the system ffmpeg |

If you bundle, drop the licence text at `packaging/common/licenses/` and record
the upstream tag the build came from. You need it for the source offer.

---

## 1. Bump the version

One place: `[workspace.package] version` in `Cargo.toml`. Every build script
reads it from there, so the installer filename, the window title, the About page
and the artifact names cannot disagree.

Three copies are **not** derived from it and need updating by hand:

- the `#ifndef AppVersion` fallback in `packaging/windows/figura-obscura.iss`
- the rustdoc examples in `crates/ob-core/src/version.rs`
- the `--version` output quoted in [`HOST-BUILD.md`](HOST-BUILD.md)

The release workflow **fails a tag that disagrees with `Cargo.toml`**, in a job
that runs before the three builds, so a mismatch costs you four seconds rather
than three runners. The bump commit therefore has to land before the tag.

## 2. Tag and push

```sh
git tag -a v0.5.1 -m "Figura Obscura 0.5.1"
git push origin v0.5.1
```

That is the whole release build. Nine minutes later there is a draft release
with four assets: the Windows `setup.exe`, the Linux `.tar.gz` and `.AppImage`,
and the macOS `.dmg`.

What CI produces, and how it differs from a local build:

- **No bundled ffmpeg** on any platform, per option 1 above.
- **CPU execution provider only.** GPU builds are a separate, manual artifact;
  see the known gaps below.
- **The macOS build is Apple Silicon only**, unsigned and un-notarised. A
  universal build needs `packaging/macos/build.sh --universal` on a Mac.
- **The Windows installer is unsigned.**

`fail-fast: false`, so one platform failing still leaves you the other two as
workflow artifacts. But the `release` job `needs` all three, so a single failure
means it is **skipped** and no draft appears at all. Fix the cause and re-run via
`workflow_dispatch` with the tag; the release step creates the release only if it
is missing and uploads with `--clobber`, so re-running is safe and idempotent.

> **The Windows staging list and the `.iss` must agree, and nothing makes them.**
> `packaging/stage.sh` stages the Unix builds; `build.ps1` reimplements it in
> PowerShell because Windows has no bash. Adding a `Source:` line to
> `figura-obscura.iss` without the matching `Copy-Item` broke every Windows
> release build for 25 days, with Linux and macOS green throughout. `build.ps1`
> now asserts over the files the installer requires, which is the guard that was
> missing.

## 3. Release notes

The draft's body is `.github/release-notes.md`, read from the **tag's** tree, so
edit it before tagging rather than after. What has to be in it:

- [ ] What changed, with figures where a claim is measurable
- [ ] That models are downloaded on first run (~56 MB) and that **nothing the
      user processes leaves their machine**
- [ ] That ffmpeg is not bundled, what needs it (video, not images), and that the
      app says where it looked
- [ ] The unsigned and un-notarised status, with the exact click-through each
      platform demands. SmartScreen's "Windows protected your PC" and Gatekeeper's
      "damaged and can't be opened" both read as malware or breakage rather than
      as a warning, so saying so first is the difference between a caveat and a
      bug report.
- [ ] What is still unverified: GPU execution has never run on real hardware, and
      detection quality across art styles is unmeasured
- [ ] Credit to the model authors (the About page does; the notes should too)

## 4. Smoke-test the draft's artifacts

On a machine that is not the build machine. The most common release bug is a
dependency that happens to be installed on the developer's box.

- [ ] Installer runs without admin rights and creates working shortcuts
- [ ] First launch shows the setup screen and downloads the models
- [ ] A folder of images processes end to end
- [ ] **A video processes.** This is the one that matters most, because it is
      what proves the ffmpeg lookup works on a machine that never built the
      thing. About → Locations shows which `ffmpeg` is in use.
- [ ] Stop cancels a running batch and leaves no truncated output
- [ ] Uninstall removes the app and leaves the model cache alone

Then publish the draft.

## 5. Signing, when it becomes worth it

Not required to publish, and not done for any release so far. It is what removes
the two click-throughs above.

**Windows** needs a code-signing certificate:

```powershell
signtool sign /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 /a `
    target\installer\FiguraObscura-<version>-windows-x64-setup.exe
```

**macOS** needs a Developer ID and one-time keychain setup, after which the build
script does both steps:

```sh
xcrun notarytool store-credentials figura-obscura \
    --apple-id you@example.com --team-id TEAMID --password <app-specific-password>

packaging/macos/build.sh --universal \
    --sign "Developer ID Application: Your Name (TEAMID)" \
    --notarize-profile figura-obscura
```

Both require a build on the platform itself: `ort` downloads a prebuilt ONNX
Runtime for the *host* triple, so there is no cross-compilation path. Per-platform
commands are in [`HOST-BUILD.md`](HOST-BUILD.md).

## Known gaps

- **GPU builds are untested.** No GPU was available during development. CI ships
  the CPU build; treat a CUDA or webgpu build as an optional extra download and
  test it on real hardware first. `--gpu rocm` is a trap on Linux, see
  [`HOST-BUILD.md`](HOST-BUILD.md).
- **Model recall across art styles is unmeasured.** Only `nudenet-320n` has ever
  been run here, because no other weights were downloadable in the build
  container. Run a representative set through before promising anything about
  detection quality.
