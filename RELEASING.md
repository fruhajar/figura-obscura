# Releasing Figura Obscura on itch.io

Everything here runs on a **host** machine, not in the dev container: `ort`
downloads a prebuilt ONNX Runtime for the *host* triple, so each platform's
binaries must be built on that platform. There is no cross-compilation path.

---

## 0. One-time setup

### Pin the model checksums

The registry ships with empty `sha256` fields, because no model host was
reachable from the build container (`crates/ob-core/src/registry.rs`). **Pin them
before the first paid release** — an unpinned model means a corrupted or
substituted download is only caught by the ONNX header sniff, not by a digest.

```sh
cargo run --release -p ob-cli -- setup --all
# each download prints:  sha256 = <digest>
```

Paste each digest into its entry's `sha256`, then confirm:

```sh
cargo run --release -p ob-cli -- models verify nudenet-320n   # "checksum OK"
```

> Never pin a digest computed inside the dev container. GitHub's release-asset
> host answers there with a sign-in page, so the digest would be the hash of an
> HTML document.

### Decide how ffmpeg reaches the user

Obscura spawns ffmpeg as a child process and never links libav, so **FFmpeg's licence
does not reach Figura Obscura's own code** in any of these options. Pick one:

1. **Don't bundle it.** Omit `--ffmpeg` from the build. No obligations at all;
   the app finds a user-installed ffmpeg and shows an install command when it
   cannot. Images work with no ffmpeg present; only video needs it.
2. **Bundle an LGPL build** (the default). Lightest obligations of the bundling
   options.
3. **Bundle a GPL build** with `--allow-gpl`. Also fine — you must publish the
   corresponding FFmpeg source for that exact build.

A `--enable-nonfree` build is refused unconditionally: nobody may redistribute
it. `packaging/common/check-ffmpeg-licence.sh` enforces all of this.

| Platform | Where |
|---|---|
| Windows | <https://github.com/BtbN/FFmpeg-Builds> — an asset with `lgpl` in the name |
| macOS | build it: `./configure --disable-gpl --disable-nonfree` |
| Linux | most distro builds are GPL; build LGPL, or omit `--ffmpeg` and let the tarball use the system ffmpeg |

If you bundle, drop the licence text at `packaging/common/licenses/` and record
the upstream tag the build came from — you will need it for the source offer.

### itch.io project

Create the project, set it to **paid**, and add the platform tags matching the
butler channels (`windows`, `linux`, `osx`). Install
[butler](https://itch.io/docs/butler/) and `butler login`.

---

## 1. Bump the version

One place — `[workspace.package] version` in `Cargo.toml`. Every build script
reads it from there, so the installer filename, the window title, the About page
and the itch build number cannot disagree.

---

## 2. Build each platform

### Windows (on Windows, with Inno Setup 6 installed)

```powershell
.\packaging\windows\build.ps1 -FfmpegDir C:\ffmpeg-lgpl\bin
```

Produces `target\installer\FiguraObscura-<version>-windows-x64-setup.exe`.

**Sign it.** An unsigned installer triggers SmartScreen's "Windows protected
your PC" on every buyer's machine, which reads as malware, not as a warning:

```powershell
signtool sign /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 /a `
    target\installer\FiguraObscura-<version>-windows-x64-setup.exe
```

### Linux

```sh
packaging/linux/build.sh --ffmpeg /path/to/lgpl-ffmpeg/bin
```

Produces a `.tar.gz` (with `install.sh`) and an `.AppImage` in `target/dist/`.

### macOS (on macOS)

```sh
packaging/macos/build.sh --ffmpeg /path/to/lgpl-ffmpeg/bin --universal \
    --sign "Developer ID Application: Your Name (TEAMID)" \
    --notarize-profile figura-obscura
```

**Notarise it.** Without notarisation macOS reports a downloaded app as
"damaged and can't be opened" — indistinguishable from a broken product. Set the
keychain profile up once:

```sh
xcrun notarytool store-credentials figura-obscura \
    --apple-id you@example.com --team-id TEAMID --password <app-specific-password>
```

---

## 3. Smoke-test each build on a clean machine

Not the build machine. The most common release bug is a dependency that happens
to be installed on the developer's box.

- [ ] Installer runs without admin rights and creates working shortcuts
- [ ] First launch shows the setup screen and downloads the models
- [ ] A folder of images processes end to end
- [ ] **A video processes** — this is what proves the bundled ffmpeg was found;
      About → Locations shows which `ffmpeg` is in use
- [ ] Stop cancels a running batch and leaves no truncated output
- [ ] Uninstall removes the app and leaves the model cache alone

---

## 4. Publish

```sh
packaging/itch/push.sh --user YOURNAME --game figura-obscura --dry-run   # check the plan
packaging/itch/push.sh --user YOURNAME --game figura-obscura
```

Linux and macOS are pushed as *directories*, not archives — butler diffs file
trees, so an archive would re-upload in full every release.

---

## 5. Store-page checklist

- [ ] Screenshots of the Batch page with a preview, and the Models page
- [ ] State plainly that models are downloaded on first run (~56 MB) and that
      **nothing the user processes leaves their machine**
- [ ] System requirements: 64-bit Windows 10+/macOS 11+/glibc 2.31+; ~200 MB
      disk plus models
- [ ] If ffmpeg is bundled, link the source for that exact build; if it is not,
      say so and name the install command
- [ ] Credit the model authors (the About page does; the store page should too)

## Known gaps

- **GPU builds are untested.** No GPU was available during development. Ship the
  CPU build as the default; treat a CUDA build as an optional extra download and
  test it on real hardware first. `--gpu rocm` is a trap on Linux — see
  `HOST-BUILD.md`.
- **Model recall across art styles is unmeasured.** No weights were downloadable
  in the build container, so detection quality has never been observed. Run a
  representative set through before promising anything on the store page.
