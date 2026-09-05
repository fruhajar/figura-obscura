Offline batch censoring for images and video — GUI and CLI. Nothing on the
processing path touches the network; detector models are downloaded once, on
first run.

## Install

**Windows** — run `FiguraObscura-*-windows-x64-setup.exe`. It installs to
Program Files, adds a Start menu entry, offers to put `obscura.exe` on your
`PATH`, and downloads the detection models on its last page.

The installer is **not code-signed**, so SmartScreen will interrupt you:
*More info → Run anyway*.

**Linux** — either the tarball:

```sh
tar -xzf FiguraObscura-*-linux-x86_64.tar.gz
cd FiguraObscura-*-linux-x86_64
./install.sh          # installs to ~/.local, no root; --prefix for elsewhere
```

or the `.AppImage`, which needs no install — `chmod +x` and run it.

**macOS** — open the `.dmg`, drag the app to Applications. It is **not signed or
notarised**, so the first launch must be *right-click → Open*; a double-click
will be refused by Gatekeeper. Built for Apple Silicon.

## Two things to know

**ffmpeg is not bundled.** Video needs `ffmpeg` and `ffprobe` on your `PATH` —
install them from your package manager, or from ffmpeg.org. Images work without
them. Bundling ffmpeg obliges shipping its licence text and offering the
corresponding source, so these builds leave that choice to you; the app tells
you exactly where it looked when it cannot find one.

**Models download on first run**, roughly 56 MB, and only then. `obscura setup`
does it from the command line; the desktop app offers it on first launch. After
that the tool never opens a socket.

## Not yet verified

- **GPU execution has never been run on real hardware.** These are CPU builds,
  which work everywhere and are what the installers ship.
- **Detection quality across art styles is unmeasured.** The model choices are
  reasoned from published class lists and F1 figures, not from observed recall.
  Try it on your own material before relying on it.
