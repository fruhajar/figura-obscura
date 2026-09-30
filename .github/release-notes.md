Offline batch censoring for images and video — GUI and CLI. Nothing on the
processing path touches the network; detector models are downloaded once, on
first run.

## What changed since 0.1.0

- **Image batches run roughly twice as fast.** The release profile had been
  optimising for size, which gives up exactly what this program spends its time
  in: pixel loops and decompression. A 20-image batch goes from 1.37 s to
  0.62 s, a 10-second 1080p video from 6.88 s to 4.46 s. It costs 3.3 MB of
  binary. Letterboxing a 4K frame for the detector is 22× faster, and the
  censor renderers 3–7×.
- **The detector sizes its thread pool from physical cores**, not logical ones.
  Giving compute-bound inference twice the threads it wants was measurably
  slower than leaving it alone.
- **Long GIFs no longer grow without bound.** The palette filter graph held
  every frame in memory until end of input, so a long clip grew until the
  machine swapped, which is what looked like a hang. Frames now pass through a
  lossless intermediate; the output is byte-identical, and peak memory on a
  600-frame clip drops from 844 MB to 170 MB.
- **The window stays responsive with very large file lists.**
- **`--version` and the About page name the execution provider actually in
  use**, so a silent CPU fallback cannot pass for a GPU run.
- **The output codec follows the container you chose**, rather than the
  container being pushed around by the codec.
- **An installed GPU build finds its own runtime libraries** (Linux). Before
  this, a webgpu install did not start at all and a cuda install fell back to
  CPU without saying so.
- The Windows installer's final page opens the copy of the app it just
  installed, not whichever one was already on `PATH`.

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
