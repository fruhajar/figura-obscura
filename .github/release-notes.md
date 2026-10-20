Offline batch censoring for images and video — GUI and CLI. Nothing on the
processing path touches the network; detector models are downloaded once, on
first run.

## What changed since 0.5.1

- **A new Edit page censors one image by hand**, for what no detector will
  find: a licence plate, a face in the background, a region the model missed.
  Drag boxes onto the picture, give each one its own style and corner rounding
  from the same picker the Tuning page uses, then save the result to a new
  file. Ctrl-click adds boxes to the selection, so a style change lands on all
  of them at once; Ctrl-Z and Ctrl-Y undo and redo, and a whole drag is one
  step rather than sixty. A hand-drawn box goes through the same censor code as
  a detected one, so what the canvas shows is what the pixels become.
- **Video says where it has got to while it runs.** The activity list shows
  `frame 1234 / 5400` beside the file, the CLI puts the same counter in its
  progress bar, and the bar and ETA both move with it instead of waiting for
  the file to finish. On a batch of one long clip the bar used to sit at 0%
  from start to end, which is exactly where a 4K video legitimately taking tens
  of minutes cannot be told apart from a hang.
- **The estimate holds up on long clips at a low detection rate.** A video
  frame's inference and its decode, composite and encode are now costed
  separately: the first as a step function of frame size, the second by area.
  Counting the skipped frames as free made a clip at one detection in thirty
  look an order of magnitude cheaper than it was.

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
