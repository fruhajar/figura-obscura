Offline batch censoring for images and video, with a GUI and a CLI over the same
engine. Nothing on the processing path touches the network; detector models are
downloaded once, on first run.

**This is the last planned release.** The feature set is complete and the
project is closed to new work. The source stays up, and the tool keeps working:
it has no server behind it and nothing to expire.

## What changed since 0.6.0

- **Review before writing turns a batch into two passes.** The detector runs
  over everything and stops, writing nothing. The new **Review** tab then walks
  you through what it found, on the same canvas the Edit page uses, where you
  can drag a box, resize it, delete a false positive or draw one over a region
  the model missed. **Write** commits the result with no detector involved.
  Files the detector found *nothing* in are listed too, and called out: those
  are where a miss goes unnoticed, and a tool like this one fails quietly by
  design. Only the geometry is reviewed, so changing the censor style
  afterwards re-paints without another detection pass.

  Both halves resolve their boxes through the same call, so the canvas is not
  an impression of the output: it is the output, with the pixels still
  reversible. The cost is decoding twice, which is why it is off by default.

  Video review is **experimental**. A slider steps between the frames the
  detector actually ran on (one in `detect_every`), and a correction there
  carries through the frames either side of it via the tracker, exactly as a
  detection would. The seek is keyframe-accurate rather than frame-accurate, so
  the picture beside the boxes is approximate.
- **A Settings page, and a theme that is actually legible in both modes.**
  Light, dark, or whatever the desktop asks for, plus a choice of six accent
  colours. The page shows the controls the palette has to carry (buttons,
  pills, a progress bar, a slider) so a scheme can be judged in one place
  rather than by switching pages to find what went wrong. The choice is
  per-machine and saved with your other settings, so it does not travel when
  you export a profile. Settings files written by earlier versions still load.
- **Fixed:** the GUI tests required ffmpeg to be installed, so a checkout
  tested on a machine without it failed for a reason that had nothing to do
  with the change under test.

## Install

**Windows.** Run `FiguraObscura-*-windows-x64-setup.exe`. It installs to Program
Files, adds a Start menu entry, offers to put `obscura.exe` on your `PATH`, and
downloads the detection models on its last page.

The installer is **not code-signed**, so SmartScreen will interrupt you:
*More info*, then *Run anyway*.

**Linux.** Either the tarball:

```sh
tar -xzf FiguraObscura-*-linux-x86_64.tar.gz
cd FiguraObscura-*-linux-x86_64
./install.sh          # installs to ~/.local, no root; --prefix for elsewhere
```

or the `.AppImage`, which needs no install at all: `chmod +x` and run it.

**macOS.** Open the `.dmg` and drag the app to Applications. It is **not signed
or notarised**, so the first launch must be *right-click, then Open*; a
double-click will be refused by Gatekeeper. Built for Apple Silicon.

## Two things to know

**ffmpeg is not bundled.** Video needs `ffmpeg` and `ffprobe` on your `PATH`.
Install them from your package manager, or from ffmpeg.org. Images work without
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
  Try it on your own material before relying on it. `obscura process --dry-run
  --report audit.csv` is the quick way to see what fires on your own files.

## Credit

The detectors are other people's work, downloaded at run time and not
redistributed here. NudeNet is by
[notAI-tech](https://github.com/notAI-tech/NudeNet) (Apache-2.0); the anime
detectors are [deepghs'
`anime_censor_detection`](https://huggingface.co/deepghs/anime_censor_detection)
(MIT). `obscura models list` prints the licence and source of each, and the
About page says the same.
