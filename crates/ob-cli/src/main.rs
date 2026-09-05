//! `obscura` — the Figura Obscura command-line interface.
//!
//! The CLI is a thin front-end over `ob-job`: it builds a [`Profile`], resolves
//! the chosen model + its per-model settings (validated against the registry
//! metadata, so tooltips/ranges live in one place — R7), constructs the ONNX
//! detector, and runs the batch engine.

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use ob_core::cancel::CancelToken;
use ob_core::profile::Profile;
use ob_core::registry::{human_bytes, ModelEntry};
use ob_core::settings::{SettingValue, SettingValues};
use ob_detect::ensemble::EnsembleDetector;
use ob_detect::Detector;
use ob_job::expand::InputSpec;
use ob_job::{run, JobConfig, ProgressEvent};
use ob_media::video::VideoEncodeOpts;
use ob_models::{DownloadProgress, FetchOptions, FetchOutcome, ModelStatus};
use ob_track::TrackConfig;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "obscura",
    about = "Figura Obscura — offline batch censoring for images and video"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Process files/directories, writing censored output to a directory.
    Process(ProcessArgs),
    /// Manage detector models.
    #[command(subcommand)]
    Models(ModelsCmd),
    /// Save or load job profiles.
    #[command(subcommand)]
    Profile(ProfileCmd),
    /// First-run setup: download the models Obscura needs to work.
    ///
    /// This is what the installers run after copying files, and what to run by
    /// hand on a machine that was installed offline.
    Setup(SetupArgs),
    /// Censor a single file and write the result (no batch).
    Preview {
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long)]
        profile: Option<PathBuf>,
        #[arg(long, default_value = "nudenet-320n")]
        model: String,
    },
}

#[derive(Parser)]
struct ProcessArgs {
    /// Input files and/or directories.
    #[arg(required = true)]
    inputs: Vec<PathBuf>,
    /// Output directory (structure is mirrored from input dirs).
    #[arg(short, long)]
    output: PathBuf,
    /// Model id (see `obscura models list`).
    #[arg(long, default_value = "nudenet-320n")]
    model: String,
    /// Load a saved profile as the base configuration.
    #[arg(long)]
    profile: Option<PathBuf>,
    /// Override a model setting, e.g. `--set conf_threshold=0.3` (repeatable).
    #[arg(long = "set", value_name = "KEY=VALUE")]
    settings: Vec<String>,
    /// Recurse into subdirectories.
    #[arg(long, default_value_t = true)]
    recursive: bool,
    /// Only include paths matching these globs (repeatable).
    #[arg(long)]
    include: Vec<String>,
    /// Exclude paths matching these globs (repeatable).
    #[arg(long)]
    exclude: Vec<String>,
    /// List what would happen without writing output.
    #[arg(long)]
    dry_run: bool,
    /// Video: detect every Nth frame (tracker coasts between).
    #[arg(long, default_value_t = 3)]
    detect_every: u32,
    /// Cross-examine with another model as well (repeatable). Each model keeps
    /// its own default settings, so their published thresholds are respected.
    #[arg(long = "also-model", value_name = "ID")]
    also_models: Vec<String>,
    /// How many models must independently find a region before it is censored.
    /// 1 (the default) censors anything any model saw; higher demands consensus
    /// and will miss regions only one model found. Votes are counted per
    /// category among the models whose taxonomy covers it, so pairing a
    /// 3-class anime model with an 18-class one never deletes the 15
    /// categories only one of them can see.
    #[arg(long, default_value_t = 1)]
    min_votes: usize,
    /// Do not download a missing model; fail instead. Use in scripts and CI
    /// where an unexpected 100 MB transfer would be a surprise.
    #[arg(long)]
    no_auto_fetch: bool,
    /// Write a per-file CSV of how many regions were detected.
    ///
    /// The way to answer "is this model actually finding things on *my*
    /// material": combine with --dry-run to audit a folder without writing any
    /// output. Files with 0 regions are the ones to look at by eye — they are
    /// either clean or a miss, and only you can tell which.
    #[arg(long, value_name = "FILE")]
    report: Option<PathBuf>,
}

#[derive(Parser)]
struct SetupArgs {
    /// Download every model in the registry, not just the defaults.
    #[arg(long)]
    all: bool,
    /// Download only these model ids (repeatable). Overrides `--all`.
    #[arg(long = "model", value_name = "ID")]
    models: Vec<String>,
    /// Re-download models that are already present.
    #[arg(long)]
    force: bool,
    /// No progress bars — for installer logs and CI.
    #[arg(long)]
    quiet: bool,
}

#[derive(Subcommand)]
enum ModelsCmd {
    /// List built-in models with domain, license and download state.
    List,
    /// Show a model's settings and their tooltips.
    Show { id: String },
    /// Download a model into the local cache.
    Fetch {
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        force: bool,
    },
    /// Verify a downloaded model's checksum.
    Verify { id: String },
    /// Print the directory models are cached in.
    Path,
    /// Delete a downloaded model from the cache.
    Remove { id: String },
}

#[derive(Subcommand)]
enum ProfileCmd {
    /// Write a default profile to a file to edit.
    Save { path: PathBuf },
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Command::Process(args) => cmd_process(args),
        Command::Models(cmd) => cmd_models(cmd),
        Command::Setup(args) => cmd_setup(args),
        Command::Profile(cmd) => cmd_profile(cmd),
        Command::Preview {
            input,
            output,
            profile,
            model,
        } => cmd_preview(input, output, profile, model),
    }
}

/// Parse `--set key=value` pairs and coerce them against the model's metadata.
fn parse_settings(model_id: &str, pairs: &[String]) -> Result<SettingValues> {
    let entry =
        ob_core::registry::find(model_id).with_context(|| format!("unknown model `{model_id}`"))?;
    let mut out = SettingValues::new();
    for pair in pairs {
        let (k, v) = pair
            .split_once('=')
            .with_context(|| format!("bad --set `{pair}` (expected KEY=VALUE)"))?;
        let setting = entry
            .settings
            .iter()
            .find(|s| s.key == k)
            .with_context(|| format!("model `{model_id}` has no setting `{k}`"))?;
        // Coerce the string into the right SettingValue by the setting's kind.
        let raw = match &setting.kind {
            ob_core::settings::SettingKind::Float { .. } => SettingValue::Float(v.parse()?),
            ob_core::settings::SettingKind::Int { .. } => SettingValue::Int(v.parse()?),
            ob_core::settings::SettingKind::Bool => SettingValue::Bool(v.parse()?),
            ob_core::settings::SettingKind::Enum { .. } | ob_core::settings::SettingKind::Path => {
                SettingValue::Text(v.to_string())
            }
            ob_core::settings::SettingKind::Color => {
                bail!("color settings are not settable via --set yet")
            }
        };
        out.insert(k.to_string(), setting.coerce(raw)?);
    }
    Ok(out)
}

/// A progress bar wired to [`ob_models::fetch_with`].
///
/// The bar is created up front in "unknown length" mode: a server that sends no
/// `Content-Length` must still show motion, or a large download over a slow
/// link is indistinguishable from a hang.
fn download_bar(entry: &ModelEntry) -> ProgressBar {
    let bar = ProgressBar::new(entry.approx_bytes);
    bar.set_style(
        ProgressStyle::with_template(
            "  {msg:24} [{bar:32}] {bytes}/{total_bytes} {binary_bytes_per_sec}",
        )
        .unwrap_or_else(|_| ProgressStyle::default_bar())
        .progress_chars("=> "),
    );
    bar.set_message(entry.id.to_string());
    bar
}

/// Download `entry`, rendering a progress bar unless `quiet`.
fn fetch_one(entry: &ModelEntry, force: bool, quiet: bool) -> Result<FetchOutcome> {
    if quiet {
        println!("fetching {} ...", entry.id);
        let outcome = ob_models::fetch_with(entry, &FetchOptions::interactive().force(force))?;
        report_outcome(entry, &outcome);
        return Ok(outcome);
    }

    let bar = download_bar(entry);
    // The closure only touches the bar, which is Send + Sync, so it satisfies
    // the callback bound without any locking of our own.
    let on_progress = |p: DownloadProgress| {
        if let Some(total) = p.total {
            bar.set_length(total);
        }
        bar.set_position(p.downloaded);
    };
    let opts = FetchOptions::interactive()
        .force(force)
        .progress(&on_progress);
    let result = ob_models::fetch_with(entry, &opts);
    bar.finish_and_clear();
    let outcome = result?;
    report_outcome(entry, &outcome);
    Ok(outcome)
}

/// One line per model describing what the fetch did.
fn report_outcome(entry: &ModelEntry, outcome: &FetchOutcome) {
    match outcome {
        FetchOutcome::AlreadyPresent(p) => {
            println!("  {} — already installed ({})", entry.id, p.display());
        }
        FetchOutcome::Downloaded { path, bytes } => {
            println!(
                "  {} — downloaded {} to {}",
                entry.id,
                human_bytes(*bytes),
                path.display()
            );
            // Print the digest so an unpinned registry entry can be pinned by
            // pasting this value into its `sha256` field.
            match ob_models::sha256_file(path) {
                Ok(sum) => {
                    println!("    sha256 = {sum}");
                    if entry.sha256.is_empty() || entry.sha256 == "TODO" {
                        println!(
                            "    (not pinned in registry — set sha256 for `{}` to enable verification)",
                            entry.id
                        );
                    }
                }
                Err(err) => println!("    (could not hash file: {err})"),
            }
        }
    }
}

/// Build the detector for a model + settings, downloading the model if needed.
///
/// `ob_detect::build_detector` applies the resampling and tiling settings, so
/// what comes back may be a tiled multi-pass detector; either way it is used
/// only through the `Detector` trait.
fn build_detector(
    model_id: &str,
    overrides: &SettingValues,
    auto_fetch: bool,
) -> Result<Box<dyn Detector>> {
    let entry =
        ob_core::registry::find(model_id).with_context(|| format!("unknown model `{model_id}`"))?;

    let path = match ob_models::status(&entry) {
        ModelStatus::Installed { .. } => ob_models::require(&entry)?,
        ModelStatus::Missing if auto_fetch => {
            // Downloading on demand beats erroring with an instruction the user
            // then has to retype — but say what is happening and how big it is,
            // because it is the one moment Obscura touches the network.
            eprintln!(
                "model `{}` is not installed — downloading {} from {}",
                entry.id,
                human_bytes(entry.approx_bytes),
                entry.homepage
            );
            fetch_one(&entry, false, false)?.path().to_path_buf()
        }
        ModelStatus::Missing => bail!(
            "model `{}` is not installed. Run `obscura setup` (or `obscura models fetch --model {}`), \
             or drop --no-auto-fetch to download it now.",
            entry.id,
            entry.id
        ),
    };
    ob_detect::build_detector(&entry, overrides, path).context("loading detector")
}

/// Build the primary detector, plus any `--also-model` companions wrapped in an
/// ensemble. `overrides` came from `--set`, which is parsed against the primary
/// model; only keys a companion also declares are applied to it, so each
/// companion keeps its own published confidence threshold unless overridden.
fn build_ensemble(
    model_id: &str,
    also: &[String],
    min_votes: usize,
    overrides: &SettingValues,
    auto_fetch: bool,
) -> Result<Box<dyn Detector>> {
    // Validate the argument combination before loading anything, so a bad flag
    // reports itself rather than surfacing as a missing-model error first.
    //
    // Asking for more votes than there are models must fail loudly rather than
    // quietly mean "all of them": the ensemble clamps the threshold to the
    // number of members that can actually vote on a category, so an unchecked
    // `--min-votes 5` across two models would silently run as `2` and look like
    // it had honoured the request.
    let member_count = 1 + also.len();
    if min_votes > member_count {
        bail!(
            "--min-votes {min_votes} needs at least {min_votes} models; \
             this run has {member_count}. Add --also-model."
        );
    }
    for id in also {
        if id == model_id {
            bail!("--also-model `{id}` is already the primary model");
        }
    }

    let primary = build_detector(model_id, overrides, auto_fetch)?;
    if also.is_empty() {
        return Ok(primary);
    }
    let mut members = vec![primary];
    for id in also {
        members.push(build_detector(id, overrides, auto_fetch)?);
    }
    Ok(Box::new(EnsembleDetector::new(members, min_votes, 0.45)))
}

fn load_profile(path: &Option<PathBuf>, model_id: &str) -> Result<Profile> {
    let mut p = match path {
        Some(p) => Profile::from_json(&std::fs::read_to_string(p)?)
            .with_context(|| format!("parsing profile {}", p.display()))?,
        None => Profile::default(),
    };
    p.model_id = model_id.to_string();
    Ok(p)
}

fn cmd_process(args: ProcessArgs) -> Result<()> {
    let overrides = parse_settings(&args.model, &args.settings)?;
    let mut profile = load_profile(&args.profile, &args.model)?;
    profile.model_settings = overrides.clone();

    let detector = build_ensemble(
        &args.model,
        &args.also_models,
        args.min_votes,
        &overrides,
        !args.no_auto_fetch,
    )?;

    // Ctrl-C asks the run to stop between files rather than killing the
    // process mid-encode, which would leave a truncated video in the output
    // directory looking like a finished censored copy.
    let cancel = CancelToken::new();
    install_interrupt_handler(cancel.clone());

    let cfg = JobConfig {
        profile: &profile,
        input: InputSpec {
            inputs: args.inputs,
            recursive: args.recursive,
            include: args.include,
            exclude: args.exclude,
        },
        output_dir: args.output,
        dry_run: args.dry_run,
        detect_every: args.detect_every,
        track: TrackConfig::default(),
        video_opts: VideoEncodeOpts::default(),
        cancel: cancel.clone(),
    };

    let bar = ProgressBar::new(0);
    bar.set_style(
        ProgressStyle::with_template("{bar:40} {pos}/{len} {msg}")
            .unwrap_or_else(|_| ProgressStyle::default_bar()),
    );

    // Per-file detection counts, for --report and the miss summary. A Mutex
    // because images are processed in parallel and the closure must be Sync.
    let audit: std::sync::Mutex<Vec<(PathBuf, Option<usize>)>> = std::sync::Mutex::new(Vec::new());
    let verbose = args.dry_run;

    let progress = |ev: ProgressEvent| match ev {
        ProgressEvent::Discovered(n) => bar.set_length(n as u64),
        ProgressEvent::FileStarted(p) => bar.set_message(p.display().to_string()),
        ProgressEvent::FileDone { path, regions } => {
            bar.inc(1);
            // A dry run exists to be read, so print as it goes rather than
            // hiding the one number the run was for behind a progress bar.
            if verbose {
                bar.suspend(|| println!("{regions:>4}  {}", path.display()));
            }
            audit
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((path, Some(regions)));
        }
        ProgressEvent::FileError { path, error } => {
            bar.inc(1);
            eprintln!("error: {}: {error}", path.display());
            audit
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((path, None));
        }
        ProgressEvent::Cancelled { remaining } => {
            bar.abandon_with_message(format!("cancelled — {remaining} not processed"))
        }
        ProgressEvent::Finished { ok, failed } => {
            bar.finish_with_message(format!("done: {ok} ok, {failed} failed"))
        }
    };

    let summary = run(&cfg, detector.as_ref(), &progress)?;

    let audit = audit.into_inner().unwrap_or_else(|e| e.into_inner());
    if let Some(path) = &args.report {
        write_report(path, &audit)?;
        println!("wrote {} rows to {}", audit.len(), path.display());
    }
    // The headline number for a quality audit: a detector that fires on
    // everything and one that fires on nothing both "succeed" otherwise.
    let scanned = audit.iter().filter(|(_, r)| r.is_some()).count();
    if scanned > 0 {
        let empty = audit.iter().filter(|(_, r)| *r == Some(0)).count();
        let total: usize = audit.iter().filter_map(|(_, r)| *r).sum();
        println!(
            "detections: {total} region(s) across {scanned} file(s); \
             {empty} file(s) had none ({:.0}%)",
            100.0 * empty as f64 / scanned as f64
        );
        if empty > 0 {
            println!(
                "  Look at those {empty} by eye: they are either clean or misses. \
                 If they are misses, lower conf_threshold (--set conf_threshold=…) \
                 or try --model/--also-model."
            );
        }
    }

    if summary.cancelled {
        eprintln!(
            "stopped at your request: {} processed, {} not started",
            summary.ok, summary.skipped
        );
    }
    // A cancelled run is not a failed one — exit non-zero only for real errors,
    // so a Ctrl-C in a shell script does not read as a processing failure.
    if summary.failed > 0 {
        std::process::exit(1);
    }
    Ok(())
}

/// Write the per-file detection audit as CSV.
///
/// CSV rather than JSON because the point is to sort it in a spreadsheet and
/// find the zero rows.
fn write_report(path: &PathBuf, rows: &[(PathBuf, Option<usize>)]) -> Result<()> {
    use std::io::Write;
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(out, "regions,status,path")?;
    // Fewest detections first: the suspicious rows land at the top.
    let mut rows: Vec<_> = rows.iter().collect();
    rows.sort_by_key(|(p, r)| (r.unwrap_or(usize::MAX), p.clone()));
    for (p, regions) in rows {
        match regions {
            Some(n) => writeln!(out, "{n},ok,{}", csv_field(&p.display().to_string()))?,
            None => writeln!(out, ",error,{}", csv_field(&p.display().to_string()))?,
        }
    }
    out.flush()?;
    Ok(())
}

/// Quote a CSV field. Paths legitimately contain commas and quotes.
fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// Route Ctrl-C to `cancel` instead of terminating the process.
///
/// A second Ctrl-C exits immediately: if the first one is not taking effect
/// (a wedged ffmpeg, say), the user must still be able to get out.
fn install_interrupt_handler(cancel: CancelToken) {
    let result = ctrlc::set_handler(move || {
        if cancel.is_cancelled() {
            eprintln!("\ninterrupted again — exiting now");
            std::process::exit(130);
        }
        eprintln!("\nstopping after the current file… (Ctrl-C again to force)");
        cancel.cancel();
    });
    if let Err(e) = result {
        // Not fatal: without a handler Ctrl-C keeps its default meaning.
        eprintln!("warning: could not install the interrupt handler: {e}");
    }
}

fn cmd_models(cmd: ModelsCmd) -> Result<()> {
    match cmd {
        ModelsCmd::List => {
            for m in ob_core::registry::builtin_registry() {
                let state = match ob_models::status(&m) {
                    ModelStatus::Installed {
                        bytes,
                        verified: true,
                    } => {
                        format!("installed ({}, verified)", human_bytes(bytes))
                    }
                    ModelStatus::Installed {
                        bytes,
                        verified: false,
                    } => {
                        format!("installed ({}, unpinned)", human_bytes(bytes))
                    }
                    ModelStatus::Missing => {
                        format!("not installed (~{})", human_bytes(m.approx_bytes))
                    }
                };
                println!(
                    "{:<22} {:<9} {:<10} {:<8} {}",
                    m.id,
                    format!("{:?}", m.domain),
                    format!("{:?}", m.license),
                    if m.default_download { "default" } else { "" },
                    state
                );
            }
        }
        ModelsCmd::Show { id } => {
            let m = ob_core::registry::find(&id).context("unknown model")?;
            println!("{} — {}\n", m.id, m.display_name);
            for s in &m.settings {
                println!("  {} [{}]  default={:?}", s.key, s.label, s.default);
                println!("      {}", s.tooltip);
            }
        }
        ModelsCmd::Fetch { model, force } => {
            let entries = match model {
                Some(id) => vec![ob_core::registry::find(&id).context("unknown model")?],
                None => ob_core::registry::builtin_registry(),
            };
            let mut failed = 0;
            for e in entries {
                if let Err(err) = fetch_one(&e, force, false) {
                    eprintln!("  {} — FAILED: {err}", e.id);
                    failed += 1;
                }
            }
            if failed > 0 {
                bail!("{failed} model(s) could not be downloaded");
            }
        }
        ModelsCmd::Verify { id } => {
            let m = ob_core::registry::find(&id).context("unknown model")?;
            match ob_models::verify(&m)? {
                true => println!("{id}: checksum OK"),
                false => println!("{id}: present but checksum not pinned"),
            }
        }
        ModelsCmd::Path => {
            println!("{}", ob_models::cache_dir()?.display());
        }
        ModelsCmd::Remove { id } => {
            let m = ob_core::registry::find(&id).context("unknown model")?;
            ob_models::remove(&m)?;
            println!("{id}: removed from the cache");
        }
    }
    Ok(())
}

/// `obscura setup` — get the machine to a working state in one command.
///
/// Run by every installer after copying files, and the thing to tell a user to
/// run when their install predates a new model. Idempotent: already-installed
/// models are reported and skipped, so re-running it is always safe.
fn cmd_setup(args: SetupArgs) -> Result<()> {
    let entries: Vec<ModelEntry> = if !args.models.is_empty() {
        args.models
            .iter()
            .map(|id| ob_core::registry::find(id).with_context(|| format!("unknown model `{id}`")))
            .collect::<Result<_>>()?
    } else if args.all {
        ob_core::registry::builtin_registry()
    } else {
        ob_core::registry::default_downloads()
    };

    let pending: Vec<&ModelEntry> = entries
        .iter()
        .filter(|e| args.force || !ob_models::status(e).is_installed())
        .collect();

    println!("Figura Obscura setup");
    println!("  model cache: {}", ob_models::cache_dir()?.display());
    if pending.is_empty() {
        println!(
            "  all {} model(s) already installed — nothing to do",
            entries.len()
        );
    } else {
        let total: u64 = pending.iter().map(|e| e.approx_bytes).sum();
        println!(
            "  downloading {} model(s), about {}",
            pending.len(),
            human_bytes(total)
        );
    }

    let mut failures = Vec::new();
    for e in &entries {
        if let Err(err) = fetch_one(e, args.force, args.quiet) {
            eprintln!("  {} — FAILED: {err}", e.id);
            failures.push(e.id);
        }
    }

    // Report the runtime dependencies too: a missing ffmpeg is the other way a
    // fresh install fails, and setup is the moment to catch it.
    let missing_tools = ob_media::tools::missing_tools();
    for tool in &missing_tools {
        eprintln!("\nwarning: {}", ob_media::tools::search_description(*tool));
    }
    if missing_tools.is_empty() {
        println!("  ffmpeg and ffprobe: OK");
    }

    if !failures.is_empty() {
        bail!(
            "setup incomplete — could not download: {}. \
             Check your connection and run `obscura setup` again.",
            failures.join(", ")
        );
    }
    if missing_tools.is_empty() {
        println!("\nSetup complete. Launch the app, or run `obscura process --help`.");
    } else {
        println!("\nModels are installed, but video support needs the tools above.");
    }
    Ok(())
}

fn cmd_profile(cmd: ProfileCmd) -> Result<()> {
    match cmd {
        ProfileCmd::Save { path } => {
            std::fs::write(&path, Profile::default().to_json()?)?;
            println!("wrote default profile to {}", path.display());
        }
    }
    Ok(())
}

fn cmd_preview(
    input: PathBuf,
    output: PathBuf,
    profile: Option<PathBuf>,
    model: String,
) -> Result<()> {
    let prof = load_profile(&profile, &model)?;
    let detector = build_detector(&model, &prof.model_settings, true)?;
    let frame = ob_job::preview(&input, detector.as_ref(), &prof)?;
    ob_media::save_image(&frame, &output)?;
    println!("wrote preview to {}", output.display());
    Ok(())
}
