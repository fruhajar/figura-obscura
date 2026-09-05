//! # ob-models
//!
//! Manages the local ONNX model cache and is the **only** crate in Obscura that
//! performs network I/O (requirement R8). `fetch` downloads a model once,
//! verifies its SHA-256, and stores it under the platform cache dir. Every other
//! crate operates purely on the already-downloaded file path.

use ob_core::registry::ModelEntry;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("could not determine a cache directory")]
    NoCacheDir,
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("download of `{0}` failed: {1}")]
    Download(String, String),
    #[error("checksum mismatch for `{id}`: expected {expected}, got {got}")]
    Checksum {
        id: String,
        expected: String,
        got: String,
    },
    #[error("model `{0}` has no configured download URL yet")]
    NoUrl(String),
    #[error(
        "the download for `{id}` is not an ONNX model ({reason}). \
         The URL likely served a login, proxy or error page instead of the file. \
         Download it manually and place it at {dest}"
    )]
    NotAModel {
        id: String,
        reason: String,
        dest: String,
    },
    #[error("model `{0}` is not downloaded (run `obscura models fetch --model {0}`)")]
    Missing(String),
    #[error("download of `{0}` was cancelled")]
    Cancelled(String),
}

impl ModelError {
    /// Whether retrying the same request could plausibly succeed.
    ///
    /// Only transport failures are retried. A checksum mismatch, an HTML page
    /// where a model should be, or a missing URL will fail identically every
    /// time — retrying those just multiplies the wait before the user sees the
    /// real problem.
    pub fn is_transient(&self) -> bool {
        matches!(self, ModelError::Download(..) | ModelError::Io(_))
    }
}

/// Root cache directory for Obscura models, e.g. `~/.cache/figura-obscura/models`.
/// Overridable via `OBSCURA_MODEL_DIR`.
pub fn cache_dir() -> Result<PathBuf, ModelError> {
    if let Ok(dir) = std::env::var("OBSCURA_MODEL_DIR") {
        return Ok(PathBuf::from(dir));
    }
    let base = dirs::cache_dir().ok_or(ModelError::NoCacheDir)?;
    Ok(base.join("figura-obscura").join("models"))
}

/// Local path a model's ONNX file lives at (whether or not it exists yet).
pub fn model_path(entry: &ModelEntry) -> Result<PathBuf, ModelError> {
    Ok(cache_dir()?.join(format!("{}.onnx", entry.id)))
}

/// True if the model file is present on disk.
pub fn is_present(entry: &ModelEntry) -> Result<bool, ModelError> {
    Ok(model_path(entry)?.exists())
}

/// Return the model path, erroring if it has not been downloaded.
pub fn require(entry: &ModelEntry) -> Result<PathBuf, ModelError> {
    let p = model_path(entry)?;
    if p.exists() {
        Ok(p)
    } else {
        Err(ModelError::Missing(entry.id.to_string()))
    }
}

/// Compute the lowercase-hex SHA-256 of a file.
pub fn sha256_file(path: &std::path::Path) -> Result<String, ModelError> {
    let mut f = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Verify a downloaded model against its expected checksum. A `sha256` of
/// `"TODO"`/empty in the registry is treated as "not yet pinned" and skipped
/// with a returned `false` (verified=false) rather than an error.
pub fn verify(entry: &ModelEntry) -> Result<bool, ModelError> {
    let path = require(entry)?;
    if entry.sha256.is_empty() || entry.sha256 == "TODO" {
        return Ok(false); // present but checksum not pinned
    }
    let got = sha256_file(&path)?;
    if got.eq_ignore_ascii_case(entry.sha256) {
        Ok(true)
    } else {
        Err(ModelError::Checksum {
            id: entry.id.to_string(),
            expected: entry.sha256.to_string(),
            got,
        })
    }
}

/// How far along a download is.
///
/// `total` is `None` until the server reports a `Content-Length`, and stays
/// `None` for chunked responses — a progress UI must handle an unknown total
/// (show bytes so far and a spinner) rather than assume a percentage exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DownloadProgress {
    pub downloaded: u64,
    pub total: Option<u64>,
}

impl DownloadProgress {
    /// Completed fraction in `0.0..=1.0`, or `None` when the total is unknown.
    pub fn fraction(&self) -> Option<f32> {
        let total = self.total?;
        if total == 0 {
            return None;
        }
        Some((self.downloaded as f32 / total as f32).clamp(0.0, 1.0))
    }
}

/// What a fetch actually did — the caller reports "already installed" very
/// differently from "downloaded 44 MB".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FetchOutcome {
    /// The model was already in the cache and `force` was not set.
    AlreadyPresent(PathBuf),
    /// The model was downloaded and committed during this call.
    Downloaded { path: PathBuf, bytes: u64 },
}

impl FetchOutcome {
    pub fn path(&self) -> &std::path::Path {
        match self {
            FetchOutcome::AlreadyPresent(p) => p,
            FetchOutcome::Downloaded { path, .. } => path,
        }
    }
}

/// Knobs for [`fetch_with`]. `Default` is a plain, silent, uncancellable fetch.
#[derive(Default)]
pub struct FetchOptions<'a> {
    /// Re-download even if the file is already cached.
    pub force: bool,
    /// Called as bytes arrive. Invoked at most every 64 KiB, so it is cheap
    /// enough to push straight into a channel.
    pub progress: Option<&'a (dyn Fn(DownloadProgress) + Send + Sync)>,
    /// Polled during the transfer; set it to abort. The partial file is
    /// removed, so a cancelled download never leaves a half model behind.
    pub cancel: Option<&'a AtomicBool>,
    /// Extra attempts after a *transient* failure. 0 means try once.
    pub retries: u32,
}

impl<'a> FetchOptions<'a> {
    /// Sensible defaults for an interactive download: two retries, so a single
    /// dropped connection on a hotel/mobile link is invisible to the user.
    pub fn interactive() -> Self {
        Self {
            retries: 2,
            ..Default::default()
        }
    }

    pub fn force(mut self, force: bool) -> Self {
        self.force = force;
        self
    }

    pub fn progress(mut self, f: &'a (dyn Fn(DownloadProgress) + Send + Sync)) -> Self {
        self.progress = Some(f);
        self
    }

    pub fn cancel(mut self, flag: &'a AtomicBool) -> Self {
        self.cancel = Some(flag);
        self
    }

    fn cancelled(&self) -> bool {
        self.cancel.is_some_and(|c| c.load(Ordering::Relaxed))
    }
}

/// Identify Obscura to the model hosts.
///
/// HuggingFace and GitHub both treat an absent User-Agent as a bot signal and
/// may answer with a challenge page rather than the file — which Obscura would then
/// correctly reject as "not a model", leaving the user with a puzzling error
/// for what is really a missing header.
fn user_agent() -> String {
    format!("FiguraObscura/{} (+model-fetch)", env!("CARGO_PKG_VERSION"))
}

/// A configured HTTP agent. Timeouts matter: without a read timeout a stalled
/// connection hangs the download thread forever and the GUI's cancel button
/// only takes effect once a byte finally arrives.
fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .user_agent(&user_agent())
        .timeout_connect(Duration::from_secs(20))
        .timeout_read(Duration::from_secs(60))
        .build()
}

/// Download the model if missing (or if `force`), then verify its checksum.
///
/// This is the sole network operation in Obscura. It writes to a `.part` file and
/// renames on success so an interrupted fetch never looks complete.
pub fn fetch(entry: &ModelEntry, force: bool) -> Result<PathBuf, ModelError> {
    fetch_with(entry, &FetchOptions::default().force(force)).map(|o| o.path().to_path_buf())
}

/// Download `entry` into the cache, reporting progress and honouring cancellation.
///
/// Retries only transient transport errors (see [`ModelError::is_transient`]),
/// with a short linear backoff. Every failure path removes the `.part` file, so
/// the cache only ever contains complete, sniffed and (where pinned) verified
/// models.
pub fn fetch_with(entry: &ModelEntry, opts: &FetchOptions) -> Result<FetchOutcome, ModelError> {
    let dest = model_path(entry)?;
    if dest.exists() && !opts.force {
        return Ok(FetchOutcome::AlreadyPresent(dest));
    }
    if entry.url.is_empty() || entry.url.starts_with("TODO") {
        return Err(ModelError::NoUrl(entry.id.to_string()));
    }
    fs::create_dir_all(dest.parent().ok_or(ModelError::NoCacheDir)?)?;

    let mut attempt = 0;
    loop {
        match download_once(entry, &dest, opts) {
            Ok(outcome) => return Ok(outcome),
            Err(e) => {
                if attempt >= opts.retries || !e.is_transient() || opts.cancelled() {
                    return Err(e);
                }
                attempt += 1;
                // Linear backoff. Long enough for a flapping link to settle,
                // short enough that a cancel is still felt promptly.
                sleep_cancellable(Duration::from_secs(attempt as u64 * 2), opts);
                if opts.cancelled() {
                    return Err(ModelError::Cancelled(entry.id.to_string()));
                }
            }
        }
    }
}

/// Sleep, waking early if the download is cancelled meanwhile.
fn sleep_cancellable(total: Duration, opts: &FetchOptions) {
    let step = Duration::from_millis(100);
    let mut slept = Duration::ZERO;
    while slept < total {
        if opts.cancelled() {
            return;
        }
        std::thread::sleep(step);
        slept += step;
    }
}

/// One download attempt: GET, stream to `.part`, sniff, verify, commit.
fn download_once(
    entry: &ModelEntry,
    dest: &std::path::Path,
    opts: &FetchOptions,
) -> Result<FetchOutcome, ModelError> {
    if opts.cancelled() {
        return Err(ModelError::Cancelled(entry.id.to_string()));
    }

    let resp = agent()
        .get(entry.url)
        // Required by GitHub's release-asset endpoint, which otherwise answers
        // with the asset's JSON metadata instead of its bytes. Harmless
        // elsewhere: every other host we fetch from serves the file regardless.
        .set("Accept", "application/octet-stream")
        .call()
        .map_err(|e| ModelError::Download(entry.id.to_string(), e.to_string()))?;

    let total: Option<u64> = resp
        .header("Content-Length")
        .and_then(|v| v.trim().parse().ok())
        .filter(|n| *n > 0);

    let tmp = dest.with_extension("part");
    let written = stream_to_file(resp, &tmp, total, entry, opts).inspect_err(|_| {
        // Never leave a partial file behind: on the next run its mere presence
        // would make `is_present` report the model as installed.
        let _ = fs::remove_file(&tmp);
    })?;

    // An intercepted download (captive portal, proxy, expired release asset)
    // commonly answers 200 with an HTML page. Committing that as a `.onnx`
    // fails much later with a confusing inference error, and an unpinned
    // checksum won't catch it — so sniff the payload before committing.
    if let Err(reason) = looks_like_onnx(&tmp) {
        let _ = fs::remove_file(&tmp);
        return Err(ModelError::NotAModel {
            id: entry.id.to_string(),
            reason,
            dest: dest.display().to_string(),
        });
    }

    // Verify before committing when a checksum is pinned.
    if !entry.sha256.is_empty() && entry.sha256 != "TODO" {
        let got = sha256_file(&tmp)?;
        if !got.eq_ignore_ascii_case(entry.sha256) {
            let _ = fs::remove_file(&tmp);
            return Err(ModelError::Checksum {
                id: entry.id.to_string(),
                expected: entry.sha256.to_string(),
                got,
            });
        }
    }

    fs::rename(&tmp, dest)?;
    Ok(FetchOutcome::Downloaded {
        path: dest.to_path_buf(),
        bytes: written,
    })
}

/// Copy the response body to `tmp`, emitting progress and checking cancellation.
///
/// Hand-rolled rather than `std::io::copy` because those two behaviours are the
/// whole point: a 100 MB model over a slow link is otherwise an unresponsive UI
/// with no way out.
fn stream_to_file(
    resp: ureq::Response,
    tmp: &std::path::Path,
    total: Option<u64>,
    entry: &ModelEntry,
    opts: &FetchOptions,
) -> Result<u64, ModelError> {
    let mut reader = resp.into_reader();
    let mut out = std::io::BufWriter::new(fs::File::create(tmp)?);
    let mut buf = vec![0u8; 64 * 1024];
    let mut downloaded: u64 = 0;

    // Report 0/total immediately so the UI can render a real bar before the
    // first chunk lands, rather than sitting blank on a slow connection.
    if let Some(f) = opts.progress {
        f(DownloadProgress {
            downloaded: 0,
            total,
        });
    }

    loop {
        if opts.cancelled() {
            return Err(ModelError::Cancelled(entry.id.to_string()));
        }
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        use std::io::Write;
        out.write_all(&buf[..n])?;
        downloaded += n as u64;
        if let Some(f) = opts.progress {
            f(DownloadProgress { downloaded, total });
        }
    }

    use std::io::Write;
    out.flush()?;
    Ok(downloaded)
}

/// Ensure `entry` is available locally, downloading it if it is not.
///
/// The one call the GUI and `obscura process --auto-fetch` need: it returns a usable
/// path whether or not the model was already there.
pub fn ensure(entry: &ModelEntry, opts: &FetchOptions) -> Result<PathBuf, ModelError> {
    fetch_with(entry, opts).map(|o| o.path().to_path_buf())
}

/// Installed state of one model, for the GUI's model list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelStatus {
    /// Not downloaded.
    Missing,
    /// Downloaded. `verified` is false when the registry pins no checksum.
    Installed { bytes: u64, verified: bool },
}

impl ModelStatus {
    pub fn is_installed(&self) -> bool {
        matches!(self, ModelStatus::Installed { .. })
    }
}

/// Inspect a model's local state without touching the network.
///
/// A checksum *mismatch* is reported as `Missing` rather than an error: to the
/// user a corrupt file is simply not a usable model, and the fix — download it
/// again — is the same one the missing case offers.
pub fn status(entry: &ModelEntry) -> ModelStatus {
    let Ok(path) = model_path(entry) else {
        return ModelStatus::Missing;
    };
    let Ok(meta) = fs::metadata(&path) else {
        return ModelStatus::Missing;
    };
    if !meta.is_file() || meta.len() == 0 {
        return ModelStatus::Missing;
    }
    match verify(entry) {
        Ok(verified) => ModelStatus::Installed {
            bytes: meta.len(),
            verified,
        },
        Err(_) => ModelStatus::Missing,
    }
}

/// Delete a downloaded model from the cache. Succeeds if it was already gone.
pub fn remove(entry: &ModelEntry) -> Result<(), ModelError> {
    let path = model_path(entry)?;
    match fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.into()),
    }
}

/// Cheap sanity check that a downloaded file is really an ONNX model.
///
/// ONNX is protobuf: a `ModelProto` starts with field 1 (`ir_version`), i.e. a
/// `0x08` tag byte. We don't parse the graph — we only reject the payloads that
/// actually show up in practice (HTML sign-in/error pages and empty bodies), so
/// a valid-but-unusual model is never turned away.
fn looks_like_onnx(path: &std::path::Path) -> Result<(), String> {
    let len = fs::metadata(path).map_err(|e| e.to_string())?.len();
    if len == 0 {
        return Err("the file is empty".to_string());
    }

    let mut head = [0u8; 512];
    let read = {
        let mut f = fs::File::open(path).map_err(|e| e.to_string())?;
        read_up_to(&mut f, &mut head)?
    };
    let head = &head[..read];

    let text_start = head
        .iter()
        .position(|b| !b.is_ascii_whitespace())
        .unwrap_or(read);
    let body = &head[text_start..];
    let lower: Vec<u8> = body
        .iter()
        .take(64)
        .map(|b| b.to_ascii_lowercase())
        .collect();
    for marker in [&b"<!doctype"[..], &b"<html"[..], &b"<?xml"[..], &b"{"[..]] {
        if lower.starts_with(marker) {
            return Err(format!(
                "it starts with `{}`, which looks like a web page, not a model ({len} bytes)",
                String::from_utf8_lossy(&body[..marker.len().min(body.len())])
            ));
        }
    }

    if body.first() != Some(&0x08) {
        return Err(format!(
            "it does not begin with an ONNX ModelProto header ({len} bytes)"
        ));
    }
    Ok(())
}

/// Fill `buf` with up to `buf.len()` bytes, tolerating short reads.
fn read_up_to(f: &mut fs::File, buf: &mut [u8]) -> Result<usize, String> {
    let mut total = 0;
    while total < buf.len() {
        match f.read(&mut buf[total..]) {
            Ok(0) => break,
            Ok(n) => total += n,
            Err(e) => return Err(e.to_string()),
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, MutexGuard};

    /// `OBSCURA_MODEL_DIR` is process-global, but Rust runs tests in parallel
    /// threads. Any test that sets it must hold this lock, or it will observe
    /// (and clobber) another test's cache directory.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Point the model cache at a private temp directory for one test.
    struct CacheGuard {
        _lock: MutexGuard<'static, ()>,
        dir: PathBuf,
    }

    impl CacheGuard {
        fn new(name: &str) -> Self {
            let lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let dir = std::env::temp_dir().join(format!("ob-models-test-{name}"));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).unwrap();
            std::env::set_var("OBSCURA_MODEL_DIR", &dir);
            Self { _lock: lock, dir }
        }
    }

    impl Drop for CacheGuard {
        fn drop(&mut self) {
            std::env::remove_var("OBSCURA_MODEL_DIR");
            let _ = fs::remove_dir_all(&self.dir);
        }
    }

    /// A registry entry rewritten to a unique id so it cannot collide with a
    /// real cached model or another test.
    ///
    /// The checksum is cleared explicitly. Tests must not inherit whether a
    /// shipped entry happens to be pinned — that is a fact about the registry
    /// that changes over time (and did: pinning `nudenet-320n` silently broke
    /// this fixture). A test that cares about pinning sets `sha256` itself.
    fn test_entry(id: &'static str) -> ModelEntry {
        let mut e = ob_core::registry::find("nudenet-320n").unwrap();
        e.id = id;
        e.sha256 = "";
        e
    }

    #[test]
    fn cache_dir_respects_env_override() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("OBSCURA_MODEL_DIR", "/tmp/ob-models-test");
        assert_eq!(cache_dir().unwrap(), PathBuf::from("/tmp/ob-models-test"));
        std::env::remove_var("OBSCURA_MODEL_DIR");
    }

    #[test]
    fn progress_fraction_needs_a_known_total() {
        assert_eq!(
            DownloadProgress {
                downloaded: 5,
                total: Some(10)
            }
            .fraction(),
            Some(0.5)
        );
        // Chunked responses report no length; a UI must not divide by it.
        assert_eq!(
            DownloadProgress {
                downloaded: 5,
                total: None
            }
            .fraction(),
            None
        );
        assert_eq!(
            DownloadProgress {
                downloaded: 5,
                total: Some(0)
            }
            .fraction(),
            None
        );
        // A server that under-reports Content-Length must not yield >100%.
        assert_eq!(
            DownloadProgress {
                downloaded: 30,
                total: Some(10)
            }
            .fraction(),
            Some(1.0)
        );
    }

    #[test]
    fn cancellation_is_honoured_before_any_request() {
        // Proves cancel is checked ahead of the network: this entry has a real
        // URL, and the test suite must never reach out to it.
        let _g = CacheGuard::new("cancel");
        let entry = test_entry("obscura-test-cancel");
        let flag = AtomicBool::new(true);
        let opts = FetchOptions::default().cancel(&flag);
        assert!(matches!(
            fetch_with(&entry, &opts),
            Err(ModelError::Cancelled(_))
        ));
        // And nothing was committed to the cache.
        assert_eq!(status(&entry), ModelStatus::Missing);
    }

    #[test]
    fn a_cached_model_short_circuits_without_network() {
        let _g = CacheGuard::new("present");
        let entry = test_entry("obscura-test-present");
        let path = model_path(&entry).unwrap();
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, [0x08, 0x07, 0x12, 0x04]).unwrap();

        let outcome = fetch_with(&entry, &FetchOptions::default()).unwrap();
        assert_eq!(outcome, FetchOutcome::AlreadyPresent(path.clone()));
        assert_eq!(outcome.path(), path);
    }

    #[test]
    fn status_reports_installed_unverified_when_no_checksum_is_pinned() {
        let _g = CacheGuard::new("status");
        let entry = test_entry("obscura-test-status");
        assert_eq!(status(&entry), ModelStatus::Missing);

        let path = model_path(&entry).unwrap();
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, [0x08, 0x07, 0x12, 0x04]).unwrap();
        // Registry entries currently pin no digest, so this is the shipping case.
        assert_eq!(
            status(&entry),
            ModelStatus::Installed {
                bytes: 4,
                verified: false
            }
        );
        assert!(status(&entry).is_installed());
    }

    #[test]
    fn status_treats_a_corrupt_file_as_missing() {
        let _g = CacheGuard::new("corrupt");
        let mut entry = test_entry("obscura-test-corrupt");
        // Pin a digest the file will not match.
        entry.sha256 = "0000000000000000000000000000000000000000000000000000000000000000";
        let path = model_path(&entry).unwrap();
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, [0x08, 0x07]).unwrap();
        // Reported as Missing, not as an error: "download it again" is the fix
        // for both, and the UI should offer that rather than a checksum dump.
        assert_eq!(status(&entry), ModelStatus::Missing);
    }

    #[test]
    fn an_empty_cached_file_is_not_a_model() {
        let _g = CacheGuard::new("empty");
        let entry = test_entry("obscura-test-empty");
        let path = model_path(&entry).unwrap();
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, b"").unwrap();
        // A zero-byte file is what a disk-full or killed download leaves.
        assert_eq!(status(&entry), ModelStatus::Missing);
    }

    #[test]
    fn remove_is_idempotent() {
        let _g = CacheGuard::new("remove");
        let entry = test_entry("obscura-test-remove");
        let path = model_path(&entry).unwrap();
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, [0x08]).unwrap();
        remove(&entry).unwrap();
        assert!(!path.exists());
        // Removing an already-absent model is success, not an error: the GUI
        // calls this to reach a known state.
        remove(&entry).unwrap();
    }

    #[test]
    fn only_transport_errors_are_retried() {
        assert!(ModelError::Download("m".into(), "reset".into()).is_transient());
        // A checksum mismatch or an HTML page fails identically every time.
        assert!(!ModelError::Checksum {
            id: "m".into(),
            expected: "a".into(),
            got: "b".into()
        }
        .is_transient());
        assert!(!ModelError::NoUrl("m".into()).is_transient());
        assert!(!ModelError::Cancelled("m".into()).is_transient());
    }

    #[test]
    fn hex_encoding_is_lowercase_padded() {
        assert_eq!(hex(&[0x0a, 0xff, 0x00]), "0aff00");
    }

    fn tmp_file(name: &str, bytes: &[u8]) -> PathBuf {
        let p = std::env::temp_dir().join(name);
        fs::write(&p, bytes).unwrap();
        p
    }

    #[test]
    fn sniffer_rejects_a_github_sign_in_page() {
        // Exactly the payload an intercepted release-asset download returns.
        let p = tmp_file(
            "obscura-sniff-html.onnx",
            b"\n\n\n\n<!DOCTYPE html>\n<html lang=\"en\"><title>Sign in to GitHub</title>",
        );
        let err = looks_like_onnx(&p).unwrap_err();
        assert!(err.contains("web page"), "unexpected reason: {err}");
        fs::remove_file(&p).ok();
    }

    #[test]
    fn sniffer_rejects_empty_and_non_onnx_files() {
        let empty = tmp_file("obscura-sniff-empty.onnx", b"");
        assert!(looks_like_onnx(&empty).unwrap_err().contains("empty"));
        fs::remove_file(&empty).ok();

        let junk = tmp_file("obscura-sniff-junk.onnx", b"not a model at all");
        assert!(looks_like_onnx(&junk).unwrap_err().contains("ModelProto"));
        fs::remove_file(&junk).ok();
    }

    #[test]
    fn sniffer_accepts_an_onnx_modelproto_header() {
        // field 1 (ir_version) varint = 0x08, then a plausible protobuf body.
        let p = tmp_file(
            "obscura-sniff-ok.onnx",
            &[0x08, 0x07, 0x12, 0x04, b't', b'e', b's', b't'],
        );
        assert!(looks_like_onnx(&p).is_ok());
        fs::remove_file(&p).ok();
    }

    #[test]
    fn fetch_rejects_placeholder_url() {
        // Built here rather than taken from the registry: every shipped entry
        // now has a real URL (`every_registry_url_is_real` enforces that), and
        // fetching one would hit the network during an offline unit-test run.
        let mut entry = ob_core::registry::find("nudenet-320n").unwrap();
        // A unique id keeps the cache-hit early-return out of the way, so the
        // result can't depend on what happens to be downloaded already.
        entry.id = "obscura-test-placeholder-url";
        entry.url = "TODO://not/a/real/url";
        assert!(matches!(fetch(&entry, false), Err(ModelError::NoUrl(_))));
    }
}
