//! Locating the `ffmpeg`/`ffprobe` child processes.
//!
//! Obscura ships as an installed desktop application, so the tools it shells out to
//! are **bundled next to the executable** rather than assumed to be on `PATH`.
//! A paying user should never have to install ffmpeg by hand, and on Windows
//! and macOS they usually have not.
//!
//! Resolution order, first hit wins:
//!
//! 1. `SB_FFMPEG` / `SB_FFPROBE` — an explicit override, for packagers and for
//!    anyone who wants a hardware-accelerated build of their own.
//! 2. The bundle: the executable's own directory, its `bin/` subdirectory, and
//!    the macOS `.app` layouts (`../Resources/bin`, `../Frameworks`).
//! 3. `PATH`, as a bare command name — the developer/`cargo run` case, and the
//!    Linux distro-package case where ffmpeg is a real dependency.
//!
//! The result is cached: the search touches the filesystem, and a batch job
//! spawns ffmpeg once per video.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Executable suffix for the target platform.
#[cfg(windows)]
const EXE_SUFFIX: &str = ".exe";
#[cfg(not(windows))]
const EXE_SUFFIX: &str = "";

/// One of the two external tools Obscura drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tool {
    Ffmpeg,
    Ffprobe,
}

impl Tool {
    /// Base command name, without any platform executable suffix.
    pub fn name(self) -> &'static str {
        match self {
            Tool::Ffmpeg => "ffmpeg",
            Tool::Ffprobe => "ffprobe",
        }
    }

    /// Environment variable that overrides this tool's location.
    pub fn env_var(self) -> &'static str {
        match self {
            Tool::Ffmpeg => "SB_FFMPEG",
            Tool::Ffprobe => "SB_FFPROBE",
        }
    }
}

/// Directories that may hold a bundled copy of the tools, most specific first.
///
/// Every layout the three installers produce is covered here: Windows puts the
/// tools beside `obscura-gui.exe`; the AppImage/tarball uses `bin/`; the macOS
/// `.app` puts binaries in `Contents/MacOS` with resources in
/// `Contents/Resources`.
fn bundle_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    let Ok(exe) = std::env::current_exe() else {
        return dirs;
    };
    let Some(exe_dir) = exe.parent() else {
        return dirs;
    };
    dirs.push(exe_dir.to_path_buf());
    dirs.push(exe_dir.join("bin"));
    if let Some(parent) = exe_dir.parent() {
        // macOS .app: Contents/MacOS/obscura-gui -> Contents/Resources/bin
        dirs.push(parent.join("Resources").join("bin"));
        dirs.push(parent.join("Resources"));
        dirs.push(parent.join("Frameworks"));
        // Unix prefix install: <prefix>/bin/sb -> <prefix>/lib/figura-obscura
        dirs.push(parent.join("lib").join("figura-obscura"));
    }
    dirs
}

/// Search the bundle for `tool`, falling back to the bare name for `PATH`.
fn resolve(tool: Tool) -> PathBuf {
    if let Some(explicit) = std::env::var_os(tool.env_var()) {
        if !explicit.is_empty() {
            return PathBuf::from(explicit);
        }
    }
    resolve_in(tool, &bundle_dirs())
}

/// The directory-search half of [`resolve`], taking its candidates explicitly.
///
/// Split out so the search order can be tested against a temporary directory
/// rather than against whatever happens to sit beside the test binary — the
/// difference between proving the bundle wins and merely observing that some
/// ffmpeg was found on `PATH`.
fn resolve_in(tool: Tool, dirs: &[PathBuf]) -> PathBuf {
    let file = format!("{}{}", tool.name(), EXE_SUFFIX);
    for dir in dirs {
        let candidate = dir.join(&file);
        if is_executable_file(&candidate) {
            return candidate;
        }
    }
    // Not bundled: let the OS resolve it against PATH.
    PathBuf::from(file)
}

/// Whether `path` is a file we could plausibly execute.
///
/// On Unix this checks the executable bit as well as existence: a *readable but
/// not executable* file at the bundle path (a broken install, or an archive
/// extracted without permissions) should fall through to `PATH` rather than be
/// selected and then fail at spawn time with a confusing error.
fn is_executable_file(path: &Path) -> bool {
    let Ok(meta) = std::fs::metadata(path) else {
        return false;
    };
    if !meta.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        meta.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

/// Cached path to `ffmpeg`.
pub fn ffmpeg() -> &'static Path {
    static P: OnceLock<PathBuf> = OnceLock::new();
    P.get_or_init(|| resolve(Tool::Ffmpeg))
}

/// Cached path to `ffprobe`.
pub fn ffprobe() -> &'static Path {
    static P: OnceLock<PathBuf> = OnceLock::new();
    P.get_or_init(|| resolve(Tool::Ffprobe))
}

/// Path Obscura will use for `tool` (bundled, overridden, or a bare `PATH` name).
pub fn path_for(tool: Tool) -> &'static Path {
    match tool {
        Tool::Ffmpeg => ffmpeg(),
        Tool::Ffprobe => ffprobe(),
    }
}

/// Whether `tool` can actually be run right now.
///
/// Spawning it is the only honest test — a bare name resolved against `PATH`
/// cannot be checked with a filesystem probe. `-version` is cheap and exits 0.
pub fn is_available(tool: Tool) -> bool {
    std::process::Command::new(path_for(tool))
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .stdin(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Which of the required tools are missing, if any.
///
/// The GUI calls this at startup so a broken install is reported once, up
/// front, instead of as a per-file error in the middle of a batch.
pub fn missing_tools() -> Vec<Tool> {
    [Tool::Ffmpeg, Tool::Ffprobe]
        .into_iter()
        .filter(|t| !is_available(*t))
        .collect()
}

/// The usual way to install ffmpeg on this platform, as a copyable command.
///
/// Offered because using the *user's own* ffmpeg is a fully supported mode, not
/// just a fallback: Obscura never links libav, it only spawns the binary, so an
/// externally installed copy carries no redistribution obligations for us at
/// all. See `packaging/common/THIRD-PARTY.md`.
pub fn install_hint() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "winget install --id Gyan.FFmpeg"
    }
    #[cfg(target_os = "macos")]
    {
        "brew install ffmpeg"
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        // Debian/Ubuntu is the most common case; the others are one line down
        // in `install_hint_all`.
        "sudo apt install ffmpeg"
    }
}

/// Install commands for the common package managers on this platform, for the
/// details view — the single hint above is right more often than not, but a
/// user on Arch or Fedora should not have to go and look it up.
pub fn install_hint_all() -> &'static [(&'static str, &'static str)] {
    #[cfg(target_os = "windows")]
    {
        &[
            ("winget", "winget install --id Gyan.FFmpeg"),
            ("Chocolatey", "choco install ffmpeg"),
            ("Scoop", "scoop install ffmpeg"),
        ]
    }
    #[cfg(target_os = "macos")]
    {
        &[
            ("Homebrew", "brew install ffmpeg"),
            ("MacPorts", "sudo port install ffmpeg"),
        ]
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        &[
            ("Debian/Ubuntu", "sudo apt install ffmpeg"),
            ("Arch", "sudo pacman -S ffmpeg"),
            ("Fedora", "sudo dnf install ffmpeg"),
        ]
    }
}

/// A human explanation of where Obscura looked for `tool`, for error messages.
pub fn search_description(tool: Tool) -> String {
    let mut lines = vec![format!(
        "`{}` could not be run. Obscura looked for it, in order, at:",
        tool.name()
    )];
    lines.push(format!(
        "  1. the {} environment variable (currently {})",
        tool.env_var(),
        match std::env::var(tool.env_var()) {
            Ok(v) if !v.is_empty() => format!("`{v}`"),
            _ => "unset".to_string(),
        }
    ));
    let file = format!("{}{}", tool.name(), EXE_SUFFIX);
    for (i, dir) in bundle_dirs().iter().enumerate() {
        lines.push(format!("  {}. {}", i + 2, dir.join(&file).display()));
    }
    lines.push("  last. your PATH".to_string());
    lines.push(String::new());
    lines.push("Install it with one of:".to_string());
    for (manager, command) in install_hint_all() {
        lines.push(format!("  {manager}: {command}"));
    }
    lines.push(format!(
        "…or point {} at an existing binary.",
        tool.env_var()
    ));
    lines.join("\n")
}

/// Build a `Command` for `tool` with Obscura's standard child-process hygiene.
pub fn command(tool: Tool) -> std::process::Command {
    // `mut` is used only by the Windows branch below.
    #[allow(unused_mut)]
    let mut cmd = std::process::Command::new(path_for(tool));
    // A bundled ffmpeg on Windows would otherwise flash a console window every
    // time the GUI probes or transcodes a file.
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_env_override_wins() {
        // The override is taken verbatim and is *not* probed for existence: a
        // packager pointing at a path that only exists on the target machine
        // must not be silently ignored in favour of PATH.
        std::env::set_var("SB_FFMPEG", "/opt/custom/ffmpeg");
        assert_eq!(resolve(Tool::Ffmpeg), PathBuf::from("/opt/custom/ffmpeg"));
        // An empty value means "unset", not "run the empty string".
        std::env::set_var("SB_FFMPEG", "");
        assert_eq!(
            resolve(Tool::Ffmpeg),
            PathBuf::from(format!("ffmpeg{EXE_SUFFIX}"))
        );
        std::env::remove_var("SB_FFMPEG");
    }

    #[test]
    fn bundle_search_starts_at_the_executable_dir() {
        let dirs = bundle_dirs();
        // current_exe always resolves under `cargo test`, so this is non-empty
        // and its first entry is the directory holding the test binary.
        assert!(!dirs.is_empty());
        let exe = std::env::current_exe().unwrap();
        assert_eq!(dirs[0], exe.parent().unwrap());
    }

    #[test]
    fn unresolved_tool_falls_back_to_a_bare_name() {
        // No bundled ffmpeg sits next to the test binary, so resolution must
        // yield a bare command name for the OS to look up on PATH — not an
        // absolute path into the target directory that would never exist.
        let p = resolve(Tool::Ffprobe);
        assert_eq!(p, PathBuf::from(format!("ffprobe{EXE_SUFFIX}")));
    }

    #[test]
    fn search_description_names_the_env_var_path_and_a_way_to_fix_it() {
        let d = search_description(Tool::Ffmpeg);
        assert!(d.contains("SB_FFMPEG"));
        assert!(d.contains("PATH"));
        // An error that only says what is missing leaves the user stuck; this
        // one has to carry a command they can run.
        assert!(d.contains("ffmpeg"), "no install command in:\n{d}");
        assert!(d.contains(install_hint()), "the platform hint is missing");
    }

    #[test]
    fn install_hints_are_present_for_this_platform() {
        assert!(!install_hint().is_empty());
        let all = install_hint_all();
        assert!(!all.is_empty());
        for (manager, command) in all {
            assert!(!manager.is_empty() && !command.is_empty());
        }
    }

    #[test]
    fn a_directory_is_not_an_executable() {
        assert!(!is_executable_file(&std::env::temp_dir()));
    }

    /// Create `dir/name` with the given executable bit.
    fn stub(dir: &std::path::Path, name: &str, executable: bool) -> PathBuf {
        std::fs::create_dir_all(dir).unwrap();
        let path = dir.join(name);
        std::fs::write(&path, b"#!/bin/sh\nexit 0\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = if executable { 0o755 } else { 0o644 };
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
        }
        let _ = executable;
        path
    }

    #[test]
    fn a_bundled_tool_wins_over_path() {
        // The whole point of bundling: a shipped ffmpeg must be preferred to
        // whatever the user happens to have installed, so the behaviour of a
        // paid install does not depend on their system.
        let dir = std::env::temp_dir().join("ob-tools-bundled");
        let _ = std::fs::remove_dir_all(&dir);
        let expected = stub(&dir, &format!("ffmpeg{EXE_SUFFIX}"), true);

        assert_eq!(
            resolve_in(Tool::Ffmpeg, std::slice::from_ref(&dir)),
            expected
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn earlier_directories_win() {
        let base = std::env::temp_dir().join("ob-tools-order");
        let _ = std::fs::remove_dir_all(&base);
        let first = base.join("first");
        let second = base.join("second");
        let expected = stub(&first, &format!("ffprobe{EXE_SUFFIX}"), true);
        stub(&second, &format!("ffprobe{EXE_SUFFIX}"), true);

        assert_eq!(resolve_in(Tool::Ffprobe, &[first, second]), expected);
        std::fs::remove_dir_all(&base).ok();
    }

    #[cfg(unix)]
    #[test]
    fn a_non_executable_file_is_skipped_for_path() {
        // An archive extracted without permissions leaves a readable but
        // non-executable ffmpeg. Selecting it would fail at spawn time with a
        // confusing error; falling through to PATH at least works.
        let dir = std::env::temp_dir().join("ob-tools-noexec");
        let _ = std::fs::remove_dir_all(&dir);
        stub(&dir, "ffmpeg", false);

        assert_eq!(
            resolve_in(Tool::Ffmpeg, std::slice::from_ref(&dir)),
            PathBuf::from("ffmpeg")
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
