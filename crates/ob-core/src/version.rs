//! What build is this?
//!
//! Both binaries answer with [`LONG`], so a bug report identifies the exact
//! commit rather than the release-wide `0.1.0` that every build shares.

/// The crate version from `Cargo.toml` (`0.1.0`).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// `git describe` of the checkout this was built from — a short commit hash,
/// suffixed `-dirty` if the tree had uncommitted changes, or `unknown` when
/// built without a repository (a source tarball).
pub const GIT: &str = env!("OB_GIT_DESCRIBE");

/// Version and commit together, e.g. `0.1.0 (3e0e567)` — what `--version`
/// prints.
pub const LONG: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    " (",
    env!("OB_GIT_DESCRIBE"),
    ")"
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_carries_both_halves() {
        assert!(LONG.starts_with(VERSION));
        assert!(LONG.contains(GIT));
        assert!(LONG.ends_with(')'));
    }

    #[test]
    fn git_is_never_empty() {
        // An empty commit field would render as `0.1.0 ()`, which looks like a
        // bug in the version string rather than a build without a repository.
        assert!(!GIT.is_empty());
    }
}
