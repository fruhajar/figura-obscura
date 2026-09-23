//! Stamps the build with the commit it came from.
//!
//! `CARGO_PKG_VERSION` alone cannot answer the question that actually matters
//! in a bug report — *which build is this?* — because every build between two
//! releases carries the same `0.4.1`. A user running a binary from before a
//! fix, and a user running one from after it, report the same version. So the
//! commit goes in too.
//!
//! Git is best-effort: a build from a source tarball has no repository, and
//! that is not a build failure. It just gets `unknown`.

use std::path::Path;
use std::process::Command;

fn main() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
    let repo = Path::new(&manifest).join("../..");

    // Rebuild when the checkout moves. `.git/HEAD` covers checkouts and
    // commits on a detached HEAD; the file HEAD points at covers a commit on a
    // branch. Neither notices an uncommitted edit, so `-dirty` can go stale
    // until something else triggers a rebuild of this crate — acceptable for a
    // diagnostic string, and the hash itself stays right.
    let git_dir = repo.join(".git");
    if git_dir.join("HEAD").exists() {
        println!("cargo:rerun-if-changed={}", git_dir.join("HEAD").display());
        if let Some(r) = head_ref(&git_dir) {
            let path = git_dir.join(&r);
            if path.exists() {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }

    println!("cargo:rustc-env=OB_GIT_DESCRIBE={}", describe(&repo));
}

/// The ref `HEAD` is a symbolic link to, e.g. `refs/heads/dev`.
fn head_ref(git_dir: &Path) -> Option<String> {
    let head = std::fs::read_to_string(git_dir.join("HEAD")).ok()?;
    Some(head.trim().strip_prefix("ref: ")?.to_string())
}

fn describe(repo: &Path) -> String {
    let out = Command::new("git")
        .args(["describe", "--always", "--dirty", "--abbrev=7"])
        .current_dir(repo)
        .output();
    match out {
        Ok(o) if o.status.success() => {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if s.is_empty() {
                "unknown".into()
            } else {
                s
            }
        }
        _ => "unknown".into(),
    }
}
