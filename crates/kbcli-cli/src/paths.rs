//! Resolve database paths from CLI inputs.
//!
//! By default, named databases live at `~/.kbcli/<name>.db`. The `--path`
//! flag overrides this to point at an arbitrary file.

use std::path::PathBuf;

use kbcli_core::{Error, Result};

/// Resolve the directory where named databases live by default.
pub fn default_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| Error::other("could not resolve home directory"))?;
    Ok(home.join(".kbcli"))
}

/// Resolve the on-disk path for a database, honoring `--path` if provided.
pub fn resolve_db(name: &str, explicit: Option<&PathBuf>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        return Ok(p.clone());
    }
    if name.is_empty() {
        return Err(Error::invalid("db name cannot be empty"));
    }
    let dir = default_dir()?;
    Ok(dir.join(format!("{name}.db")))
}

/// Ensure the parent directory of `path` exists.
pub fn ensure_parent(path: &std::path::Path) -> Result<()> {
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p)?;
    }
    Ok(())
}
