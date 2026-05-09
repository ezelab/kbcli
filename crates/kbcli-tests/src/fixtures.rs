//! Tiny in-repo fixtures for functional tests.

/// A handful of short documents covering distinct topics. Each tuple is
/// `(id, text, lang_tag, year)`.
pub fn small_corpus() -> Vec<(&'static str, &'static str, &'static str, i64)> {
    vec![
        (
            "rust1",
            "Rust is a systems programming language focused on safety and performance.",
            "rust",
            2010,
        ),
        (
            "rust2",
            "Cargo is the Rust package manager and build tool, used by every Rust project.",
            "rust",
            2014,
        ),
        (
            "py1",
            "Python is a popular high-level scripting language with extensive libraries.",
            "python",
            1991,
        ),
        (
            "py2",
            "NumPy provides efficient array computation for Python and is widely used in data science.",
            "python",
            2006,
        ),
        (
            "go1",
            "Go is a compiled language designed at Google with fast builds and a tidy syntax.",
            "go",
            2009,
        ),
        (
            "js1",
            "JavaScript powers most interactive websites and the modern web platform.",
            "js",
            1995,
        ),
        (
            "ml1",
            "Machine learning models learn statistical patterns from labeled data sets.",
            "ml",
            2000,
        ),
        (
            "ml2",
            "Embeddings represent text as vectors where similar meanings yield similar directions.",
            "ml",
            2013,
        ),
    ]
}

pub fn jsonl_lines() -> Vec<String> {
    small_corpus()
        .iter()
        .map(|(id, text, lang, year)| {
            serde_json::json!({
                "id": id,
                "text": text,
                "meta": { "lang": lang, "year": year }
            })
            .to_string()
        })
        .collect()
}
