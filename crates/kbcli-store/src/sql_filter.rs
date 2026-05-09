//! Translate a [`kbcli_core::Filter`] into a SQL `WHERE` fragment for the
//! storage backends. Both backends store metadata as a JSON column on the
//! `documents` table, so the same translator works for either.

use kbcli_core::{Filter, MetaValue, Predicate};

/// SQL-fragment + bound JSON parameter list emitted by [`filter_to_sql`].
#[derive(Debug, Default)]
pub struct FilterSql {
    /// SQL boolean expression (without leading `WHERE`). `"1"` for `Filter::All`.
    pub sql: String,
    /// Parameter values to bind in order, encoded as JSON text.
    pub params: Vec<String>,
}

/// Convert a [`Filter`] into SQL evaluated against the `documents.meta`
/// JSON column (alias `m`).
pub fn filter_to_sql(filter: &Filter) -> FilterSql {
    let mut out = FilterSql::default();
    match filter {
        Filter::All => out.sql = "1".to_string(),
        _ => emit(filter, &mut out),
    }
    out
}

fn emit(f: &Filter, out: &mut FilterSql) {
    match f {
        Filter::All => out.sql.push('1'),
        Filter::And(items) => emit_combo(items, " AND ", out),
        Filter::Or(items) => emit_combo(items, " OR ", out),
        Filter::Not(inner) => {
            out.sql.push_str("NOT (");
            emit(inner, out);
            out.sql.push(')');
        }
        Filter::Atom { key, predicate } => emit_atom(key, predicate, out),
    }
}

fn emit_combo(items: &[Filter], sep: &str, out: &mut FilterSql) {
    if items.is_empty() {
        out.sql.push('1');
        return;
    }
    out.sql.push('(');
    for (i, it) in items.iter().enumerate() {
        if i > 0 {
            out.sql.push_str(sep);
        }
        emit(it, out);
    }
    out.sql.push(')');
}

fn emit_atom(key: &str, pred: &Predicate, out: &mut FilterSql) {
    let path = format!("$.{}", json_path_key(key));
    let extract = format!(
        "json_extract(documents.meta, '{}')",
        path.replace('\'', "''")
    );
    match pred {
        Predicate::Exists => {
            out.sql.push_str(&format!(
                "json_type(documents.meta, '{}') IS NOT NULL",
                path.replace('\'', "''")
            ));
        }
        Predicate::Missing => {
            out.sql.push_str(&format!(
                "json_type(documents.meta, '{}') IS NULL",
                path.replace('\'', "''")
            ));
        }
        Predicate::Eq(v) => emit_compare(&extract, "=", v, out),
        Predicate::Ne(v) => emit_compare(&extract, "!=", v, out),
        Predicate::Lt(v) => emit_compare(&extract, "<", v, out),
        Predicate::Le(v) => emit_compare(&extract, "<=", v, out),
        Predicate::Gt(v) => emit_compare(&extract, ">", v, out),
        Predicate::Ge(v) => emit_compare(&extract, ">=", v, out),
        Predicate::In(values) => emit_in(&extract, values, false, out),
        Predicate::NotIn(values) => emit_in(&extract, values, true, out),
        Predicate::Contains(needle) => {
            out.sql
                .push_str(&format!("LOWER(CAST({extract} AS TEXT)) LIKE ?"));
            let n = needle
                .to_lowercase()
                .replace('%', r"\%")
                .replace('_', r"\_");
            out.params.push(format!("%{}%", n));
        }
    }
}

fn emit_compare(extract: &str, op: &str, v: &MetaValue, out: &mut FilterSql) {
    out.sql.push_str(&format!("{extract} {op} ?"));
    out.params.push(meta_to_json(v));
}

fn emit_in(extract: &str, values: &[MetaValue], negate: bool, out: &mut FilterSql) {
    if values.is_empty() {
        out.sql.push(if negate { '1' } else { '0' });
        return;
    }
    let placeholders = std::iter::repeat("?")
        .take(values.len())
        .collect::<Vec<_>>()
        .join(",");
    let prefix = if negate { "NOT " } else { "" };
    out.sql
        .push_str(&format!("{extract} {prefix}IN ({placeholders})"));
    for v in values {
        out.params.push(meta_to_json(v));
    }
}

fn meta_to_json(v: &MetaValue) -> String {
    let j: serde_json::Value = v.clone().into();
    match j {
        // Unwrap simple scalars so SQLite's type system can compare them
        // directly (json_extract returns scalars as native types).
        serde_json::Value::String(s) => s,
        other => other.to_string(),
    }
}

fn json_path_key(key: &str) -> String {
    if key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        key.to_string()
    } else {
        format!("\"{}\"", key.replace('"', "\\\""))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_emits_one() {
        let r = filter_to_sql(&Filter::All);
        assert_eq!(r.sql, "1");
        assert!(r.params.is_empty());
    }

    #[test]
    fn eq_uses_one_bind() {
        let f = Filter::eq("lang", MetaValue::Str("rust".into()));
        let r = filter_to_sql(&f);
        assert!(r.sql.contains("json_extract"));
        assert!(r.sql.contains(" = ?"));
        assert_eq!(r.params, vec!["rust"]);
    }

    #[test]
    fn and_combines() {
        let f = Filter::and([
            Filter::eq("lang", MetaValue::Str("rust".into())),
            Filter::Atom {
                key: "score".into(),
                predicate: Predicate::Ge(MetaValue::Int(10)),
            },
        ]);
        let r = filter_to_sql(&f);
        assert!(r.sql.contains(" AND "));
        assert_eq!(r.params.len(), 2);
    }

    #[test]
    fn exists_missing() {
        let r = filter_to_sql(&Filter::exists("k"));
        assert!(r.sql.contains("IS NOT NULL"));
        let r = filter_to_sql(&Filter::Atom {
            key: "k".into(),
            predicate: Predicate::Missing,
        });
        assert!(r.sql.contains("IS NULL"));
    }

    #[test]
    fn in_list_uses_n_binds() {
        let f = Filter::Atom {
            key: "tag".into(),
            predicate: Predicate::In(vec![MetaValue::Str("a".into()), MetaValue::Str("b".into())]),
        };
        let r = filter_to_sql(&f);
        assert!(r.sql.contains(" IN (?,?)"));
        assert_eq!(r.params.len(), 2);
    }
}
