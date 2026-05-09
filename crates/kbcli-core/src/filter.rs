use serde::{Deserialize, Serialize};

use crate::{Error, MetaValue};

/// Comparison/membership predicate evaluated against a single metadata key.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", content = "value", rename_all = "snake_case")]
pub enum Predicate {
    Eq(MetaValue),
    Ne(MetaValue),
    Lt(MetaValue),
    Le(MetaValue),
    Gt(MetaValue),
    Ge(MetaValue),
    In(Vec<MetaValue>),
    NotIn(Vec<MetaValue>),
    Exists,
    Missing,
    /// Substring contains, case-insensitive (string-typed values only).
    Contains(String),
}

/// A boolean tree of predicates over a document's metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Filter {
    /// Match every document.
    All,
    /// `key` must satisfy `predicate`.
    Atom {
        key: String,
        predicate: Predicate,
    },
    And(Vec<Filter>),
    Or(Vec<Filter>),
    Not(Box<Filter>),
}

impl Default for Filter {
    fn default() -> Self {
        Filter::All
    }
}

impl Filter {
    pub fn eq<K: Into<String>>(key: K, value: MetaValue) -> Self {
        Filter::Atom {
            key: key.into(),
            predicate: Predicate::Eq(value),
        }
    }

    pub fn exists<K: Into<String>>(key: K) -> Self {
        Filter::Atom {
            key: key.into(),
            predicate: Predicate::Exists,
        }
    }

    pub fn and(filters: impl IntoIterator<Item = Filter>) -> Self {
        Filter::And(filters.into_iter().collect())
    }

    pub fn or(filters: impl IntoIterator<Item = Filter>) -> Self {
        Filter::Or(filters.into_iter().collect())
    }

    pub fn not(f: Filter) -> Self {
        Filter::Not(Box::new(f))
    }

    /// Returns true if this filter matches every document.
    pub fn is_all(&self) -> bool {
        matches!(self, Filter::All)
    }

    /// Parse a single CLI filter expression.
    ///
    /// Supported syntax (all whitespace tolerant):
    ///   `key`                  → `Exists`
    ///   `!key`                 → `Missing`
    ///   `key=value`            → `Eq`
    ///   `key!=value`           → `Ne`
    ///   `key>value` `>=` `<` `<=` → numeric comparisons
    ///   `key in [a,b,c]`       → `In`
    ///   `key contains "text"`  → case-insensitive substring
    ///
    /// Values are parsed with [`MetaValue::parse_cli`].
    pub fn parse_cli(expr: &str) -> Result<Filter, Error> {
        let s = expr.trim();
        if s.is_empty() {
            return Err(Error::invalid("empty filter expression"));
        }
        if let Some(rest) = s.strip_prefix('!') {
            let key = rest.trim();
            if key.is_empty() {
                return Err(Error::invalid("filter `!` requires a key"));
            }
            return Ok(Filter::Atom {
                key: key.to_string(),
                predicate: Predicate::Missing,
            });
        }

        // " in " / " contains " (whitespace-bounded keywords)
        if let Some((key, list)) = split_keyword(s, " in ") {
            let inner = list
                .trim()
                .strip_prefix('[')
                .and_then(|x| x.strip_suffix(']'))
                .ok_or_else(|| Error::invalid("filter `in` expects [a,b,c]"))?;
            let values = inner
                .split(',')
                .map(|v| MetaValue::parse_cli(v.trim()))
                .collect();
            return Ok(Filter::Atom {
                key: key.trim().to_string(),
                predicate: Predicate::In(values),
            });
        }
        if let Some((key, val)) = split_keyword(s, " contains ") {
            let v = val.trim().trim_matches('"').trim_matches('\'').to_string();
            return Ok(Filter::Atom {
                key: key.trim().to_string(),
                predicate: Predicate::Contains(v),
            });
        }

        // Comparison operators: order matters — match longer ops first.
        for (op, ctor) in [
            (">=", Predicate::Ge as fn(MetaValue) -> Predicate),
            ("<=", Predicate::Le),
            ("!=", Predicate::Ne),
            ("=", Predicate::Eq),
            (">", Predicate::Gt),
            ("<", Predicate::Lt),
        ] {
            if let Some(idx) = s.find(op) {
                let (k, rest) = s.split_at(idx);
                let v = &rest[op.len()..];
                let key = k.trim();
                if key.is_empty() {
                    return Err(Error::invalid(format!("filter `{}` missing key", op)));
                }
                return Ok(Filter::Atom {
                    key: key.to_string(),
                    predicate: ctor(MetaValue::parse_cli(v.trim())),
                });
            }
        }

        // Bare key → Exists
        Ok(Filter::Atom {
            key: s.to_string(),
            predicate: Predicate::Exists,
        })
    }

    /// Combine multiple CLI `--filter` invocations with AND.
    pub fn parse_cli_many<I, S>(exprs: I) -> Result<Filter, Error>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let parsed: Result<Vec<_>, _> = exprs
            .into_iter()
            .map(|s| Filter::parse_cli(s.as_ref()))
            .collect();
        let v = parsed?;
        Ok(match v.len() {
            0 => Filter::All,
            1 => v.into_iter().next().unwrap(),
            _ => Filter::And(v),
        })
    }
}

/// Split `s` at the first whitespace-bounded keyword (e.g. " in ", " contains ").
fn split_keyword<'a>(s: &'a str, kw: &str) -> Option<(&'a str, &'a str)> {
    let lower = s.to_ascii_lowercase();
    let idx = lower.find(kw)?;
    Some((&s[..idx], &s[idx + kw.len()..]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_eq() {
        let f = Filter::parse_cli("lang=rust").unwrap();
        match f {
            Filter::Atom {
                key,
                predicate: Predicate::Eq(MetaValue::Str(s)),
            } => {
                assert_eq!(key, "lang");
                assert_eq!(s, "rust");
            }
            _ => panic!("wrong: {:?}", f),
        }
    }

    #[test]
    fn parse_numeric_ge() {
        let f = Filter::parse_cli("score>=10").unwrap();
        match f {
            Filter::Atom {
                key,
                predicate: Predicate::Ge(MetaValue::Int(10)),
            } => {
                assert_eq!(key, "score");
            }
            _ => panic!("wrong: {:?}", f),
        }
    }

    #[test]
    fn parse_exists_missing() {
        assert!(matches!(
            Filter::parse_cli("foo").unwrap(),
            Filter::Atom {
                predicate: Predicate::Exists,
                ..
            }
        ));
        assert!(matches!(
            Filter::parse_cli("!foo").unwrap(),
            Filter::Atom {
                predicate: Predicate::Missing,
                ..
            }
        ));
    }

    #[test]
    fn parse_in_list() {
        let f = Filter::parse_cli("tag in [a,b,c]").unwrap();
        match f {
            Filter::Atom {
                key,
                predicate: Predicate::In(values),
            } => {
                assert_eq!(key, "tag");
                assert_eq!(values.len(), 3);
            }
            _ => panic!("wrong: {:?}", f),
        }
    }

    #[test]
    fn parse_contains() {
        let f = Filter::parse_cli(r#"title contains "rust""#).unwrap();
        assert!(matches!(
            f,
            Filter::Atom { predicate: Predicate::Contains(ref s), .. } if s == "rust"
        ));
    }

    #[test]
    fn parse_many_combines_with_and() {
        let f = Filter::parse_cli_many(["lang=rust", "score>=10"]).unwrap();
        match f {
            Filter::And(v) => assert_eq!(v.len(), 2),
            _ => panic!("expected And"),
        }
    }

    #[test]
    fn empty_is_invalid() {
        assert!(Filter::parse_cli("").is_err());
        assert!(Filter::parse_cli("!").is_err());
    }
}
