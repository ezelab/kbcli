//! Filter parser/evaluator coverage.

use kbcli_core::{Filter, MetaValue, Predicate};
use kbcli_store::filter_to_sql;

#[test]
fn parse_eq_neq() {
    let f = Filter::parse_cli("lang=rust").unwrap();
    assert!(matches!(
        f,
        Filter::Atom {
            predicate: Predicate::Eq(MetaValue::Str(_)),
            ..
        }
    ));
    let f = Filter::parse_cli("lang!=python").unwrap();
    assert!(matches!(
        f,
        Filter::Atom {
            predicate: Predicate::Ne(MetaValue::Str(_)),
            ..
        }
    ));
}

#[test]
fn parse_numeric_comparisons() {
    for (expr, want) in [
        ("score>10", "Gt"),
        ("score>=10", "Ge"),
        ("score<10", "Lt"),
        ("score<=10", "Le"),
    ] {
        let f = Filter::parse_cli(expr).unwrap();
        let dbg = format!("{f:?}");
        assert!(dbg.contains(want), "{expr} -> {dbg}");
    }
}

#[test]
fn parse_in_list_and_contains_and_exists() {
    let f = Filter::parse_cli("tag in [a,b,c]").unwrap();
    assert!(matches!(
        f,
        Filter::Atom {
            predicate: Predicate::In(_),
            ..
        }
    ));
    let f = Filter::parse_cli(r#"title contains "rust""#).unwrap();
    assert!(matches!(
        f,
        Filter::Atom {
            predicate: Predicate::Contains(_),
            ..
        }
    ));
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
fn many_filters_combine_with_and() {
    let f = Filter::parse_cli_many(["lang=rust", "score>=10"]).unwrap();
    match f {
        Filter::And(v) => assert_eq!(v.len(), 2),
        _ => panic!(),
    }
}

#[test]
fn filter_translates_to_sql() {
    let f = Filter::parse_cli("lang=rust").unwrap();
    let s = filter_to_sql(&f);
    assert!(s.sql.contains("json_extract"));
    assert!(!s.params.is_empty());
}

#[test]
fn empty_filter_invalid() {
    assert!(Filter::parse_cli("").is_err());
}
