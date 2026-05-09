//! Numerical assertions used by parity and recall tests.

/// Cosine similarity for arbitrary (not necessarily normalized) vectors.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-12);
    dot / denom
}

/// Recall@k of `predicted` against `ground_truth`.
///
/// Both arguments are flat lists of doc ids; only the first `k` of
/// `predicted` are considered. Repeated ids in `predicted` are
/// deduplicated before counting.
pub fn recall_at_k(predicted: &[String], ground_truth: &[String], k: usize) -> f32 {
    if ground_truth.is_empty() {
        return 0.0;
    }
    let mut seen = std::collections::HashSet::new();
    for p in predicted.iter().take(k) {
        seen.insert(p.clone());
    }
    let hits = ground_truth.iter().filter(|g| seen.contains(*g)).count();
    hits as f32 / ground_truth.len() as f32
}

/// Symmetric Jaccard overlap of two top-k result sets.
pub fn jaccard(a: &[String], b: &[String]) -> f32 {
    let sa: std::collections::HashSet<_> = a.iter().collect();
    let sb: std::collections::HashSet<_> = b.iter().collect();
    if sa.is_empty() && sb.is_empty() {
        return 1.0;
    }
    let inter = sa.intersection(&sb).count() as f32;
    let union = sa.union(&sb).count() as f32;
    inter / union
}

/// NDCG@k with binary relevance.
///
/// `relevant` is the set of relevant doc ids; ranking is the predicted
/// order. NDCG is the discounted cumulative gain divided by the ideal
/// gain. Result is in `[0.0, 1.0]`.
pub fn ndcg_at_k(
    ranking: &[String],
    relevant: &std::collections::HashSet<String>,
    k: usize,
) -> f32 {
    let mut dcg = 0.0f32;
    for (i, doc) in ranking.iter().take(k).enumerate() {
        if relevant.contains(doc) {
            // gain = 1, discount = log2(rank + 1)
            dcg += 1.0 / ((i as f32 + 2.0).log2());
        }
    }
    let ideal_count = relevant.len().min(k);
    let mut idcg = 0.0f32;
    for i in 0..ideal_count {
        idcg += 1.0 / ((i as f32 + 2.0).log2());
    }
    if idcg <= 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

/// Mean reciprocal rank @ k.
///
/// Returns `1 / rank_of_first_relevant` (rank starting at 1) if any of the
/// first `k` predicted ids is in `relevant`; otherwise 0.0.
pub fn mrr_at_k(ranking: &[String], relevant: &std::collections::HashSet<String>, k: usize) -> f32 {
    for (i, doc) in ranking.iter().take(k).enumerate() {
        if relevant.contains(doc) {
            return 1.0 / (i as f32 + 1.0);
        }
    }
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_basic() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }

    #[test]
    fn recall_full_overlap() {
        let p = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let gt = vec!["a".to_string(), "c".to_string()];
        assert_eq!(recall_at_k(&p, &gt, 3), 1.0);
    }

    #[test]
    fn recall_partial() {
        let p = vec!["a".to_string(), "x".to_string()];
        let gt = vec!["a".to_string(), "b".to_string()];
        assert!((recall_at_k(&p, &gt, 2) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn jaccard_basic() {
        let a = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let b = vec!["b".to_string(), "c".to_string(), "d".to_string()];
        // intersection 2, union 4 -> 0.5
        assert!((jaccard(&a, &b) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn ndcg_perfect() {
        let r: std::collections::HashSet<_> =
            ["a".to_string(), "b".to_string()].into_iter().collect();
        let pred = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        assert!((ndcg_at_k(&pred, &r, 3) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn mrr_first_position() {
        let r: std::collections::HashSet<_> = ["a".to_string()].into_iter().collect();
        let pred = vec!["a".to_string(), "b".to_string()];
        assert!((mrr_at_k(&pred, &r, 5) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn mrr_third_position() {
        let r: std::collections::HashSet<_> = ["c".to_string()].into_iter().collect();
        let pred = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        assert!((mrr_at_k(&pred, &r, 5) - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn mrr_miss_returns_zero() {
        let r: std::collections::HashSet<_> = ["z".to_string()].into_iter().collect();
        let pred = vec!["a".to_string(), "b".to_string()];
        assert_eq!(mrr_at_k(&pred, &r, 5), 0.0);
    }
}
