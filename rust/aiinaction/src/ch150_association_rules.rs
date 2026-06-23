//! Association rule and frequent-pattern mining from scratch (Rust).
//!
//! Mirrors the Python module `aiinaction.ch150_association_rules` and the Julia
//! module `AIInAction.Ch150AssociationRules`. Implements Apriori (Agrawal and
//! Srikant, 1994) and FP-Growth (Han, Pei and Yin, 2000) plus association-rule
//! extraction with support, confidence, lift, leverage and conviction.
//!
//! Items are `i64`; a transaction is a set of items. Frequent itemsets are
//! returned as a map from a sorted `Vec<i64>` to its support, so Apriori and
//! FP-Growth return identical results, which the shared fixtures rely on.
//! std-only: no external dependencies.

use std::collections::BTreeMap;
use std::collections::BTreeSet;

/// A frequent itemset (sorted, deduplicated item ids).
pub type Itemset = Vec<i64>;

/// A single association rule with its interestingness measures.
#[derive(Clone, Debug)]
pub struct Rule {
    pub antecedent: Vec<i64>,
    pub consequent: Vec<i64>,
    pub support: f64,
    pub confidence: f64,
    pub lift: f64,
    pub leverage: f64,
    /// `f64::INFINITY` for a perfectly confident rule.
    pub conviction: f64,
}

fn normalize(transactions: &[Vec<i64>]) -> Result<Vec<BTreeSet<i64>>, String> {
    if transactions.is_empty() {
        return Err("transactions must be non-empty".to_string());
    }
    Ok(transactions
        .iter()
        .map(|t| t.iter().copied().collect::<BTreeSet<i64>>())
        .collect())
}

fn check_min_support(min_support: f64) -> Result<(), String> {
    if !(min_support > 0.0 && min_support <= 1.0) {
        return Err(format!("min_support must be in (0, 1], got {}", min_support));
    }
    Ok(())
}

/// Smallest integer count meeting the support threshold (ceil with epsilon).
fn min_count(min_support: f64, n: usize) -> usize {
    let raw = (min_support * n as f64 - 1e-9).ceil();
    let c = if raw < 1.0 { 1.0 } else { raw };
    c as usize
}

/// Fraction of transactions that contain every item of `itemset`.
pub fn support(transactions: &[Vec<i64>], itemset: &[i64]) -> Result<f64, String> {
    let data = normalize(transactions)?;
    let target: BTreeSet<i64> = itemset.iter().copied().collect();
    if target.is_empty() {
        return Err("itemset must be non-empty".to_string());
    }
    let cover = data.iter().filter(|t| target.is_subset(t)).count();
    Ok(cover as f64 / data.len() as f64)
}

fn frequent_singletons(data: &[BTreeSet<i64>], mc: usize) -> BTreeMap<i64, usize> {
    let mut counts: BTreeMap<i64, usize> = BTreeMap::new();
    for t in data {
        for &it in t {
            *counts.entry(it).or_insert(0) += 1;
        }
    }
    counts.into_iter().filter(|&(_, c)| c >= mc).collect()
}

/// Join + prune step. `prev` is a sorted list of frequent (k-1)-itemsets.
fn apriori_gen(prev: &[Itemset]) -> Vec<Itemset> {
    if prev.is_empty() {
        return Vec::new();
    }
    let prev_set: BTreeSet<&Itemset> = prev.iter().collect();
    let k = prev[0].len() + 1;
    let mut candidates: Vec<Itemset> = Vec::new();
    for i in 0..prev.len() {
        for j in (i + 1)..prev.len() {
            let a = &prev[i];
            let b = &prev[j];
            let a_pref = &a[..a.len() - 1];
            let b_pref = &b[..b.len() - 1];
            if a_pref == b_pref && a[a.len() - 1] < b[b.len() - 1] {
                let mut cand = a.clone();
                cand.push(b[b.len() - 1]);
                // Prune: every (k-1)-subset must be frequent.
                let mut ok = true;
                for skip in 0..cand.len() {
                    let sub: Itemset = cand
                        .iter()
                        .enumerate()
                        .filter(|&(idx, _)| idx != skip)
                        .map(|(_, &v)| v)
                        .collect();
                    if sub.len() == k - 1 && !prev_set.contains(&sub) {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    candidates.push(cand);
                }
            } else if a_pref != b_pref {
                break;
            }
        }
    }
    candidates.sort();
    candidates
}

/// Mine all frequent itemsets with the Apriori level-wise algorithm.
pub fn apriori(
    transactions: &[Vec<i64>],
    min_support: f64,
) -> Result<BTreeMap<Itemset, f64>, String> {
    let data = normalize(transactions)?;
    check_min_support(min_support)?;
    let n = data.len();
    let mc = min_count(min_support, n);

    let mut frequent: BTreeMap<Itemset, usize> = BTreeMap::new();
    let singles = frequent_singletons(&data, mc);
    let mut level: Vec<Itemset> = singles.keys().map(|&it| vec![it]).collect();
    level.sort();
    for it in &level {
        frequent.insert(it.clone(), singles[&it[0]]);
    }

    while !level.is_empty() {
        let candidates = apriori_gen(&level);
        if candidates.is_empty() {
            break;
        }
        let mut counts: Vec<usize> = vec![0; candidates.len()];
        for t in &data {
            for (ci, c) in candidates.iter().enumerate() {
                if c.iter().all(|item| t.contains(item)) {
                    counts[ci] += 1;
                }
            }
        }
        let mut next: Vec<Itemset> = Vec::new();
        for (ci, c) in candidates.iter().enumerate() {
            if counts[ci] >= mc {
                next.push(c.clone());
                frequent.insert(c.clone(), counts[ci]);
            }
        }
        next.sort();
        level = next;
    }

    Ok(frequent
        .into_iter()
        .map(|(k, c)| (k, c as f64 / n as f64))
        .collect())
}

// --------------------------------------------------------------------------- //
// FP-Growth
// --------------------------------------------------------------------------- //

/// FP-tree node stored in an arena; children/links are node indices.
struct FpNode {
    item: i64,
    count: usize,
    parent: usize,
    children: BTreeMap<i64, usize>,
    link: Option<usize>,
}

struct FpTree {
    nodes: Vec<FpNode>,
    header: BTreeMap<i64, usize>, // item -> head node index
}

impl FpTree {
    fn new() -> FpTree {
        // node 0 is the root with sentinel item.
        FpTree {
            nodes: vec![FpNode {
                item: i64::MIN,
                count: 0,
                parent: usize::MAX,
                children: BTreeMap::new(),
                link: None,
            }],
            header: BTreeMap::new(),
        }
    }

    fn link(&mut self, idx: usize) {
        let item = self.nodes[idx].item;
        match self.header.get(&item).copied() {
            None => {
                self.header.insert(item, idx);
            }
            Some(mut head) => {
                while let Some(next) = self.nodes[head].link {
                    head = next;
                }
                self.nodes[head].link = Some(idx);
            }
        }
    }

    /// Insert an ordered item list with the given multiplicity.
    fn insert(&mut self, ordered: &[i64], weight: usize) {
        let mut node = 0usize;
        for &it in ordered {
            let child = self.nodes[node].children.get(&it).copied();
            let child_idx = match child {
                Some(c) => c,
                None => {
                    let new_idx = self.nodes.len();
                    self.nodes.push(FpNode {
                        item: it,
                        count: 0,
                        parent: node,
                        children: BTreeMap::new(),
                        link: None,
                    });
                    self.nodes[node].children.insert(it, new_idx);
                    self.link(new_idx);
                    new_idx
                }
            };
            self.nodes[child_idx].count += weight;
            node = child_idx;
        }
    }

    /// Path of items from a node's parent up to (excluding) the root.
    fn ascend(&self, idx: usize) -> Vec<i64> {
        let mut path = Vec::new();
        let mut cur = self.nodes[idx].parent;
        while cur != usize::MAX && self.nodes[cur].item != i64::MIN {
            path.push(self.nodes[cur].item);
            cur = self.nodes[cur].parent;
        }
        path
    }
}

/// Order key: descending support, ties by ascending item id.
fn order_items(items: &[i64], counts: &BTreeMap<i64, usize>) -> Vec<i64> {
    let mut v: Vec<i64> = items.to_vec();
    v.sort_by(|&x, &y| {
        let cx = counts.get(&x).copied().unwrap_or(0);
        let cy = counts.get(&y).copied().unwrap_or(0);
        cy.cmp(&cx).then(x.cmp(&y))
    });
    v
}

fn build_tree(
    data: &[(Vec<i64>, usize)],
    counts: &BTreeMap<i64, usize>,
    frequent_items: &BTreeSet<i64>,
) -> FpTree {
    let mut tree = FpTree::new();
    for (items, weight) in data {
        let filtered: Vec<i64> = items
            .iter()
            .copied()
            .filter(|it| frequent_items.contains(it))
            .collect();
        let ordered = order_items(&filtered, counts);
        if !ordered.is_empty() {
            tree.insert(&ordered, *weight);
        }
    }
    tree
}

fn mine_tree(
    tree: &FpTree,
    counts: &BTreeMap<i64, usize>,
    mc: usize,
    suffix: &[i64],
    frequent: &mut BTreeMap<Itemset, usize>,
) {
    // Ascending support order (least frequent first).
    let mut items: Vec<i64> = tree.header.keys().copied().collect();
    items.sort_by(|&x, &y| counts[&x].cmp(&counts[&y]).then(x.cmp(&y)));

    for item in items {
        let mut new_suffix: Itemset = suffix.to_vec();
        new_suffix.push(item);
        new_suffix.sort();
        frequent.insert(new_suffix.clone(), counts[&item]);

        // Conditional pattern base for `item`.
        let mut cond_patterns: Vec<(Vec<i64>, usize)> = Vec::new();
        let mut node = tree.header.get(&item).copied();
        while let Some(idx) = node {
            let cnt = tree.nodes[idx].count;
            let prefix = tree.ascend(idx);
            if !prefix.is_empty() {
                cond_patterns.push((prefix, cnt));
            }
            node = tree.nodes[idx].link;
        }

        // Counts within the conditional pattern base.
        let mut cond_counts: BTreeMap<i64, usize> = BTreeMap::new();
        for (prefix, cnt) in &cond_patterns {
            for &it in prefix {
                *cond_counts.entry(it).or_insert(0) += cnt;
            }
        }
        let cond_frequent: BTreeMap<i64, usize> = cond_counts
            .iter()
            .filter(|&(_, &c)| c >= mc)
            .map(|(&k, &v)| (k, v))
            .collect();
        if cond_frequent.is_empty() {
            continue;
        }
        let cond_items: BTreeSet<i64> = cond_frequent.keys().copied().collect();

        // Conditional data is the filtered pattern base with multiplicities.
        let cond_data: Vec<(Vec<i64>, usize)> = cond_patterns
            .iter()
            .map(|(prefix, cnt)| {
                let kept: Vec<i64> = prefix
                    .iter()
                    .copied()
                    .filter(|it| cond_items.contains(it))
                    .collect();
                (kept, *cnt)
            })
            .filter(|(kept, _)| !kept.is_empty())
            .collect();

        let cond_tree = build_tree(&cond_data, &cond_frequent, &cond_items);
        if !cond_tree.header.is_empty() {
            mine_tree(&cond_tree, &cond_frequent, mc, &new_suffix, frequent);
        }
    }
}

/// Mine all frequent itemsets with FP-Growth. Returns the same map as `apriori`.
pub fn fpgrowth(
    transactions: &[Vec<i64>],
    min_support: f64,
) -> Result<BTreeMap<Itemset, f64>, String> {
    let data = normalize(transactions)?;
    check_min_support(min_support)?;
    let n = data.len();
    let mc = min_count(min_support, n);

    let mut counts: BTreeMap<i64, usize> = BTreeMap::new();
    for t in &data {
        for &it in t {
            *counts.entry(it).or_insert(0) += 1;
        }
    }
    let frequent_items: BTreeSet<i64> = counts
        .iter()
        .filter(|&(_, &c)| c >= mc)
        .map(|(&k, _)| k)
        .collect();

    let weighted: Vec<(Vec<i64>, usize)> = data
        .iter()
        .map(|t| (t.iter().copied().collect::<Vec<i64>>(), 1usize))
        .collect();
    let tree = build_tree(&weighted, &counts, &frequent_items);

    let mut frequent: BTreeMap<Itemset, usize> = BTreeMap::new();
    mine_tree(&tree, &counts, mc, &[], &mut frequent);

    Ok(frequent
        .into_iter()
        .map(|(k, c)| (k, c as f64 / n as f64))
        .collect())
}

// --------------------------------------------------------------------------- //
// Rule generation
// --------------------------------------------------------------------------- //

fn subsets_of_size(items: &[i64], r: usize) -> Vec<Vec<i64>> {
    let mut out = Vec::new();
    let n = items.len();
    if r == 0 || r > n {
        return out;
    }
    let mut idx: Vec<usize> = (0..r).collect();
    loop {
        out.push(idx.iter().map(|&i| items[i]).collect::<Vec<i64>>());
        // advance combination indices
        let mut i = r;
        loop {
            if i == 0 {
                return out;
            }
            i -= 1;
            if idx[i] != i + n - r {
                break;
            }
        }
        idx[i] += 1;
        for j in (i + 1)..r {
            idx[j] = idx[j - 1] + 1;
        }
    }
}

/// Extract association rules from a map of frequent itemsets to supports.
pub fn association_rules(
    frequent_itemsets: &BTreeMap<Itemset, f64>,
    min_confidence: f64,
) -> Result<Vec<Rule>, String> {
    if !(0.0..=1.0).contains(&min_confidence) {
        return Err(format!(
            "min_confidence must be in [0, 1], got {}",
            min_confidence
        ));
    }
    let mut rules: Vec<Rule> = Vec::new();
    for (z, &supp_z) in frequent_itemsets {
        if z.len() < 2 {
            continue;
        }
        for r in 1..z.len() {
            for antecedent in subsets_of_size(z, r) {
                let aset: BTreeSet<i64> = antecedent.iter().copied().collect();
                let consequent: Vec<i64> =
                    z.iter().copied().filter(|it| !aset.contains(it)).collect();
                let supp_a = match frequent_itemsets.get(&antecedent) {
                    Some(&v) => v,
                    None => continue,
                };
                let supp_c = match frequent_itemsets.get(&consequent) {
                    Some(&v) => v,
                    None => continue,
                };
                let conf = supp_z / supp_a;
                if conf < min_confidence {
                    continue;
                }
                let lift = conf / supp_c;
                let leverage = supp_z - supp_a * supp_c;
                let conviction = if conf >= 1.0 {
                    f64::INFINITY
                } else {
                    (1.0 - supp_c) / (1.0 - conf)
                };
                rules.push(Rule {
                    antecedent,
                    consequent,
                    support: supp_z,
                    confidence: conf,
                    lift,
                    leverage,
                    conviction,
                });
            }
        }
    }
    rules.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap()
            .then(a.antecedent.cmp(&b.antecedent))
            .then(a.consequent.cmp(&b.consequent))
    });
    Ok(rules)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shared fixtures: identical to the Python and Julia suites.
    fn fixture() -> Vec<Vec<i64>> {
        vec![
            vec![1, 2, 5],
            vec![2, 4],
            vec![2, 3],
            vec![1, 2, 4],
            vec![1, 3],
            vec![2, 3],
            vec![1, 3],
            vec![1, 2, 3, 5],
            vec![1, 2, 3],
        ]
    }

    const MIN_SUPPORT: f64 = 2.0 / 9.0;
    const TOL: f64 = 1e-9;

    fn expected_itemsets() -> Vec<(Vec<i64>, f64)> {
        vec![
            (vec![1], 6.0 / 9.0),
            (vec![2], 7.0 / 9.0),
            (vec![3], 6.0 / 9.0),
            (vec![4], 2.0 / 9.0),
            (vec![5], 2.0 / 9.0),
            (vec![1, 2], 4.0 / 9.0),
            (vec![1, 3], 4.0 / 9.0),
            (vec![1, 5], 2.0 / 9.0),
            (vec![2, 3], 4.0 / 9.0),
            (vec![2, 4], 2.0 / 9.0),
            (vec![2, 5], 2.0 / 9.0),
            (vec![1, 2, 3], 2.0 / 9.0),
            (vec![1, 2, 5], 2.0 / 9.0),
        ]
    }

    #[test]
    fn apriori_finds_expected_itemsets() {
        let fis = apriori(&fixture(), MIN_SUPPORT).unwrap();
        assert_eq!(fis.len(), 13);
        for (k, v) in expected_itemsets() {
            assert!((fis[&k] - v).abs() < TOL, "itemset {:?}", k);
        }
    }

    #[test]
    fn fpgrowth_matches_apriori() {
        let a = apriori(&fixture(), MIN_SUPPORT).unwrap();
        let f = fpgrowth(&fixture(), MIN_SUPPORT).unwrap();
        let ak: Vec<&Itemset> = a.keys().collect();
        let fk: Vec<&Itemset> = f.keys().collect();
        assert_eq!(ak, fk);
        for (k, v) in &a {
            assert!((f[k] - v).abs() < TOL);
        }
    }

    #[test]
    fn fpgrowth_supports_match_fixture() {
        let f = fpgrowth(&fixture(), MIN_SUPPORT).unwrap();
        for (k, v) in expected_itemsets() {
            assert!((f[&k] - v).abs() < TOL, "itemset {:?}", k);
        }
    }

    #[test]
    fn support_helper() {
        let d = fixture();
        assert!((support(&d, &[1, 2]).unwrap() - 4.0 / 9.0).abs() < TOL);
        assert!((support(&d, &[2]).unwrap() - 7.0 / 9.0).abs() < TOL);
        assert!(support(&d, &[4, 5]).unwrap().abs() < TOL);
    }

    #[test]
    fn high_confidence_rules() {
        let fis = apriori(&fixture(), MIN_SUPPORT).unwrap();
        let rules = association_rules(&fis, 0.7).unwrap();
        assert_eq!(rules.len(), 6);
        let mut keys: Vec<(Vec<i64>, Vec<i64>)> = rules
            .iter()
            .map(|r| (r.antecedent.clone(), r.consequent.clone()))
            .collect();
        keys.sort();
        let mut expected = vec![
            (vec![1, 5], vec![2]),
            (vec![2, 5], vec![1]),
            (vec![4], vec![2]),
            (vec![5], vec![1]),
            (vec![5], vec![1, 2]),
            (vec![5], vec![2]),
        ];
        expected.sort();
        assert_eq!(keys, expected);
    }

    #[test]
    fn rule_metrics_match_fixture() {
        let fis = apriori(&fixture(), MIN_SUPPORT).unwrap();
        let rules = association_rules(&fis, 0.7).unwrap();
        let r = rules
            .iter()
            .find(|r| r.antecedent == vec![5] && r.consequent == vec![1, 2])
            .unwrap();
        assert!((r.support - 2.0 / 9.0).abs() < TOL);
        assert!((r.confidence - 1.0).abs() < TOL);
        assert!((r.lift - 2.25).abs() < TOL);
        assert!((r.leverage - 0.12345679012345678).abs() < TOL);
        assert!(r.conviction.is_infinite());

        let r2 = rules
            .iter()
            .find(|r| r.antecedent == vec![2, 5] && r.consequent == vec![1])
            .unwrap();
        assert!((r2.lift - 1.5).abs() < TOL);
        assert!((r2.leverage - 0.07407407407407407).abs() < TOL);
    }

    #[test]
    fn finite_conviction_rule() {
        let fis = apriori(&fixture(), MIN_SUPPORT).unwrap();
        let rules = association_rules(&fis, 0.5).unwrap();
        let r = rules
            .iter()
            .find(|r| r.antecedent == vec![1] && r.consequent == vec![2])
            .unwrap();
        assert!((r.confidence - 2.0 / 3.0).abs() < TOL);
        assert!((r.lift - 0.8571428571428571).abs() < TOL);
        assert!((r.leverage - (-0.07407407407407407)).abs() < TOL);
        assert!((r.conviction - 0.6666666666666665).abs() < TOL);
    }

    #[test]
    fn rules_sorted_by_descending_confidence() {
        let fis = apriori(&fixture(), MIN_SUPPORT).unwrap();
        let rules = association_rules(&fis, 0.0).unwrap();
        for w in rules.windows(2) {
            assert!(w[0].confidence >= w[1].confidence - TOL);
        }
    }

    #[test]
    fn empty_transactions_errors() {
        assert!(apriori(&[], 0.5).is_err());
    }

    #[test]
    fn bad_min_support_errors() {
        assert!(apriori(&fixture(), 0.0).is_err());
        assert!(apriori(&fixture(), 1.5).is_err());
    }

    #[test]
    fn bad_min_confidence_errors() {
        let fis = apriori(&fixture(), MIN_SUPPORT).unwrap();
        assert!(association_rules(&fis, 1.5).is_err());
    }

    #[test]
    fn dense_data_agreement() {
        let data: Vec<Vec<i64>> = vec![
            vec![1, 2, 3, 4],
            vec![1, 2, 3],
            vec![1, 2, 4],
            vec![1, 2],
            vec![2, 3, 4],
        ];
        for &ms in &[0.2f64, 0.4, 0.6, 0.8] {
            let a = apriori(&data, ms).unwrap();
            let f = fpgrowth(&data, ms).unwrap();
            let ak: Vec<&Itemset> = a.keys().collect();
            let fk: Vec<&Itemset> = f.keys().collect();
            assert_eq!(ak, fk, "min_support {}", ms);
            for (k, v) in &a {
                assert!((f[k] - v).abs() < TOL);
            }
        }
    }
}
