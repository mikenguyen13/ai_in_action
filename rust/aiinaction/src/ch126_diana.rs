//! DIANA: divisive (top-down) hierarchical clustering.
//!
//! Mirrors the Python module `aiinaction.ch126_diana` and the Julia module
//! `AIInAction.Ch126Diana`. The shared fixtures in the tests below match the
//! Python/Julia suites, which is what keeps the three libraries at parity.
//!
//! The algorithm is DIANA (Kaufman & Rousseeuw) with the Macnaughton-Smith
//! splinter-group heuristic. std-only; distance matrices are `&[Vec<f64>]`.

/// A single recorded division produced by DIANA.
#[derive(Debug, Clone, PartialEq)]
pub struct Split {
    /// Sorted indices of the cluster that was split.
    pub parent: Vec<usize>,
    /// Sorted indices of the breakaway (splinter) group.
    pub splinter: Vec<usize>,
    /// Sorted indices of the objects that stayed behind.
    pub remainder: Vec<usize>,
    /// Diameter of `parent` at the time of the split (the split height).
    pub diameter: f64,
}

/// Validate that `dist` is a square, symmetric, hollow, non-negative matrix.
fn validate_matrix(dist: &[Vec<f64>]) -> Result<usize, String> {
    let n = dist.len();
    if n == 0 {
        return Err("distance matrix must be non-empty".to_string());
    }
    for row in dist {
        if row.len() != n {
            return Err(format!(
                "distance matrix must be square 2-D; got a row of length {} for n={}",
                row.len(),
                n
            ));
        }
    }
    for i in 0..n {
        for j in 0..n {
            let v = dist[i][j];
            if !v.is_finite() {
                return Err("distance matrix must contain only finite values".to_string());
            }
            if v < 0.0 {
                return Err("distance matrix must be non-negative".to_string());
            }
            if (v - dist[j][i]).abs() > 1e-12 {
                return Err("distance matrix must be symmetric".to_string());
            }
        }
        if dist[i][i].abs() > 1e-12 {
            return Err("distance matrix must have a zero diagonal".to_string());
        }
    }
    Ok(n)
}

/// Validate a member list against a matrix of size `n`.
fn check_members(members: &[usize], n: usize) -> Result<(), String> {
    if members.is_empty() {
        return Err("member list must be non-empty".to_string());
    }
    let mut seen = vec![false; n];
    for &i in members {
        if i >= n {
            return Err(format!("member index {} out of range [0, {})", i, n));
        }
        if seen[i] {
            return Err(format!("duplicate member index {}", i));
        }
        seen[i] = true;
    }
    Ok(())
}

/// Return the diameter of `members`: the largest pairwise dissimilarity.
/// A cluster of fewer than two members has diameter `0.0`.
pub fn diameter(dist: &[Vec<f64>], members: &[usize]) -> Result<f64, String> {
    let n = validate_matrix(dist)?;
    check_members(members, n)?;
    if members.len() < 2 {
        return Ok(0.0);
    }
    let mut d = 0.0_f64;
    for (a, &i) in members.iter().enumerate() {
        for &j in &members[a + 1..] {
            if dist[i][j] > d {
                d = dist[i][j];
            }
        }
    }
    Ok(d)
}

fn avg_to(dist: &[Vec<f64>], i: usize, group: &[usize], exclude_self: bool) -> f64 {
    let mut sum = 0.0;
    let mut cnt = 0usize;
    for &j in group {
        if exclude_self && j == i {
            continue;
        }
        sum += dist[i][j];
        cnt += 1;
    }
    sum / cnt as f64
}

/// Split one cluster into `(splinter, remainder)` via Macnaughton-Smith.
/// Both returned vectors are sorted ascending.
pub fn macnaughton_smith_split(
    dist: &[Vec<f64>],
    members: &[usize],
) -> Result<(Vec<usize>, Vec<usize>), String> {
    let n = validate_matrix(dist)?;
    check_members(members, n)?;

    if members.len() < 2 {
        let mut rem = members.to_vec();
        rem.sort_unstable();
        return Ok((Vec::new(), rem));
    }

    let mut remainder: Vec<usize> = members.to_vec();
    let mut splinter: Vec<usize> = Vec::new();

    // Seed: object with largest average dissimilarity to the rest of the cluster.
    let mut best_seed = remainder[0];
    let mut best_avg = f64::NEG_INFINITY;
    for &i in &remainder {
        let avg = avg_to(dist, i, &remainder, true);
        if avg > best_avg {
            best_avg = avg;
            best_seed = i;
        }
    }
    remainder.retain(|&x| x != best_seed);
    splinter.push(best_seed);

    // Grow the splinter while some object prefers it.
    while remainder.len() > 1 {
        let mut best_obj: Option<usize> = None;
        let mut best_d = 0.0_f64;
        for &i in &remainder {
            let avg_rem = avg_to(dist, i, &remainder, true);
            let avg_spl = avg_to(dist, i, &splinter, false);
            let d_i = avg_rem - avg_spl;
            if d_i > best_d {
                best_d = d_i;
                best_obj = Some(i);
            }
        }
        match best_obj {
            Some(o) => {
                remainder.retain(|&x| x != o);
                splinter.push(o);
            }
            None => break,
        }
    }

    splinter.sort_unstable();
    remainder.sort_unstable();
    Ok((splinter, remainder))
}

fn sorted(mut v: Vec<usize>) -> Vec<usize> {
    v.sort_unstable();
    v
}

/// Run DIANA to completion, returning the ordered list of splits
/// (largest-diameter-first), one per internal dendrogram node (`n - 1` total).
pub fn diana(dist: &[Vec<f64>]) -> Result<Vec<Split>, String> {
    let n = validate_matrix(dist)?;
    if n == 1 {
        return Ok(Vec::new());
    }

    let mut clusters: Vec<Vec<usize>> = vec![(0..n).collect()];
    let mut splits: Vec<Split> = Vec::new();

    while !clusters.is_empty() {
        let mut best_k = 0usize;
        let mut best_diam = -1.0_f64;
        for (k, c) in clusters.iter().enumerate() {
            let dm = diameter(dist, c)?;
            if dm > best_diam {
                best_diam = dm;
                best_k = k;
            }
        }
        let target = clusters.remove(best_k);
        let (splinter, remainder) = macnaughton_smith_split(dist, &target)?;
        splits.push(Split {
            parent: sorted(target),
            splinter: splinter.clone(),
            remainder: remainder.clone(),
            diameter: best_diam,
        });
        for part in [splinter, remainder] {
            if part.len() > 1 {
                clusters.push(part);
            }
        }
    }

    Ok(splits)
}

/// Cut the DIANA hierarchy into exactly `k` flat clusters, labelled `0..k-1`
/// in order of each cluster's smallest member index.
pub fn diana_labels(dist: &[Vec<f64>], k: usize) -> Result<Vec<usize>, String> {
    let n = validate_matrix(dist)?;
    if k < 1 || k > n {
        return Err(format!("k must be in [1, {}]; got {}", n, k));
    }

    let mut clusters: Vec<Vec<usize>> = vec![(0..n).collect()];
    while clusters.len() < k {
        let mut best_i: isize = -1;
        let mut best_diam = -1.0_f64;
        for (i, c) in clusters.iter().enumerate() {
            if c.len() < 2 {
                continue;
            }
            let dm = diameter(dist, c)?;
            if dm > best_diam {
                best_diam = dm;
                best_i = i as isize;
            }
        }
        if best_i < 0 {
            break;
        }
        let target = clusters.remove(best_i as usize);
        let (splinter, remainder) = macnaughton_smith_split(dist, &target)?;
        clusters.push(splinter);
        clusters.push(remainder);
    }

    clusters.sort_by_key(|c| *c.iter().min().unwrap());
    let mut labels = vec![0usize; n];
    for (label, c) in clusters.iter().enumerate() {
        for &obj in c {
            labels[obj] = label;
        }
    }
    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shared fixture: 1-D points {1,2,3,20,21,22}, d = absolute difference.
    // Identical to the Python and Julia test suites.
    fn fixture() -> Vec<Vec<f64>> {
        let pts = [1.0, 2.0, 3.0, 20.0, 21.0, 22.0];
        let n = pts.len();
        (0..n)
            .map(|i| (0..n).map(|j| (pts[i] - pts[j]).abs()).collect())
            .collect()
    }

    const TOL: f64 = 1e-9;

    #[test]
    fn diameter_matches_fixture() {
        let d = fixture();
        assert!((diameter(&d, &[0, 1, 2, 3, 4, 5]).unwrap() - 21.0).abs() < TOL);
    }

    #[test]
    fn diameter_singleton_is_zero() {
        let d = fixture();
        assert_eq!(diameter(&d, &[3]).unwrap(), 0.0);
    }

    #[test]
    fn first_split_matches_fixture() {
        let d = fixture();
        let (spl, rem) = macnaughton_smith_split(&d, &[0, 1, 2, 3, 4, 5]).unwrap();
        assert_eq!(spl, vec![0, 1, 2]);
        assert_eq!(rem, vec![3, 4, 5]);
    }

    #[test]
    fn singleton_split() {
        let d = fixture();
        let (spl, rem) = macnaughton_smith_split(&d, &[2]).unwrap();
        assert!(spl.is_empty());
        assert_eq!(rem, vec![2]);
    }

    #[test]
    fn full_diana_split_count_and_order() {
        let d = fixture();
        let splits = diana(&d).unwrap();
        assert_eq!(splits.len(), 5);
        let first = &splits[0];
        assert_eq!(first.parent, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(first.splinter, vec![0, 1, 2]);
        assert_eq!(first.remainder, vec![3, 4, 5]);
        assert!((first.diameter - 21.0).abs() < TOL);
        // Diameters non-increasing in split order.
        for w in splits.windows(2) {
            assert!(w[0].diameter >= w[1].diameter - TOL);
        }
    }

    #[test]
    fn labels_k2() {
        let d = fixture();
        assert_eq!(diana_labels(&d, 2).unwrap(), vec![0, 0, 0, 1, 1, 1]);
    }

    #[test]
    fn labels_k3() {
        let d = fixture();
        assert_eq!(diana_labels(&d, 3).unwrap(), vec![0, 1, 1, 2, 2, 2]);
    }

    #[test]
    fn labels_k_equals_n() {
        let d = fixture();
        assert_eq!(diana_labels(&d, 6).unwrap(), vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn labels_k1() {
        let d = fixture();
        assert_eq!(diana_labels(&d, 1).unwrap(), vec![0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn single_object_dataset() {
        assert_eq!(diana(&[vec![0.0]]).unwrap().len(), 0);
        assert_eq!(diana_labels(&[vec![0.0]], 1).unwrap(), vec![0]);
    }

    #[test]
    fn asymmetric_matrix_errors() {
        assert!(diana(&[vec![0.0, 1.0], vec![2.0, 0.0]]).is_err());
    }

    #[test]
    fn nonzero_diagonal_errors() {
        assert!(diana(&[vec![1.0, 1.0], vec![1.0, 1.0]]).is_err());
    }

    #[test]
    fn negative_distance_errors() {
        assert!(diana(&[vec![0.0, -1.0], vec![-1.0, 0.0]]).is_err());
    }

    #[test]
    fn bad_member_errors() {
        let d = fixture();
        assert!(macnaughton_smith_split(&d, &[0, 99]).is_err());
        assert!(macnaughton_smith_split(&d, &[0, 0]).is_err());
    }

    #[test]
    fn bad_k_errors() {
        let d = fixture();
        assert!(diana_labels(&d, 0).is_err());
        assert!(diana_labels(&d, 7).is_err());
    }
}
