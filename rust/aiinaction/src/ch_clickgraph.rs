//! Clickstream graphs: node embeddings, burstiness, graph attention, quantile loss (Rust).
//!
//! Mirrors the Python module `aiinaction.ch_clickgraph` and the Julia module
//! `AIInAction.ChClickgraph`. The shared fixtures in the tests below match the
//! Python/Julia suites. Every numeric function is ported one-to-one; the seeded
//! simulator `simulate_clickstream` is Python only.
//!
//! * **Graph construction.** [`session_graph`], [`average_graph`],
//!   [`normalize_edge_weights`], [`upstream_neighbors`].
//! * **Scaling and burstiness.** [`robust_median_scale`], [`burstiness`],
//!   [`neighbor_burstiness`], [`ema_last`], [`burst_forecast`].
//! * **Embeddings.** [`lstm_embed`] maps a time series to a vector;
//!   [`attention_neighbor_embedding`] is a one-head graph transformer layer with an edge
//!   term whose softmax weights act as learned conversion rates.
//! * **Forecast assembly and loss.** [`gated_forecast`], [`pinball_loss`],
//!   [`total_quantile_loss`], [`weighted_quantile_loss`], [`interval_coverage`].
//!
//! Matrices are row-major `Vec<Vec<f64>>`. Errors are `Err(String)` wherever the Python
//! module raises `ValueError`. Node ids are 0-based, as in Python.

/// A matrix stored as a vector of rows.
pub type Matrix = Vec<Vec<f64>>;

/// Result of [`robust_median_scale`].
#[derive(Debug, Clone, PartialEq)]
pub struct RobustScaled {
    pub median: f64,
    pub mad: f64,
    pub scaled: Vec<f64>,
}

fn median(a: &[f64]) -> f64 {
    let mut s = a.to_vec();
    s.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

/// Centers on the median and divides by the mean absolute deviation about the median.
/// A zero deviation (constant series) uses a divisor of 1.
pub fn robust_median_scale(x: &[f64]) -> Result<RobustScaled, String> {
    if x.is_empty() {
        return Err("x must be non-empty".to_string());
    }
    let med = median(x);
    let mad = x.iter().fold(0.0, |acc, v| acc + (v - med).abs()) / x.len() as f64;
    let div = if mad > 0.0 { mad } else { 1.0 };
    Ok(RobustScaled {
        median: med,
        mad,
        scaled: x.iter().map(|v| (v - med) / div).collect(),
    })
}

/// Lappas et al. burstiness of each single interval, `x_t / sum(x) - 1/T`. An all-zero
/// window returns zeros.
pub fn burstiness(x: &[f64]) -> Result<Vec<f64>, String> {
    if x.is_empty() {
        return Err("x must be non-empty".to_string());
    }
    if x.iter().any(|&v| v < 0.0) {
        return Err("counts must be non-negative".to_string());
    }
    let total = x.iter().fold(0.0, |a, &v| a + v);
    if total <= 0.0 {
        return Ok(vec![0.0; x.len()]);
    }
    let uniform = 1.0 / x.len() as f64;
    Ok(x.iter().map(|&v| v / total - uniform).collect())
}

/// Node counts and directed transition matrix from sessions (0-based node ids).
/// Consecutive repeats count as occurrences but not as transitions.
pub fn session_graph(sessions: &[Vec<usize>], n_nodes: usize) -> Result<(Vec<f64>, Matrix), String> {
    if n_nodes == 0 {
        return Err("n_nodes must be positive".to_string());
    }
    let mut counts = vec![0.0; n_nodes];
    let mut w = vec![vec![0.0; n_nodes]; n_nodes];
    for s in sessions {
        let mut prev: Option<usize> = None;
        for &node in s {
            if node >= n_nodes {
                return Err(format!("node id {} outside [0, {})", node, n_nodes));
            }
            counts[node] += 1.0;
            if let Some(p) = prev {
                if p != node {
                    w[p][node] += 1.0;
                }
            }
            prev = Some(node);
        }
    }
    Ok((counts, w))
}

/// Elementwise mean of daily adjacency matrices.
pub fn average_graph(graphs: &[Matrix]) -> Result<Matrix, String> {
    if graphs.is_empty() {
        return Err("need at least one graph".to_string());
    }
    let (r, c) = (graphs[0].len(), graphs[0][0].len());
    let mut acc = vec![vec![0.0; c]; r];
    for g in graphs {
        if g.len() != r || g.iter().any(|row| row.len() != c) {
            return Err("graphs must share a shape".to_string());
        }
        for i in 0..r {
            for j in 0..c {
                acc[i][j] += g[i][j];
            }
        }
    }
    let n = graphs.len() as f64;
    for row in acc.iter_mut() {
        for v in row.iter_mut() {
            *v /= n;
        }
    }
    Ok(acc)
}

/// Divides each edge `i -> j` by the average quantity of target `j` (zero if that is not
/// positive).
pub fn normalize_edge_weights(w: &Matrix, mean_counts: &[f64]) -> Result<Matrix, String> {
    let n = w.len();
    if w.iter().any(|row| row.len() != n) || mean_counts.len() != n {
        return Err("W must be square and mean_counts must match its size".to_string());
    }
    let mut out = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            if mean_counts[j] > 0.0 {
                out[i][j] = w[i][j] / mean_counts[j];
            }
        }
    }
    Ok(out)
}

/// Nodes `i != j` with a positive edge `i -> j`, ascending.
pub fn upstream_neighbors(w: &Matrix, j: usize) -> Vec<usize> {
    (0..w.len()).filter(|&i| i != j && w[i][j] > 0.0).collect()
}

/// Averages the burstiness series (rows of `b`) over the upstream neighbors; zeros if none.
pub fn neighbor_burstiness(b: &Matrix, upstream: &[usize]) -> Result<Vec<f64>, String> {
    if b.is_empty() {
        return Err("B must be non-empty".to_string());
    }
    let t = b[0].len();
    let mut out = vec![0.0; t];
    if upstream.is_empty() {
        return Ok(out);
    }
    for &u in upstream {
        if u >= b.len() {
            return Err(format!("upstream index {} out of range", u));
        }
        for k in 0..t {
            out[k] += b[u][k];
        }
    }
    let n = upstream.len() as f64;
    Ok(out.into_iter().map(|v| v / n).collect())
}

/// Final value of the EMA `s_t = a x_t + (1 - a) s_(t-1)`.
pub fn ema_last(x: &[f64], alpha: f64) -> Result<f64, String> {
    if !(alpha > 0.0 && alpha <= 1.0) {
        return Err("alpha must be in (0, 1]".to_string());
    }
    if x.is_empty() {
        return Err("x must be non-empty".to_string());
    }
    let mut s = x[0];
    for &v in &x[1..] {
        s = alpha * v + (1.0 - alpha) * s;
    }
    Ok(s)
}

/// Burst forecasts `tanh(raw) * trend`.
pub fn burst_forecast(raw: &[f64], trend: f64) -> Vec<f64> {
    raw.iter().map(|v| v.tanh() * trend).collect()
}

/// Numerically stable softmax.
pub fn softmax(v: &[f64]) -> Result<Vec<f64>, String> {
    if v.is_empty() {
        return Err("v must be non-empty".to_string());
    }
    let m = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = v.iter().map(|&a| (a - m).exp()).collect();
    let s: f64 = e.iter().sum();
    Ok(e.into_iter().map(|a| a / s).collect())
}

/// Logistic function, stable for large `|z|`.
pub fn sigmoid(z: f64) -> f64 {
    let ez = (-z.abs()).exp();
    if z >= 0.0 {
        1.0 / (1.0 + ez)
    } else {
        ez / (1.0 + ez)
    }
}

/// Deterministic test matrix `0.4 sin(0.7 (i cols + j + 1) + salt)` with 0-based `i`, `j`.
pub fn det_matrix(rows: usize, cols: usize, salt: f64) -> Matrix {
    (0..rows)
        .map(|i| {
            (0..cols)
                .map(|j| 0.4 * (0.7 * ((i * cols + j + 1) as f64) + salt).sin())
                .collect()
        })
        .collect()
}

fn matvec(m: &Matrix, v: &[f64]) -> Vec<f64> {
    m.iter()
        .map(|row| row.iter().zip(v).fold(0.0, |a, (x, y)| a + x * y))
        .collect()
}

/// Final hidden state of a single-layer LSTM. `x_seq` is `(T, d)`, `w` is `(4H, d + H)`
/// over `[x_t; h_(t-1)]` with gate blocks input, forget, cell candidate, output, and `b`
/// has length `4H`.
pub fn lstm_embed(x_seq: &Matrix, w: &Matrix, b: &[f64]) -> Result<Vec<f64>, String> {
    if x_seq.is_empty() {
        return Err("x_seq must be non-empty".to_string());
    }
    let d = x_seq[0].len();
    let hdim = w.len() / 4;
    if w.len() != 4 * hdim || w.iter().any(|r| r.len() != d + hdim) || b.len() != 4 * hdim {
        return Err("W must be (4H, d+H) and b must be (4H,)".to_string());
    }
    let mut h = vec![0.0; hdim];
    let mut c = vec![0.0; hdim];
    for x in x_seq {
        let mut inp = x.clone();
        inp.extend_from_slice(&h);
        let z: Vec<f64> = matvec(w, &inp).iter().zip(b).map(|(a, bb)| a + bb).collect();
        for k in 0..hdim {
            let i = sigmoid(z[k]);
            let f = sigmoid(z[hdim + k]);
            let g = z[2 * hdim + k].tanh();
            let o = sigmoid(z[3 * hdim + k]);
            c[k] = f * c[k] + i * g;
            h[k] = o * c[k].tanh();
        }
    }
    Ok(h)
}

/// One-head graph transformer layer. `q = Wq h_i`, `k_j = Wk h_j + we * e_ij`,
/// `v_j = Wv h_j`, `alpha = softmax(q . k_j / sqrt(d_k))`, `z = sum_j alpha_j v_j`.
/// Returns `(z, alpha)`; no neighbors gives `z = 0` and an empty `alpha`.
pub fn attention_neighbor_embedding(
    h_i: &[f64],
    h_nb: &Matrix,
    e_nb: &[f64],
    wq: &Matrix,
    wk: &Matrix,
    wv: &Matrix,
    we: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let dk = wq.len();
    if wk.len() != dk || we.len() != dk {
        return Err("Wq, Wk and we must share the key dimension".to_string());
    }
    if h_nb.is_empty() {
        return Ok((vec![0.0; wv.len()], Vec::new()));
    }
    if e_nb.len() != h_nb.len() {
        return Err("e_nb must have one entry per neighbor".to_string());
    }
    let q = matvec(wq, h_i);
    let mut scores = Vec::with_capacity(h_nb.len());
    for (hj, &ej) in h_nb.iter().zip(e_nb) {
        let kj: Vec<f64> = matvec(wk, hj)
            .iter()
            .zip(we)
            .map(|(a, w)| a + w * ej)
            .collect();
        let dot = kj.iter().zip(&q).fold(0.0, |a, (x, y)| a + x * y);
        scores.push(dot / (dk as f64).sqrt());
    }
    let alpha = softmax(&scores)?;
    let mut z = vec![0.0; wv.len()];
    for (hj, &a) in h_nb.iter().zip(&alpha) {
        let v = matvec(wv, hj);
        for k in 0..z.len() {
            z[k] += a * v[k];
        }
    }
    Ok((z, alpha))
}

/// Final forecast `base + burst + sigmoid(gate_logits) * neigh` (elementwise).
pub fn gated_forecast(
    base: &[f64],
    burst: &[f64],
    neigh: &[f64],
    gate_logits: &[f64],
) -> Result<Vec<f64>, String> {
    let n = base.len();
    if burst.len() != n || neigh.len() != n || gate_logits.len() != n {
        return Err("all inputs must have the same length".to_string());
    }
    Ok((0..n)
        .map(|k| base[k] + burst[k] + sigmoid(gate_logits[k]) * neigh[k])
        .collect())
}

/// Pinball loss `q (y - yhat)_+ + (1 - q) (yhat - y)_+`.
pub fn pinball_loss(y: f64, yhat: f64, q: f64) -> Result<f64, String> {
    if !(q > 0.0 && q < 1.0) {
        return Err("q must be in (0, 1)".to_string());
    }
    Ok(q * (y - yhat).max(0.0) + (1.0 - q) * (yhat - y).max(0.0))
}

/// Sum of pinball losses over horizon steps and quantiles (`preds` is `(H, Q)`).
pub fn total_quantile_loss(y: &[f64], preds: &Matrix, quantiles: &[f64]) -> Result<f64, String> {
    if preds.len() != y.len() || preds.iter().any(|r| r.len() != quantiles.len()) {
        return Err("preds must have shape (len(y), len(quantiles))".to_string());
    }
    let mut total = 0.0;
    for h in 0..y.len() {
        for (k, &q) in quantiles.iter().enumerate() {
            total += pinball_loss(y[h], preds[h][k], q)?;
        }
    }
    Ok(total)
}

/// Scale-free summary `2 sum pinball_q / sum |y|`.
pub fn weighted_quantile_loss(y: &[f64], yhat: &[f64], q: f64) -> Result<f64, String> {
    if y.len() != yhat.len() || y.is_empty() {
        return Err("y and yhat must be non-empty and the same length".to_string());
    }
    let denom: f64 = y.iter().map(|v| v.abs()).sum();
    if denom == 0.0 {
        return Err("sum |y| is zero; wQL is undefined".to_string());
    }
    let mut total = 0.0;
    for (a, b) in y.iter().zip(yhat) {
        total += pinball_loss(*a, *b, q)?;
    }
    Ok(2.0 * total / denom)
}

/// Fraction of observations inside `[lo, hi]` (inclusive).
pub fn interval_coverage(y: &[f64], lo: &[f64], hi: &[f64]) -> Result<f64, String> {
    if y.len() != lo.len() || y.len() != hi.len() || y.is_empty() {
        return Err("y, lo and hi must be non-empty and the same length".to_string());
    }
    let inside = (0..y.len()).filter(|&i| y[i] >= lo[i] && y[i] <= hi[i]).count();
    Ok(inside as f64 / y.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shared fixtures: identical to the Python and Julia test suites.
    const PAPER_SERIES: [f64; 3] = [9.0, 14.0, 17.0];
    const SCALED: [f64; 3] = [-1.875, 0.0, 1.125];
    const BURST: [f64; 3] = [-0.10833333333333331, 0.016666666666666663, 0.09166666666666667];
    const SESSION_COUNTS: [f64; 5] = [4.0, 3.0, 1.0, 3.0, 1.0];
    const EMA_123_07: f64 = 2.61;
    const LSTM_H: [f64; 2] = [-0.007358975720281007, -0.3247841816214326];
    const ATT_H: [f64; 4] = [0.5, -0.2, 0.1, 0.8];
    const ATT_E: [f64; 3] = [0.2, 1.0, 0.5];
    const ATT_Z: [f64; 2] = [0.13473023562592665, -0.13723523527276552];
    const ATT_ALPHA: [f64; 3] = [0.3514830614166104, 0.28412005829708203, 0.3643968802863075];
    const TOTAL_QL: f64 = 1.2;
    const WQL_09: f64 = 0.16;
    const GATED: [f64; 2] = [2.5, 4.424234314520019];
    const BURST_FC: [f64; 2] = [0.0, 7.6159415595576485];

    fn close(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
    }

    fn sessions() -> Vec<Vec<usize>> {
        vec![vec![0, 1, 2], vec![0, 1], vec![0, 3, 0, 1], vec![3, 3, 4]]
    }

    #[test]
    fn scaling_matches_paper_example() {
        let r = robust_median_scale(&PAPER_SERIES).unwrap();
        assert_eq!(r.median, 14.0);
        assert!((r.mad - 8.0 / 3.0).abs() < 1e-12);
        assert!(close(&r.scaled, &SCALED, 1e-12));
        let c = robust_median_scale(&[3.0, 3.0, 3.0]).unwrap();
        assert_eq!(c.mad, 0.0);
        assert_eq!(c.scaled, vec![0.0; 3]);
        assert!(robust_median_scale(&[]).is_err());
    }

    #[test]
    fn burstiness_matches_paper_example() {
        let b = burstiness(&PAPER_SERIES).unwrap();
        assert!(close(&b, &BURST, 1e-12));
        assert!(b.iter().sum::<f64>().abs() < 1e-12);
        assert_eq!(burstiness(&[0.0, 0.0, 0.0]).unwrap(), vec![0.0; 3]);
        assert!(burstiness(&[1.0, -1.0]).is_err());
    }

    #[test]
    fn session_graph_and_neighbors() {
        let (counts, w) = session_graph(&sessions(), 5).unwrap();
        assert_eq!(counts, SESSION_COUNTS.to_vec());
        assert_eq!(w[0], vec![0.0, 3.0, 0.0, 1.0, 0.0]);
        assert_eq!(w[1], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(w[3], vec![1.0, 0.0, 0.0, 0.0, 1.0]);
        assert_eq!(w[2], vec![0.0; 5]);
        assert!(session_graph(&[vec![0, 9]], 5).is_err());
        assert_eq!(upstream_neighbors(&w, 1), vec![0]);
        assert_eq!(upstream_neighbors(&w, 0), vec![3]);
        assert_eq!(upstream_neighbors(&w, 4), vec![3]);
        assert_eq!(upstream_neighbors(&w, 3), vec![0]);
        let b = vec![
            burstiness(&[9.0, 14.0, 17.0]).unwrap(),
            burstiness(&[2.0, 4.0, 4.0]).unwrap(),
            burstiness(&[1.0, 1.0, 1.0]).unwrap(),
        ];
        let nb = neighbor_burstiness(&b, &[0, 1]).unwrap();
        let want: Vec<f64> = (0..3).map(|k| (b[0][k] + b[1][k]) / 2.0).collect();
        assert!(close(&nb, &want, 1e-15));
        assert_eq!(neighbor_burstiness(&b, &[]).unwrap(), vec![0.0; 3]);
        let a = average_graph(&[vec![vec![1.0, 0.0], vec![0.0, 1.0]], vec![vec![3.0, 0.0], vec![0.0, 3.0]]]).unwrap();
        assert_eq!(a, vec![vec![2.0, 0.0], vec![0.0, 2.0]]);
        let n = normalize_edge_weights(&vec![vec![0.0, 4.0], vec![2.0, 0.0]], &[2.0, 0.0]).unwrap();
        assert_eq!(n, vec![vec![0.0, 0.0], vec![1.0, 0.0]]);
    }

    #[test]
    fn trend_softmax_sigmoid() {
        assert!((ema_last(&[1.0, 2.0, 3.0], 0.7).unwrap() - EMA_123_07).abs() < 1e-12);
        assert!(close(&burst_forecast(&[0.0, 1.0], 10.0), &BURST_FC, 1e-12));
        assert!(ema_last(&[1.0], 0.0).is_err());
        assert_eq!(softmax(&[1000.0, 1000.0]).unwrap(), vec![0.5, 0.5]);
        assert_eq!(sigmoid(0.0), 0.5);
        assert!(sigmoid(-800.0).abs() < 1e-300);
    }

    #[test]
    fn embeddings_match_fixtures() {
        let w = det_matrix(8, 3, 0.0);
        let b = det_matrix(1, 8, 0.3).remove(0);
        let x = vec![vec![1.0], vec![2.0], vec![0.5]];
        let h = lstm_embed(&x, &w, &b).unwrap();
        assert!(close(&h, &LSTM_H, 1e-12));
        assert!(h.iter().all(|v| v.abs() < 1.0));
        let w_bad: Matrix = w.iter().map(|r| r[..2].to_vec()).collect();
        assert!(lstm_embed(&x, &w_bad, &b).is_err());

        let (wq, wk, wv) = (det_matrix(3, 4, 0.1), det_matrix(3, 4, 0.2), det_matrix(2, 4, 0.3));
        let we = det_matrix(1, 3, 0.4).remove(0);
        let hn = det_matrix(3, 4, 0.5);
        let (z, a) = attention_neighbor_embedding(&ATT_H, &hn, &ATT_E, &wq, &wk, &wv, &we).unwrap();
        assert!(close(&a, &ATT_ALPHA, 1e-12));
        assert!(close(&z, &ATT_Z, 1e-12));
        assert!((a.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        // permutation equivariance of the neighbor set
        let perm = [2usize, 0, 1];
        let hp: Matrix = perm.iter().map(|&i| hn[i].clone()).collect();
        let ep: Vec<f64> = perm.iter().map(|&i| ATT_E[i]).collect();
        let (z2, a2) = attention_neighbor_embedding(&ATT_H, &hp, &ep, &wq, &wk, &wv, &we).unwrap();
        assert!(close(&z2, &z, 1e-12));
        let ap: Vec<f64> = perm.iter().map(|&i| a[i]).collect();
        assert!(close(&a2, &ap, 1e-12));
        let (z0, a0) = attention_neighbor_embedding(&ATT_H, &Vec::new(), &[], &wq, &wk, &wv, &we).unwrap();
        assert_eq!(z0, vec![0.0; 2]);
        assert!(a0.is_empty());
    }

    #[test]
    fn forecast_assembly_and_loss() {
        let out = gated_forecast(&[1.0, 2.0], &[0.5, -0.5], &[2.0, 4.0], &[0.0, 1.0]).unwrap();
        assert!(close(&out, &GATED, 1e-12));
        assert!(gated_forecast(&[1.0], &[1.0, 2.0], &[1.0], &[0.0]).is_err());
        assert_eq!(pinball_loss(10.0, 8.0, 0.75).unwrap(), 1.5);
        assert!((pinball_loss(8.0, 10.0, 0.75).unwrap() - 0.5).abs() < 1e-15);
        let preds = vec![vec![1.0, 2.0, 3.0], vec![2.0, 2.0, 2.0]];
        let t = total_quantile_loss(&[1.5, 2.5], &preds, &[0.1, 0.5, 0.9]).unwrap();
        assert!((t - TOTAL_QL).abs() < 1e-12);
        let w = weighted_quantile_loss(&[10.0, 20.0, 0.0], &[8.0, 25.0, 1.0], 0.9).unwrap();
        assert!((w - WQL_09).abs() < 1e-12);
        assert!(pinball_loss(1.0, 1.0, 1.0).is_err());
        assert!(weighted_quantile_loss(&[0.0], &[1.0], 0.5).is_err());
        let c = interval_coverage(&[1.0, 2.0, 3.0, 4.0], &[0.0; 4], &[2.0; 4]).unwrap();
        assert_eq!(c, 0.5);
    }
}
