//! Easy Data Augmentation (EDA) for text classification (Rust).
//!
//! Mirrors the Python module `aiinaction.ch071_eda` and the Julia module
//! `AIInAction.Ch071Eda`. Implements the four EDA operations of Wei and Zou
//! (2019): synonym replacement, random insertion, random swap, random
//! deletion. Cross-language parity requires bit-identical randomness, so this
//! uses the same Park-Miller 32-bit LCG and the same fixed synonym table as the
//! other two implementations; the shared fixtures in the tests below match the
//! Python/Julia suites.

pub mod ch071_eda {
    //! EDA operations and a deterministic generator.

    const LCG_MOD: u64 = 2_147_483_647; // 2**31 - 1
    const LCG_MULT: u64 = 16_807;

    /// Returns the deterministic, ordered synonym candidates for a word, or an
    /// empty slice if the (lowercased) word is not in the fixed table.
    pub fn synonyms_for(word: &str) -> &'static [&'static str] {
        let lower = word.to_lowercase();
        match lower.as_str() {
            "quick" => &["fast", "rapid", "swift"],
            "fast" => &["quick", "rapid", "speedy"],
            "happy" => &["glad", "joyful", "cheerful"],
            "sad" => &["unhappy", "gloomy", "downcast"],
            "big" => &["large", "huge", "massive"],
            "small" => &["tiny", "little", "compact"],
            "good" => &["great", "fine", "decent"],
            "bad" => &["poor", "awful", "lousy"],
            "smart" => &["clever", "bright", "sharp"],
            "movie" => &["film", "picture", "feature"],
            "car" => &["automobile", "vehicle", "auto"],
            "house" => &["home", "dwelling", "residence"],
            _ => &[],
        }
    }

    /// A 32-bit Park-Miller linear congruential generator.
    pub struct Lcg {
        state: u64,
    }

    impl Lcg {
        /// Creates a generator from a non-negative seed. A reduced seed of zero
        /// is remapped to one (the generator is undefined at zero).
        pub fn new(seed: u64) -> Self {
            let mut state = seed % LCG_MOD;
            if state == 0 {
                state = 1;
            }
            Lcg { state }
        }

        /// Advances the generator and returns the new 31-bit state.
        pub fn next_u32(&mut self) -> u64 {
            self.state = (self.state * LCG_MULT) % LCG_MOD;
            self.state
        }

        /// Returns the next value mapped to the half-open interval `[0, 1)`.
        pub fn next_float(&mut self) -> f64 {
            (self.next_u32() - 1) as f64 / (LCG_MOD - 1) as f64
        }

        /// Returns an integer in `[0, n)`. Panics if `n == 0`.
        pub fn randint(&mut self, n: usize) -> usize {
            assert!(n > 0, "randint bound must be positive, got {n}");
            (self.next_u32() % n as u64) as usize
        }
    }

    /// Splits text into whitespace-delimited tokens. Errors on empty input.
    pub fn tokenize(text: &str) -> Result<Vec<String>, String> {
        let tokens: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();
        if tokens.is_empty() {
            return Err("text must contain at least one token".to_string());
        }
        Ok(tokens)
    }

    fn has_any_synonym(tokens: &[String]) -> bool {
        tokens.iter().any(|t| !synonyms_for(t).is_empty())
    }

    /// Replaces up to `n` words with a synonym from the fixed table.
    pub fn synonym_replacement(
        tokens: &[String],
        n: i64,
        rng: &mut Lcg,
    ) -> Result<Vec<String>, String> {
        if tokens.is_empty() {
            return Err("tokens must be non-empty".to_string());
        }
        if n < 0 {
            return Err(format!("n must be non-negative, got {n}"));
        }
        let mut out: Vec<String> = tokens.to_vec();
        let candidate_idx: Vec<usize> = out
            .iter()
            .enumerate()
            .filter(|(_, t)| !synonyms_for(t).is_empty())
            .map(|(i, _)| i)
            .collect();
        if candidate_idx.is_empty() {
            return Ok(out);
        }
        for _ in 0..n {
            let pos = candidate_idx[rng.randint(candidate_idx.len())];
            let cands = synonyms_for(&out[pos]);
            out[pos] = cands[rng.randint(cands.len())].to_string();
        }
        Ok(out)
    }

    /// Inserts `n` synonyms of random words at random positions.
    pub fn random_insertion(
        tokens: &[String],
        n: i64,
        rng: &mut Lcg,
    ) -> Result<Vec<String>, String> {
        if tokens.is_empty() {
            return Err("tokens must be non-empty".to_string());
        }
        if n < 0 {
            return Err(format!("n must be non-negative, got {n}"));
        }
        let mut out: Vec<String> = tokens.to_vec();
        if !has_any_synonym(&out) {
            return Ok(out);
        }
        for _ in 0..n {
            let mut word_synonyms: &[&str] = &[];
            for _attempt in 0..10 {
                let cand = &out[rng.randint(out.len())];
                word_synonyms = synonyms_for(cand);
                if !word_synonyms.is_empty() {
                    break;
                }
            }
            if word_synonyms.is_empty() {
                continue;
            }
            let new_word = word_synonyms[rng.randint(word_synonyms.len())].to_string();
            let insert_at = rng.randint(out.len() + 1);
            out.insert(insert_at, new_word);
        }
        Ok(out)
    }

    /// Swaps two random tokens `n` times.
    pub fn random_swap(tokens: &[String], n: i64, rng: &mut Lcg) -> Result<Vec<String>, String> {
        if tokens.is_empty() {
            return Err("tokens must be non-empty".to_string());
        }
        if n < 0 {
            return Err(format!("n must be non-negative, got {n}"));
        }
        let mut out: Vec<String> = tokens.to_vec();
        if out.len() < 2 {
            return Ok(out);
        }
        for _ in 0..n {
            let i = rng.randint(out.len());
            let j = rng.randint(out.len());
            out.swap(i, j);
        }
        Ok(out)
    }

    /// Deletes each token independently with probability `p`; keeps one token if
    /// all would be deleted so the output is never empty.
    pub fn random_deletion(tokens: &[String], p: f64, rng: &mut Lcg) -> Result<Vec<String>, String> {
        if tokens.is_empty() {
            return Err("tokens must be non-empty".to_string());
        }
        if !(0.0..=1.0).contains(&p) {
            return Err(format!("p must be in [0, 1], got {p}"));
        }
        let out: Vec<String> = tokens.to_vec();
        if out.len() == 1 {
            return Ok(out);
        }
        let mut kept: Vec<String> = Vec::new();
        for t in &out {
            if rng.next_float() >= p {
                kept.push(t.clone());
            }
        }
        if kept.is_empty() {
            kept.push(out[rng.randint(out.len())].clone());
        }
        Ok(kept)
    }

    /// Configuration for [`eda`], with EDA-paper defaults.
    pub struct EdaConfig {
        pub alpha_sr: f64,
        pub alpha_ri: f64,
        pub alpha_rs: f64,
        pub p_rd: f64,
        pub num_aug: i64,
        pub seed: u64,
    }

    impl Default for EdaConfig {
        fn default() -> Self {
            EdaConfig {
                alpha_sr: 0.1,
                alpha_ri: 0.1,
                alpha_rs: 0.1,
                p_rd: 0.1,
                num_aug: 4,
                seed: 0,
            }
        }
    }

    fn n_for(alpha: f64, n_words: usize) -> i64 {
        let scaled = (alpha * n_words as f64).round() as i64;
        scaled.max(1)
    }

    /// Generates `cfg.num_aug` augmented sentences from `text`.
    pub fn eda(text: &str, cfg: &EdaConfig) -> Result<Vec<String>, String> {
        if cfg.num_aug < 0 {
            return Err(format!("num_aug must be non-negative, got {}", cfg.num_aug));
        }
        for (name, a) in [
            ("alpha_sr", cfg.alpha_sr),
            ("alpha_ri", cfg.alpha_ri),
            ("alpha_rs", cfg.alpha_rs),
        ] {
            if !(0.0..=1.0).contains(&a) {
                return Err(format!("{name} must be in [0, 1], got {a}"));
            }
        }
        if !(0.0..=1.0).contains(&cfg.p_rd) {
            return Err(format!("p_rd must be in [0, 1], got {}", cfg.p_rd));
        }
        let tokens = tokenize(text)?;
        let n_words = tokens.len();
        let mut rng = Lcg::new(cfg.seed);
        let mut out: Vec<String> = Vec::new();
        for i in 0..cfg.num_aug {
            let aug = match i % 4 {
                0 => synonym_replacement(&tokens, n_for(cfg.alpha_sr, n_words), &mut rng)?,
                1 => random_insertion(&tokens, n_for(cfg.alpha_ri, n_words), &mut rng)?,
                2 => random_swap(&tokens, n_for(cfg.alpha_rs, n_words), &mut rng)?,
                _ => random_deletion(&tokens, cfg.p_rd, &mut rng)?,
            };
            out.push(aug.join(" "));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::ch071_eda::*;

    // Shared fixtures: identical to the Python and Julia test suites.
    fn toks() -> Vec<String> {
        ["the", "quick", "movie", "was", "good", "and", "fast"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    #[test]
    fn lcg_u32_stream() {
        let mut rng = Lcg::new(42);
        let got: Vec<u64> = (0..5).map(|_| rng.next_u32()).collect();
        assert_eq!(got, vec![705894, 1126542223, 1579310009, 565444343, 807934826]);
    }

    #[test]
    fn lcg_float_stream() {
        let mut rng = Lcg::new(42);
        let expected = [0.000328707, 0.5245871018, 0.7354235321];
        for &e in &expected {
            assert!((rng.next_float() - e).abs() < 1e-9);
        }
    }

    #[test]
    fn lcg_randint_stream() {
        let mut rng = Lcg::new(7);
        let got: Vec<usize> = (0..6).map(|_| rng.randint(10)).collect();
        assert_eq!(got, vec![9, 3, 6, 5, 9, 7]);
    }

    #[test]
    fn lcg_seed_zero_remaps() {
        assert_ne!(Lcg::new(0).next_u32(), 0);
    }

    #[test]
    fn tokenize_basic() {
        assert_eq!(tokenize("the quick movie was good and fast").unwrap(), toks());
    }

    #[test]
    fn tokenize_empty_errors() {
        assert!(tokenize("   ").is_err());
    }

    #[test]
    fn synonym_replacement_fixture() {
        let mut rng = Lcg::new(1);
        let got = synonym_replacement(&toks(), 2, &mut rng).unwrap();
        assert_eq!(got, vec!["the", "quick", "feature", "was", "good", "and", "rapid"]);
    }

    #[test]
    fn random_insertion_fixture() {
        let mut rng = Lcg::new(2);
        let got = random_insertion(&toks(), 2, &mut rng).unwrap();
        assert_eq!(
            got,
            vec!["fine", "great", "the", "quick", "movie", "was", "good", "and", "fast"]
        );
    }

    #[test]
    fn random_swap_fixture() {
        let mut rng = Lcg::new(3);
        let got = random_swap(&toks(), 2, &mut rng).unwrap();
        assert_eq!(got, vec!["the", "quick", "movie", "good", "was", "and", "fast"]);
    }

    #[test]
    fn random_deletion_fixture() {
        let mut rng = Lcg::new(4);
        let got = random_deletion(&toks(), 0.3, &mut rng).unwrap();
        assert_eq!(got, vec!["quick", "was", "and"]);
    }

    #[test]
    fn synonym_replacement_no_candidates_identity() {
        let t: Vec<String> = ["xx", "yy", "zz"].iter().map(|s| s.to_string()).collect();
        let mut rng = Lcg::new(5);
        assert_eq!(synonym_replacement(&t, 3, &mut rng).unwrap(), t);
    }

    #[test]
    fn random_deletion_never_empty() {
        let mut rng = Lcg::new(8);
        assert_eq!(random_deletion(&toks(), 1.0, &mut rng).unwrap().len(), 1);
    }

    #[test]
    fn eda_fixture() {
        let cfg = EdaConfig { seed: 123, num_aug: 4, ..Default::default() };
        let got = eda("the quick movie was good and fast", &cfg).unwrap();
        assert_eq!(
            got,
            vec![
                "the quick feature was good and fast",
                "the quick movie was good rapid and fast",
                "the quick movie was good and fast",
                "the quick movie was good and fast",
            ]
        );
    }

    #[test]
    fn eda_alpha_fixture() {
        let cfg = EdaConfig {
            alpha_sr: 0.2,
            alpha_ri: 0.2,
            alpha_rs: 0.2,
            p_rd: 0.2,
            num_aug: 6,
            seed: 99,
        };
        let got = eda("a sleek and surprisingly fast car", &cfg).unwrap();
        assert_eq!(
            got,
            vec![
                "a sleek and surprisingly fast auto",
                "auto a sleek and surprisingly fast car",
                "car sleek and surprisingly fast a",
                "a sleek and surprisingly fast car",
                "a sleek and surprisingly fast automobile",
                "a sleek and surprisingly fast car speedy",
            ]
        );
    }

    #[test]
    fn eda_reproducible() {
        let cfg = EdaConfig { seed: 7, num_aug: 5, ..Default::default() };
        assert_eq!(
            eda("the quick movie was good and fast", &cfg).unwrap(),
            eda("the quick movie was good and fast", &cfg).unwrap()
        );
    }

    #[test]
    fn eda_negative_num_aug_errors() {
        let cfg = EdaConfig { num_aug: -1, ..Default::default() };
        assert!(eda("hi there", &cfg).is_err());
    }

    #[test]
    fn eda_bad_alpha_errors() {
        let cfg = EdaConfig { alpha_sr: 1.5, ..Default::default() };
        assert!(eda("hi there", &cfg).is_err());
    }
}
