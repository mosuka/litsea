use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use reqwest::Client;

use crate::util::ModelScheme;

/// 多クラスAveraged Perceptron分類器。
///
/// スパースな二値特徴に対する多クラス分類を行う。
/// 学習時に重みの累積平均を保持し、過学習を抑制する。
#[derive(Debug)]
pub struct AveragedPerceptron {
    /// クラスごとの重みベクトル: weights\[class\]\[feature\] = weight
    weights: HashMap<String, HashMap<String, f64>>,
    /// 平均化用の累積重み
    accumulated: HashMap<String, HashMap<String, f64>>,
    /// 各重みの最終更新タイムスタンプ
    timestamps: HashMap<String, HashMap<String, usize>>,
    /// 現在のステップ数（全インスタンスの累計）
    step: usize,
    /// 既知のクラス一覧
    classes: Vec<String>,
    /// 学習インスタンス: (特徴量セット, 正解ラベル)
    instances: Vec<(Vec<String>, String)>,
}

/// 評価指標（多クラス分類のマクロ平均）。
#[derive(Debug, Clone)]
pub struct Metrics {
    /// 正解率 (%)
    pub accuracy: f64,
    /// マクロ平均適合率 (%)
    pub macro_precision: f64,
    /// マクロ平均再現率 (%)
    pub macro_recall: f64,
    /// インスタンス数
    pub num_instances: usize,
    /// クラスごとの正解数
    pub correct_per_class: HashMap<String, usize>,
    /// クラスごとの予測数
    pub predicted_per_class: HashMap<String, usize>,
    /// クラスごとの正解ラベル数
    pub gold_per_class: HashMap<String, usize>,
}

impl Default for AveragedPerceptron {
    fn default() -> Self {
        Self::new()
    }
}

impl AveragedPerceptron {
    /// 新しいAveraged Perceptronインスタンスを作成する。
    pub fn new() -> Self {
        AveragedPerceptron {
            weights: HashMap::new(),
            accumulated: HashMap::new(),
            timestamps: HashMap::new(),
            step: 0,
            classes: Vec::new(),
            instances: Vec::new(),
        }
    }

    /// インスタンスを追加する。
    ///
    /// # Arguments
    /// * `features` - 特徴量の集合
    /// * `label` - 正解ラベル
    pub fn add_instance(&mut self, features: HashSet<String>, label: String) {
        // 新しいクラスを登録
        if !self.classes.contains(&label) {
            self.classes.push(label.clone());
            self.classes.sort();
        }
        let feats: Vec<String> = features.into_iter().collect();
        self.instances.push((feats, label));
    }

    /// 特徴量セットからラベルを予測する。
    ///
    /// 各クラスのスコアを計算し、最大スコアのクラスを返す。
    /// クラスが未登録の場合は空文字列を返す。
    #[must_use]
    pub fn predict(&self, features: &HashSet<String>) -> String {
        if self.classes.is_empty() {
            return String::new();
        }
        let mut best_class = &self.classes[0];
        let mut best_score = f64::NEG_INFINITY;

        for class in &self.classes {
            let score = self.score(class, features);
            if score > best_score {
                best_score = score;
                best_class = class;
            }
        }
        best_class.clone()
    }

    /// クラスに対するスコアを計算する。
    fn score(&self, class: &str, features: &HashSet<String>) -> f64 {
        let Some(class_weights) = self.weights.get(class) else {
            return 0.0;
        };
        let mut score = 0.0;
        for feat in features {
            if let Some(&w) = class_weights.get(feat) {
                score += w;
            }
        }
        score
    }

    /// 重みを更新する（1インスタンス分）。
    ///
    /// 予測が正解と異なる場合:
    /// - 正解クラスの重みを +1
    /// - 予測クラスの重みを -1
    fn update(&mut self, truth: &str, guess: &str, features: &[String]) {
        if truth == guess {
            return;
        }
        for feat in features {
            // 正解クラスの累積を更新してから重みを+1
            self.update_accumulated(truth, feat);
            *self
                .weights
                .entry(truth.to_string())
                .or_default()
                .entry(feat.clone())
                .or_insert(0.0) += 1.0;

            // 予測クラスの累積を更新してから重みを-1
            self.update_accumulated(guess, feat);
            *self
                .weights
                .entry(guess.to_string())
                .or_default()
                .entry(feat.clone())
                .or_insert(0.0) -= 1.0;
        }
    }

    /// 累積重みを最新のステップまで追いつかせる。
    fn update_accumulated(&mut self, class: &str, feature: &str) {
        let last_step = self
            .timestamps
            .entry(class.to_string())
            .or_default()
            .entry(feature.to_string())
            .or_insert(0);
        let elapsed = self.step - *last_step;
        if elapsed > 0 {
            let current_weight =
                self.weights.get(class).and_then(|cw| cw.get(feature)).copied().unwrap_or(0.0);
            *self
                .accumulated
                .entry(class.to_string())
                .or_default()
                .entry(feature.to_string())
                .or_insert(0.0) += current_weight * elapsed as f64;
        }
        *last_step = self.step;
    }

    /// 平均化した重みを最終モデルに反映する。
    fn average_weights(&mut self) {
        let classes: Vec<String> = self.classes.clone();
        for class in &classes {
            let features: Vec<String> = self
                .weights
                .get(class)
                .map(|cw| cw.keys().cloned().collect())
                .unwrap_or_default();
            for feat in features {
                self.update_accumulated(class, &feat);
                let acc = self
                    .accumulated
                    .get(class)
                    .and_then(|cw| cw.get(&feat))
                    .copied()
                    .unwrap_or(0.0);
                let avg = acc / self.step.max(1) as f64;
                self.weights.entry(class.to_string()).or_default().insert(feat, avg);
            }
        }
    }

    /// モデルを学習する。
    ///
    /// # Arguments
    /// * `num_epochs` - エポック数
    /// * `running` - 学習を中断するためのフラグ
    pub fn train(&mut self, num_epochs: usize, running: Arc<AtomicBool>) {
        if self.instances.is_empty() {
            return;
        }

        // インスタンスのコピーを作成（学習中にself.instancesを参照するため）
        let instances: Vec<(Vec<String>, String)> = self.instances.clone();

        for _epoch in 0..num_epochs {
            if !running.load(Ordering::SeqCst) {
                break;
            }

            for (features, truth) in &instances {
                if !running.load(Ordering::SeqCst) {
                    break;
                }

                let feature_set: HashSet<String> = features.iter().cloned().collect();
                let guess = self.predict(&feature_set);
                self.update(truth, &guess, features);
                self.step += 1;
            }
        }

        // 平均化した重みを最終モデルに反映
        self.average_weights();
    }

    /// モデルをJSON形式でファイルに保存する。
    ///
    /// フォーマット:
    /// ```text
    /// クラス数
    /// クラス名1
    /// クラス名2
    /// ...
    /// 特徴名\tクラス名\t重み
    /// 特徴名\tクラス名\t重み
    /// ...
    /// ```
    pub fn save_model(&self, path: &Path) -> std::io::Result<()> {
        if self.classes.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Cannot save an empty model",
            ));
        }

        let mut file = File::create(path)?;

        // ヘッダー: クラス数とクラス名
        writeln!(file, "{}", self.classes.len())?;
        for class in &self.classes {
            writeln!(file, "{}", class)?;
        }

        // 重み: 非ゼロの重みのみ保存
        for (class, class_weights) in &self.weights {
            for (feat, &w) in class_weights {
                if w != 0.0 {
                    writeln!(file, "{}\t{}\t{}", feat, class, w)?;
                }
            }
        }

        Ok(())
    }

    /// モデルをURIから読み込む。
    pub async fn load_model(&mut self, uri: &str) -> std::io::Result<()> {
        if uri.contains("://") {
            let parts: Vec<&str> = uri.splitn(2, "://").collect();
            if parts.len() != 2 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("Invalid URI: {}", uri),
                ));
            }
            let scheme = ModelScheme::from_str(parts[0]).map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, e.to_string())
            })?;
            match scheme {
                ModelScheme::Http | ModelScheme::Https => {
                    self.load_model_from_url(uri).await.map_err(|e| {
                        std::io::Error::other(format!("Failed to load model from URL: {}", e))
                    })
                }
                ModelScheme::File => {
                    #[cfg(target_arch = "wasm32")]
                    {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::Unsupported,
                            "file:// scheme is not supported in WASM environment.",
                        ));
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        let path = Path::new(parts[1]);
                        self.load_model_from_file(path)
                    }
                }
            }
        } else {
            #[cfg(target_arch = "wasm32")]
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "Local file paths are not supported in WASM environment.",
                ));
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                let path = Path::new(uri);
                self.load_model_from_file(path)
            }
        }
    }

    /// URLからモデルを読み込む。
    async fn load_model_from_url(&mut self, url: &str) -> std::io::Result<()> {
        let client = Client::builder()
            .user_agent(format!("Litsea/{}", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|e| std::io::Error::other(format!("Failed to create HTTP client: {}", e)))?;

        let resp = client
            .get(url)
            .send()
            .await
            .map_err(|e| std::io::Error::other(format!("Failed to download model: {}", e)))?;

        if !resp.status().is_success() {
            return Err(std::io::Error::other(format!(
                "Failed to download model: HTTP {}",
                resp.status()
            )));
        }

        let content = resp
            .bytes()
            .await
            .map_err(|e| std::io::Error::other(format!("Failed to read model content: {}", e)))?;

        let reader = BufReader::new(content.as_ref());
        self.parse_model_content(reader)
    }

    /// モデルの内容をパースする。
    pub(crate) fn parse_model_content<R: BufRead>(&mut self, reader: R) -> std::io::Result<()> {
        let mut lines = reader.lines();

        // クラス数を読む
        let num_classes: usize = lines
            .next()
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "Empty model file")
            })?
            .map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, format!("Read error: {}", e))
            })?
            .trim()
            .parse()
            .map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Invalid class count: {}", e),
                )
            })?;

        // クラス名を読む
        self.classes.clear();
        for _ in 0..num_classes {
            let class = lines
                .next()
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "Unexpected end of model file while reading classes",
                    )
                })?
                .map_err(|e| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Read error: {}", e),
                    )
                })?;
            self.classes.push(class.trim().to_string());
        }

        // 重みを読む
        self.weights.clear();
        for line in lines {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() != 3 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Invalid weight line: '{}'", line),
                ));
            }
            let feat = parts[0];
            let class = parts[1];
            let weight: f64 = parts[2].parse().map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Invalid weight value: {}", e),
                )
            })?;
            self.weights
                .entry(class.to_string())
                .or_default()
                .insert(feat.to_string(), weight);
        }

        Ok(())
    }

    /// ファイルからモデルを読み込む。
    #[cfg(not(target_arch = "wasm32"))]
    fn load_model_from_file(&mut self, path: &Path) -> std::io::Result<()> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        self.parse_model_content(reader)
    }

    #[cfg(target_arch = "wasm32")]
    fn load_model_from_file(&mut self, _path: &Path) -> std::io::Result<()> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "File system access is not supported in WASM environment",
        ))
    }

    /// 学習データに対する評価指標を計算する。
    #[must_use]
    pub fn get_metrics(&self) -> Metrics {
        let mut correct_per_class: HashMap<String, usize> = HashMap::new();
        let mut predicted_per_class: HashMap<String, usize> = HashMap::new();
        let mut gold_per_class: HashMap<String, usize> = HashMap::new();
        let mut total_correct = 0usize;

        for (features, truth) in &self.instances {
            let feature_set: HashSet<String> = features.iter().cloned().collect();
            let guess = self.predict(&feature_set);

            *gold_per_class.entry(truth.clone()).or_insert(0) += 1;
            *predicted_per_class.entry(guess.clone()).or_insert(0) += 1;

            if guess == *truth {
                total_correct += 1;
                *correct_per_class.entry(truth.clone()).or_insert(0) += 1;
            }
        }

        let num_instances = self.instances.len();
        let accuracy = total_correct as f64 / num_instances.max(1) as f64 * 100.0;

        // マクロ平均適合率・再現率
        let mut sum_precision = 0.0;
        let mut sum_recall = 0.0;
        let num_classes = self.classes.len().max(1);

        for class in &self.classes {
            let correct = correct_per_class.get(class).copied().unwrap_or(0) as f64;
            let predicted = predicted_per_class.get(class).copied().unwrap_or(0) as f64;
            let gold = gold_per_class.get(class).copied().unwrap_or(0) as f64;

            if predicted > 0.0 {
                sum_precision += correct / predicted;
            }
            if gold > 0.0 {
                sum_recall += correct / gold;
            }
        }

        Metrics {
            accuracy,
            macro_precision: sum_precision / num_classes as f64 * 100.0,
            macro_recall: sum_recall / num_classes as f64 * 100.0,
            num_instances,
            correct_per_class,
            predicted_per_class,
            gold_per_class,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    use tempfile::NamedTempFile;

    #[test]
    fn test_new() {
        let p = AveragedPerceptron::new();
        assert!(p.classes.is_empty());
        assert!(p.weights.is_empty());
        assert_eq!(p.step, 0);
    }

    #[test]
    fn test_add_instance() {
        let mut p = AveragedPerceptron::new();
        let mut feats = HashSet::new();
        feats.insert("f1".to_string());
        p.add_instance(feats, "A".to_string());
        assert_eq!(p.instances.len(), 1);
        assert_eq!(p.classes, vec!["A"]);
    }

    #[test]
    fn test_predict_empty() {
        let p = AveragedPerceptron::new();
        let feats = HashSet::new();
        assert_eq!(p.predict(&feats), "");
    }

    #[test]
    fn test_train_simple() {
        let mut p = AveragedPerceptron::new();

        // クラスAの特徴: f1, f2
        let mut feats_a = HashSet::new();
        feats_a.insert("f1".to_string());
        feats_a.insert("f2".to_string());
        p.add_instance(feats_a.clone(), "A".to_string());

        // クラスBの特徴: f3, f4
        let mut feats_b = HashSet::new();
        feats_b.insert("f3".to_string());
        feats_b.insert("f4".to_string());
        p.add_instance(feats_b.clone(), "B".to_string());

        let running = Arc::new(AtomicBool::new(true));
        p.train(10, running);

        // 学習後、正しく分類できるか
        assert_eq!(p.predict(&feats_a), "A");
        assert_eq!(p.predict(&feats_b), "B");
    }

    #[test]
    fn test_train_immediate_stop() {
        let mut p = AveragedPerceptron::new();
        let mut feats = HashSet::new();
        feats.insert("f1".to_string());
        p.add_instance(feats, "A".to_string());

        let running = Arc::new(AtomicBool::new(false));
        p.train(10, running);

        // 学習が即座に停止しても panic しない
        assert_eq!(p.step, 0);
    }

    #[test]
    fn test_train_multiclass() {
        let mut p = AveragedPerceptron::new();

        // 3クラスの学習データ
        for _ in 0..5 {
            let mut fa = HashSet::new();
            fa.insert("feat_a".to_string());
            fa.insert("shared".to_string());
            p.add_instance(fa, "CLASS_A".to_string());

            let mut fb = HashSet::new();
            fb.insert("feat_b".to_string());
            fb.insert("shared".to_string());
            p.add_instance(fb, "CLASS_B".to_string());

            let mut fc = HashSet::new();
            fc.insert("feat_c".to_string());
            fc.insert("shared".to_string());
            p.add_instance(fc, "CLASS_C".to_string());
        }

        let running = Arc::new(AtomicBool::new(true));
        p.train(20, running);

        // 固有特徴で正しく分類できる
        let mut test_a = HashSet::new();
        test_a.insert("feat_a".to_string());
        test_a.insert("shared".to_string());
        assert_eq!(p.predict(&test_a), "CLASS_A");

        let mut test_b = HashSet::new();
        test_b.insert("feat_b".to_string());
        test_b.insert("shared".to_string());
        assert_eq!(p.predict(&test_b), "CLASS_B");
    }

    #[tokio::test]
    async fn test_save_and_load_model() -> std::io::Result<()> {
        let mut p = AveragedPerceptron::new();
        let mut feats_a = HashSet::new();
        feats_a.insert("f1".to_string());
        p.add_instance(feats_a.clone(), "A".to_string());

        let mut feats_b = HashSet::new();
        feats_b.insert("f2".to_string());
        p.add_instance(feats_b.clone(), "B".to_string());

        let running = Arc::new(AtomicBool::new(true));
        p.train(5, running);

        // 保存
        let temp = NamedTempFile::new()?;
        p.save_model(temp.path())?;

        // 読み込み
        let mut p2 = AveragedPerceptron::new();
        p2.load_model(temp.path().to_str().unwrap()).await?;

        // 同じクラスが復元される
        assert_eq!(p2.classes.len(), p.classes.len());

        // 同じ予測結果が得られる
        assert_eq!(p.predict(&feats_a), p2.predict(&feats_a));
        assert_eq!(p.predict(&feats_b), p2.predict(&feats_b));

        Ok(())
    }

    #[test]
    fn test_save_model_empty() {
        let p = AveragedPerceptron::new();
        let temp = NamedTempFile::new().unwrap();
        let result = p.save_model(temp.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_get_metrics() {
        let mut p = AveragedPerceptron::new();

        let mut feats_a = HashSet::new();
        feats_a.insert("f1".to_string());
        p.add_instance(feats_a, "A".to_string());

        let mut feats_b = HashSet::new();
        feats_b.insert("f2".to_string());
        p.add_instance(feats_b, "B".to_string());

        let running = Arc::new(AtomicBool::new(true));
        p.train(10, running);

        let metrics = p.get_metrics();
        assert_eq!(metrics.num_instances, 2);
        assert!(metrics.accuracy > 0.0);
    }

    #[test]
    fn test_get_metrics_empty() {
        let p = AveragedPerceptron::new();
        let metrics = p.get_metrics();
        assert_eq!(metrics.num_instances, 0);
        assert!((metrics.accuracy - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_parse_model_content_invalid() {
        let mut p = AveragedPerceptron::new();
        // 不正なクラス数
        let result = p.parse_model_content(BufReader::new("not_a_number".as_bytes()));
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_model_content_empty() {
        let mut p = AveragedPerceptron::new();
        let result = p.parse_model_content(BufReader::new("".as_bytes()));
        assert!(result.is_err());
    }
}
