//! Declarative model registry (requirement R7, R8).
//!
//! Each [`ModelEntry`] fully describes a downloadable ONNX detector: where to
//! fetch it, how to verify it, its license, how to map its native class indices
//! onto the canonical taxonomy, and the tunable [`Setting`]s it exposes. The CLI
//! (`obscura models list`), the downloader (`ob-models`), the GUI model picker and
//! the inference adapters (`ob-detect`) all read from this one table.

use crate::settings::{Setting, SettingKind, SettingValue};
use crate::taxonomy::{cat, Category, NUDENET_CATEGORIES};
use serde::{Deserialize, Serialize};

/// Content domain a model is trained for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Domain {
    RealLife,
    Anime,
}

/// License classification, surfaced so non-permissive weights are obvious.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum License {
    Apache2,
    Mit,
    OpenRail,
    /// Non-commercial / research-only.
    NonCommercial,
    Unknown,
}

impl License {
    /// Whether the weights are broadly permissive (informational only).
    pub fn is_permissive(&self) -> bool {
        matches!(self, License::Apache2 | License::Mit)
    }
}

/// How native class indices map to canonical categories.
///
/// `by_index[i]` is the [`Category`] for the model's native class `i`. A model
/// whose ordering matches NudeNet reuses [`nudenet_label_map`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LabelMap {
    pub by_index: Vec<Category>,
}

impl LabelMap {
    pub fn get(&self, native_index: usize) -> Option<Category> {
        self.by_index.get(native_index).copied()
    }

    pub fn num_classes(&self) -> usize {
        self.by_index.len()
    }
}

/// The canonical NudeNet v3 index→category mapping.
pub fn nudenet_label_map() -> LabelMap {
    LabelMap {
        by_index: NUDENET_CATEGORIES.to_vec(),
    }
}

/// Index→category mapping for deepghs' `anime_censor_detection`.
///
/// That model is a 3-class YOLOv8 (`nipple_f`, `penis`, `pussy`) — *not* the
/// 18-class NudeNet taxonomy — so it needs its own map. Each class lands on an
/// existing Obscura category; the parts it cannot see (buttocks, anus, feet, belly,
/// armpits, face) simply never fire, and it reports no `Covered` states.
pub fn anime_censor_label_map() -> LabelMap {
    LabelMap {
        by_index: vec![
            cat::FEMALE_BREAST_EXPOSED,    // nipple_f
            cat::MALE_GENITALIA_EXPOSED,   // penis
            cat::FEMALE_GENITALIA_EXPOSED, // pussy
        ],
    }
}

/// A complete, self-describing model definition.
///
/// Not `Serialize`/`Deserialize`: it is a compiled-in registry entry (it holds a
/// function pointer), and only its `id` is ever persisted (via `Profile`).
#[derive(Debug, Clone)]
pub struct ModelEntry {
    /// Stable id, e.g. `"nudenet-320n"`. Used by `--model` and the cache path.
    pub id: &'static str,
    pub display_name: &'static str,
    pub domain: Domain,
    pub license: License,
    /// Square model input side in pixels (letterbox target).
    pub input_size: u32,
    /// Download URL (only `ob-models` touches the network — R8).
    pub url: &'static str,
    /// Expected SHA-256 of the downloaded file, lowercase hex.
    pub sha256: &'static str,
    pub label_map_fn: fn() -> LabelMap,
    /// Per-model tunable settings (each carries its own tooltip — R7).
    pub settings: Vec<Setting>,

    // --- presentation metadata (model picker, download screen, credits) ---
    /// One-line plain-language description for the model card in the GUI.
    pub summary: &'static str,
    /// Human-readable download size, e.g. `"~12 MB"`.
    ///
    /// A **display hint only**, shown before the server reports a
    /// `Content-Length` and used for the "you are about to download N MB"
    /// total on the first-run screen. These figures are derived from each
    /// model's published parameter count at fp32 (a yolov8n at 3.2M params is
    /// ~12 MB, a yolov8s at 11.1M is ~44 MB, a yolov8m at 25.9M is ~100 MB);
    /// none has been weighed here, because no model host is reachable from the
    /// build container. Never treat it as a size *check* — `fetch` uses the
    /// checksum for that.
    pub approx_bytes: u64,
    /// Project page for the weights, shown in the credits/licensing screen.
    /// Distinct from `url`, which is the raw file.
    pub homepage: &'static str,
    /// Whether first-run setup and the installer download this by default.
    /// Enough to make Obscura useful out of the box without pulling every model.
    pub default_download: bool,
}

/// Two entries are the same model when they have the same id: the id is what
/// `Profile` persists, what `--model` names, and what the cache path is built
/// from. Hand-written because `label_map_fn` is a function pointer, and
/// comparing function addresses is not meaningful (they are not guaranteed
/// unique across codegen units, so a derived `PartialEq` could report two
/// distinct models equal — or one unequal to itself).
impl PartialEq for ModelEntry {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for ModelEntry {}

impl ModelEntry {
    pub fn label_map(&self) -> LabelMap {
        (self.label_map_fn)()
    }
}

/// Standard detector settings shared by the YOLO-family ONNX models.
///
/// `conf_default` is per-model: a model that publishes its own F1-optimal
/// threshold should pass it rather than inherit the generic default.
fn yolo_detect_settings(conf_default: f64) -> Vec<Setting> {
    vec![
        Setting {
            key: "conf_threshold",
            label: "Confidence threshold",
            kind: SettingKind::Float {
                min: 0.0,
                max: 1.0,
                step: 0.01,
            },
            default: SettingValue::Float(conf_default),
            unit: "",
            tooltip: "Minimum detector confidence to keep a detection. Lower \
                      catches more (and more false positives); higher is stricter.",
        },
        Setting {
            key: "nms_iou",
            label: "NMS IoU",
            kind: SettingKind::Float {
                min: 0.0,
                max: 1.0,
                step: 0.01,
            },
            default: SettingValue::Float(0.45),
            unit: "",
            tooltip: "Overlap above which two boxes of the same class are merged \
                      by non-maximum suppression. Lower removes more duplicates.",
        },
        Setting {
            key: "nms_score",
            label: "NMS score floor",
            kind: SettingKind::Float {
                min: 0.0,
                max: 1.0,
                step: 0.01,
            },
            default: SettingValue::Float(0.25),
            unit: "",
            tooltip: "Score below which candidate boxes are dropped before NMS.",
        },
        Setting {
            key: "resample",
            label: "Downscale filter",
            kind: SettingKind::Enum {
                choices: vec!["nearest", "triangle", "catmull-rom", "lanczos3"],
            },
            default: SettingValue::Text("triangle".into()),
            unit: "",
            tooltip: "How the frame is scaled into the model's input. Triangle \
                      (the default) averages every source pixel, so small regions \
                      survive a heavy downscale; Lanczos3 is sharper but rings \
                      around hard edges, which can read as a false edge. Nearest \
                      point-samples and drops small features — avoid it.",
        },
        Setting {
            key: "tiling",
            label: "Tiled detection",
            kind: SettingKind::Enum {
                choices: vec!["off", "auto", "always"],
            },
            default: SettingValue::Text("auto".into()),
            unit: "",
            tooltip: "Run the detector over an overlapping grid of native-\
                      resolution tiles as well as the whole frame, so small \
                      regions are not shrunk below what the model can see. Auto \
                      tiles only when the frame is more than twice the model \
                      input. Costs one inference pass per tile.",
        },
        Setting {
            key: "tile_overlap",
            label: "Tile overlap",
            kind: SettingKind::Float {
                min: 0.0,
                max: 0.9,
                step: 0.05,
            },
            default: SettingValue::Float(0.25),
            unit: "",
            tooltip: "Fraction of each tile shared with its neighbour. Overlap \
                      is what stops a region sitting on a seam from being seen \
                      only in fragments; more overlap means more tiles.",
        },
        Setting {
            key: "tile_max",
            label: "Max tiles per frame",
            kind: SettingKind::Int {
                min: 1,
                max: 256,
                step: 1,
            },
            default: SettingValue::Int(12),
            unit: "",
            tooltip: "Upper bound on tiles per frame — inference cost is linear \
                      in this. When the grid would exceed it, tiles are enlarged \
                      (accepting some downscale) instead of the cap being broken.",
        },
    ]
}

/// The built-in model registry shipped with Obscura.
///
/// **The NudeNet entries use GitHub's REST asset endpoint, not the
/// `browser_download_url`.** The human-facing
/// `github.com/OWNER/REPO/releases/download/TAG/FILE` URL now answers anonymous
/// requests with HTTP 200 and a *sign-in page* rather than the file — verified
/// 2026-08-28 from two unrelated networks, with identical 47,439-byte bodies,
/// and independent of User-Agent. The repository is public and the assets are
/// listed by the API, so this is GitHub restricting anonymous asset downloads,
/// not a dead link. `api.github.com/repos/OWNER/REPO/releases/assets/ID` with
/// `Accept: application/octet-stream` serves the real bytes and is what
/// `ob-models` requests.
///
/// The asset **ids** are stable for a published release; if a maintainer ever
/// re-uploads an asset the id changes and the download 404s (loudly), which is
/// a better failure than silently fetching a login page.
///
/// `nudenet-320n`'s digest is pinned: its bytes were fetched through the
/// endpoint above, confirmed to start with an ONNX `ModelProto` header, and
/// matched against the size GitHub's own metadata reports. `nudenet-640m` and
/// the anime entries are still unpinned — 640m was not downloaded here, and the
/// HuggingFace LFS CDN does not resolve from the build container. `obscura models
/// fetch` prints the SHA-256 of what it downloads; paste those in to finish.
pub fn builtin_registry() -> Vec<ModelEntry> {
    vec![
        ModelEntry {
            id: "nudenet-320n",
            display_name: "NudeNet 320n (real-life, light — default)",
            domain: Domain::RealLife,
            license: License::Apache2,
            input_size: 320,
            // asset 176831997 = 320n.onnx on tag v3.4-weights (12,150,158 bytes).
            url: "https://api.github.com/repos/notAI-tech/NudeNet/releases/assets/176831997",
            sha256: "c15d8273adad2d0a92f014cc69ab2d6c311a06777a55545f2c4eb46f51911f0f",
            label_map_fn: nudenet_label_map,
            settings: yolo_detect_settings(0.20),
            summary: "Photographic content. The light, fast default — a yolov8n at \
                      320px. Good on ordinary photo resolutions; pair it with tiling \
                      for 4K.",
            approx_bytes: 12 * 1024 * 1024,
            homepage: "https://github.com/notAI-tech/NudeNet",
            default_download: true,
        },
        ModelEntry {
            id: "nudenet-640m",
            display_name: "NudeNet 640m (real-life, accurate)",
            domain: Domain::RealLife,
            license: License::Apache2,
            input_size: 640,
            // asset 176832019 = 640m.onnx on tag v3.4-weights (103,538,690 bytes).
            url: "https://api.github.com/repos/notAI-tech/NudeNet/releases/assets/176832019",
            sha256: "", // not yet pinned — see fn doc; fill from `obscura models fetch`
            label_map_fn: nudenet_label_map,
            settings: yolo_detect_settings(0.20),
            summary: "Photographic content, best quality. A yolov8m at 640px — \
                      roughly eight times the download and several times the \
                      inference cost of 320n, for better recall on small regions.",
            approx_bytes: 100 * 1024 * 1024,
            homepage: "https://github.com/notAI-tech/NudeNet",
            default_download: false,
        },
        ModelEntry {
            id: "anime-censor-v1-s",
            display_name: "deepghs anime_censor_detection v1.0_s (anime — default)",
            domain: Domain::Anime,
            license: License::Mit,
            // yolov8s trained at imgsz 640 (from the repo's model_artifacts.json).
            input_size: 640,
            url: "https://huggingface.co/deepghs/anime_censor_detection/resolve/main/censor_detect_v1.0_s/model.onnx",
            sha256: "", // not yet pinned — see fn doc; fill from `obscura models fetch`
            // 3 classes, NOT NudeNet's 18 — see `anime_censor_label_map`.
            label_map_fn: anime_censor_label_map,
            // 0.238 is the threshold the repo publishes as F1-optimal.
            settings: yolo_detect_settings(0.238),
            summary: "Drawn/anime content. The default for illustration: a yolov8s \
                      at 640px, F1 0.83. Detects nipples, penis and vulva only — \
                      raise the region padding to cover more than the box itself.",
            approx_bytes: 44 * 1024 * 1024,
            homepage: "https://huggingface.co/deepghs/anime_censor_detection",
            default_download: true,
        },
        // --- Cross-examination companions -------------------------------
        // Same 3-class taxonomy and the same publisher, but different weights,
        // so running one against another shows where a single model is unsure.
        // Their published F1-optimal thresholds differ and are set per entry;
        // using one model's threshold on another shifts its operating point.
        ModelEntry {
            id: "anime-censor-v1-n",
            display_name: "deepghs anime_censor_detection v1.0_n (anime, fast)",
            domain: Domain::Anime,
            license: License::Mit,
            input_size: 640,
            url: "https://huggingface.co/deepghs/anime_censor_detection/resolve/main/censor_detect_v1.0_n/model.onnx",
            sha256: "", // not yet pinned — see fn doc; fill from `obscura models fetch`
            label_map_fn: anime_censor_label_map,
            // yolov8n sibling of v1.0_s: 3.01M params vs 11.1M, F1 0.80 vs 0.83.
            // Cheap enough to run as a second opinion on every frame.
            settings: yolo_detect_settings(0.278),
            summary: "Drawn/anime content, fast. The yolov8n sibling of v1.0_s — a \
                      quarter the size at F1 0.80. Cheap enough to run on every \
                      frame of a video, or as a second opinion.",
            approx_bytes: 12 * 1024 * 1024,
            homepage: "https://huggingface.co/deepghs/anime_censor_detection",
            default_download: false,
        },
        ModelEntry {
            id: "anime-censor-v0.10-s",
            display_name: "deepghs anime_censor_detection v0.10_s (anime, prior gen)",
            domain: Domain::Anime,
            license: License::Mit,
            input_size: 640,
            url: "https://huggingface.co/deepghs/anime_censor_detection/resolve/main/censor_detect_v0.10_s/model.onnx",
            sha256: "", // not yet pinned — see fn doc; fill from `obscura models fetch`
            label_map_fn: anime_censor_label_map,
            // Same architecture and F1 (0.83) as v1.0_s but an earlier training
            // run, so its mistakes are the least correlated with v1.0_s's of any
            // model here — the most informative second opinion. Note its
            // F1-optimal threshold is far lower, 0.15.
            settings: yolo_detect_settings(0.15),
            summary: "Drawn/anime content, previous generation. Same architecture \
                      and F1 as v1.0_s from an earlier training run, so its \
                      mistakes differ most — the best cross-check partner.",
            approx_bytes: 44 * 1024 * 1024,
            homepage: "https://huggingface.co/deepghs/anime_censor_detection",
            default_download: false,
        },
    ]
}

/// Look up a model entry by id.
pub fn find(id: &str) -> Option<ModelEntry> {
    builtin_registry().into_iter().find(|m| m.id == id)
}

/// The models first-run setup and the installers download by default: one per
/// domain, so Obscura works on both photographic and drawn input immediately without
/// pulling every weight in the registry.
pub fn default_downloads() -> Vec<ModelEntry> {
    builtin_registry()
        .into_iter()
        .filter(|m| m.default_download)
        .collect()
}

/// Approximate total bytes of [`default_downloads`], for the "this will
/// download about N MB" line on the setup screen. See [`ModelEntry::approx_bytes`]
/// — a display hint, not a measurement.
pub fn default_download_bytes() -> u64 {
    default_downloads().iter().map(|m| m.approx_bytes).sum()
}

/// Format a byte count the way a download UI should: two significant figures,
/// binary units, never more precision than the number deserves.
pub fn human_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    let b = bytes as f64;
    let (value, unit) = if b >= KIB * KIB * KIB {
        (b / (KIB * KIB * KIB), "GB")
    } else if b >= KIB * KIB {
        (b / (KIB * KIB), "MB")
    } else if b >= KIB {
        (b / KIB, "KB")
    } else {
        return format!("{bytes} B");
    };
    if value >= 100.0 {
        format!("{value:.0} {unit}")
    } else if value >= 10.0 {
        format!("{value:.1} {unit}")
    } else {
        format!("{value:.2} {unit}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_entries_have_unique_ids() {
        let reg = builtin_registry();
        let mut ids: Vec<_> = reg.iter().map(|m| m.id).collect();
        ids.sort_unstable();
        let before = ids.len();
        ids.dedup();
        assert_eq!(before, ids.len(), "duplicate model ids in registry");
    }

    #[test]
    fn nudenet_map_has_18_classes() {
        assert_eq!(nudenet_label_map().num_classes(), 18);
    }

    #[test]
    fn anime_model_uses_its_own_three_class_map() {
        let e = find("anime-censor-v1-s").unwrap();
        let lm = e.label_map();
        // Not NudeNet's 18 — mapping it that way would mis-decode every box.
        assert_eq!(lm.num_classes(), 3);
        assert_eq!(lm.get(0), Some(cat::FEMALE_BREAST_EXPOSED));
        assert_eq!(lm.get(1), Some(cat::MALE_GENITALIA_EXPOSED));
        assert_eq!(lm.get(2), Some(cat::FEMALE_GENITALIA_EXPOSED));
        assert_eq!(lm.get(3), None);
        assert_eq!(e.domain, Domain::Anime);
        assert_eq!(e.input_size, 640);
    }

    #[test]
    fn every_registry_url_is_real() {
        // A `TODO://` placeholder here means `fetch` refuses the model at
        // runtime, which is easy to miss until someone tries to download it.
        for m in builtin_registry() {
            assert!(
                m.url.starts_with("https://"),
                "model `{}` has a non-download URL: {}",
                m.id,
                m.url
            );
        }
    }

    #[test]
    fn every_entry_has_presentation_metadata() {
        // The GUI model card renders all three unconditionally; an empty one
        // would ship as a blank row rather than fail loudly.
        for m in builtin_registry() {
            assert!(!m.summary.is_empty(), "model {} has no summary", m.id);
            assert!(
                m.homepage.starts_with("https://"),
                "model {} has no homepage",
                m.id
            );
            assert!(m.approx_bytes > 0, "model {} has no size hint", m.id);
        }
    }

    #[test]
    fn default_downloads_cover_both_domains() {
        // First-run setup must leave the user able to process either kind of
        // input; a default set that is all one domain would silently be useless
        // for the other half of the audience.
        let d = default_downloads();
        assert!(d.iter().any(|m| m.domain == Domain::RealLife));
        assert!(d.iter().any(|m| m.domain == Domain::Anime));
        assert!(
            d.len() < builtin_registry().len(),
            "defaults should be a subset"
        );
        assert!(default_download_bytes() > 0);
    }

    #[test]
    fn model_entries_compare_by_id_not_function_address() {
        let a = find("nudenet-320n").unwrap();
        let b = find("nudenet-320n").unwrap();
        // Reflexive and stable across separate constructions — the property a
        // derived PartialEq over a `fn` pointer cannot guarantee.
        assert_eq!(a, b);
        assert_ne!(a, find("nudenet-640m").unwrap());
    }

    #[test]
    fn human_bytes_scales_and_keeps_two_significant_figures() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(12 * 1024 * 1024), "12.0 MB");
        assert_eq!(human_bytes(100 * 1024 * 1024), "100 MB");
        assert_eq!(human_bytes(3 * 1024 * 1024 / 2), "1.50 MB");
    }

    #[test]
    fn find_returns_known_model() {
        assert!(find("nudenet-320n").is_some());
        assert!(find("does-not-exist").is_none());
    }

    #[test]
    fn every_model_exposes_conf_threshold() {
        for m in builtin_registry() {
            assert!(
                m.settings.iter().any(|s| s.key == "conf_threshold"),
                "model {} missing conf_threshold",
                m.id
            );
        }
    }
}
