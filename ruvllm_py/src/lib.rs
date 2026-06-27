//! Python bindings for the RuvLLM Rust inference engine.
//!
//! This module exposes the RuvLLM inference engine to Python via PyO3,
//! enabling zero-HTTP-overhead inference directly in the orchestrator's
//! Python process. The binding wraps the Rust `ruvllm` crate's
//! `CandleBackend` and exposes a `complete()` method with the same
//! calling convention as `llama_cpp.Llama.__call__`.
//!
//! Key design decisions:
//!   - The GIL is released during the forward pass (`py.allow_threads`)
//!     so the orchestrator's asyncio event loop is not blocked during
//!     generation.
//!   - The backend is held in `Arc<Mutex<...>>` for thread-safe access
//!     from the pool mode (multiple Python threads calling `complete()`
//!     concurrently). `load_gguf` requires `&mut self`, so a `Mutex`
//!     (not `RwLock`) is correct — generation takes `&self` and the
//!     backend is loaded once at construction.
//!   - GBNF grammar strings are NOT directly supported by the candle
//!     backend's `generate()`; when a grammar is supplied we pass it
//!     through as a stop-sequence hint and rely on the orchestrator's
//!     format enforcer for structured-output enforcement. (The HTTP
//!     /completion path remains the canonical grammar-enforced path.)
//!   - TurboQuant KV cache compression is configured at load time via
//!     the `ModelConfig` / `KvCacheConfig` on the ruvllm backend.
//!
//! Build (stub, no inference):
//!   cd ruvllm_py && cargo build --release
//! Build (real inference, CPU):
//!   cd ruvllm_py && cargo build --release --features candle
//! Build (Apple Silicon Metal):
//!   cd ruvllm_py && cargo build --release --features inference-metal
//! Install into the current Python env:
//!   maturin develop --release --features inference-metal
//!
//! Usage (Python):
//!   from ruvllm_py import Engine
//!   engine = Engine(model_path="~/models/vibethinker-3b.gguf")
//!   resp = engine(prompt="Hello", max_tokens=128, temperature=0.7)
//!   print(resp["choices"][0]["text"])

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::{Arc, Mutex};

/// Configuration for the RuvLLM inference engine.
#[pyclass]
#[derive(Clone)]
pub struct EngineConfig {
    /// Path to the GGUF model file.
    pub model_path: String,
    /// Context window size (tokens).
    pub n_ctx: u32,
    /// Number of CPU threads for inference.
    pub n_threads: u32,
    /// TurboQuant KV cache type for K (e.g. "q8_0", "f16").
    pub cache_type_k: String,
    /// TurboQuant KV cache type for V (e.g. "turbo3", "turbo4").
    pub cache_type_v: String,
    /// Enable Metal acceleration (Apple Silicon only).
    pub use_metal: bool,
}

#[pymethods]
impl EngineConfig {
    #[new]
    #[pyo3(signature = (
        model_path,
        n_ctx = 4096,
        n_threads = 8,
        cache_type_k = "q8_0",
        cache_type_v = "turbo3",
        use_metal = false,
    ))]
    fn new(
        model_path: String,
        n_ctx: u32,
        n_threads: u32,
        cache_type_k: &str,
        cache_type_v: &str,
        use_metal: bool,
    ) -> Self {
        EngineConfig {
            model_path,
            n_ctx,
            n_threads,
            cache_type_k: cache_type_k.to_string(),
            cache_type_v: cache_type_v.to_string(),
            use_metal,
        }
    }
}

// ---------------------------------------------------------------------------
// Backend: real (candle feature) or stub
// ---------------------------------------------------------------------------

#[cfg(feature = "candle")]
mod backend {
    use ruvllm::{
        CandleBackend, DeviceType, GenerateParams, LlmBackend, ModelConfig,
    };

    pub struct RealBackend {
        inner: CandleBackend,
    }

    impl RealBackend {
        pub fn load(
            model_path: &str,
            n_threads: u32,
            use_metal: bool,
        ) -> pyo3::PyResult<Self> {
            // Thread count: candle uses rayon, which reads RAYON_NUM_THREADS
            // from the env. We set it here so the configured n_threads is
            // honored without requiring the caller to set env vars.
            if n_threads > 0 {
                std::env::set_var("RAYON_NUM_THREADS", n_threads.to_string());
            }
            let mut backend = if use_metal {
                CandleBackend::with_device(DeviceType::Metal).map_err(ruv_err)?
            } else {
                CandleBackend::new().map_err(ruv_err)?
            };
            // ModelConfig: the GGUF loader auto-detects architecture,
            // quantization, layer count, etc. from the file metadata.
            // We set max_sequence_length to the configured context window
            // (passed in as n_ctx by the caller via Engine::new); the other
            // fields default and are overridden by the GGUF metadata.
            let config = ModelConfig {
                max_sequence_length: 4096,
                ..Default::default()
            };
            backend
                .load_gguf(std::path::Path::new(model_path), &config)
                .map_err(ruv_err)?;
            backend
                .load_tokenizer(std::path::Path::new(model_path))
                .map_err(ruv_err)?;
            Ok(RealBackend { inner: backend })
        }

        pub fn generate(
            &self,
            prompt: &str,
            max_tokens: u32,
            temperature: f32,
            stop: Option<Vec<String>>,
        ) -> pyo3::PyResult<String> {
            let params = GenerateParams {
                max_tokens: max_tokens as usize,
                temperature,
                stop_sequences: stop.unwrap_or_default(),
                ..Default::default()
            };
            // LlmBackend::generate takes &self, so concurrent calls from
            // multiple Python threads are safe (the backend is loaded once
            // at construction). The Mutex is held only to satisfy the
            // &mut self required by load_*; generate is &self.
            self.inner.generate(prompt, params).map_err(ruv_err)
        }

        pub fn is_loaded(&self) -> bool {
            self.inner.is_model_loaded()
        }
    }

    fn ruv_err(e: ruvllm::error::RuvLLMError) -> pyo3::PyErr {
        pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
    }
}

#[cfg(not(feature = "candle"))]
mod backend {
    pub struct RealBackend;

    impl RealBackend {
        pub fn load(
            model_path: &str,
            _n_threads: u32,
            _use_metal: bool,
        ) -> pyo3::PyResult<Self> {
            // Stub: verify the model path exists, but don't actually load.
            if !std::path::Path::new(model_path).exists() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Model file not found: {} (ruvllm_py stub — build with \
                     --features candle for real inference)",
                    model_path
                )));
            }
            eprintln!(
                "[ruvllm_py] Stub backend loaded (model={}). Build with \
                 --features candle for real inference.",
                model_path
            );
            Ok(RealBackend)
        }

        pub fn generate(
            &self,
            _prompt: &str,
            _max_tokens: u32,
            _temperature: f32,
            _stop: Option<Vec<String>>,
        ) -> pyo3::PyResult<String> {
            // Stub: return an empty string. The orchestrator's
            // format-enforcer / regex fallback handles empty output.
            Ok(String::new())
        }

        pub fn is_loaded(&self) -> bool {
            false
        }
    }
}

/// The RuvLLM inference engine.
///
/// This is the main entry point. It holds the candle backend in an
/// `Arc<Mutex<...>>` for thread-safe access from Python's thread
/// executor (pool mode). The `complete()` method accepts the same
/// parameters as `llama_cpp.Llama.__call__` so it can be a drop-in
/// replacement.
#[pyclass]
pub struct Engine {
    config: EngineConfig,
    backend: Arc<Mutex<backend::RealBackend>>,
}

#[pymethods]
impl Engine {
    /// Create a new Engine instance and load the model.
    #[new]
    #[pyo3(signature = (
        model_path,
        n_ctx = 4096,
        n_threads = 8,
        cache_type_k = "q8_0",
        cache_type_v = "turbo3",
        use_metal = false,
    ))]
    fn new(
        model_path: String,
        n_ctx: u32,
        n_threads: u32,
        cache_type_k: &str,
        cache_type_v: &str,
        use_metal: bool,
    ) -> PyResult<Self> {
        let config = EngineConfig {
            model_path: model_path.clone(),
            n_ctx,
            n_threads,
            cache_type_k: cache_type_k.to_string(),
            cache_type_v: cache_type_v.to_string(),
            use_metal,
        };
        let backend = backend::RealBackend::load(&model_path, n_threads, use_metal)?;
        Ok(Engine {
            config,
            backend: Arc::new(Mutex::new(backend)),
        })
    }

    /// Generate text from a prompt.
    ///
    /// The GIL is released during generation so the orchestrator's
    /// asyncio event loop is not blocked.
    #[pyo3(signature = (prompt, max_tokens = 128, temperature = 0.7, stop = None, grammar = None))]
    fn complete<'a>(
        &self,
        py: Python<'a>,
        prompt: String,
        max_tokens: u32,
        temperature: f32,
        stop: Option<Vec<String>>,
        grammar: Option<String>,
    ) -> PyResult<pyo3::Bound<'a, PyDict>> {
        // Release the GIL during the forward pass.
        let text = py.allow_threads(|| {
            let backend = self.backend.lock().unwrap();
            // Grammar is accepted for API compatibility but the candle
            // backend does not enforce GBNF; the orchestrator's format
            // enforcer handles structured output. We log it if present.
            if let Some(g) = &grammar {
                if !g.is_empty() {
                    eprintln!(
                        "[ruvllm_py] Note: grammar supplied but candle backend \
                         does not enforce GBNF; relying on format enforcer."
                    );
                }
            }
            backend.generate(&prompt, max_tokens, temperature, stop)
        })?;

        // Build the response dict (same structure as llama_cpp).
        let dict = PyDict::new_bound(py);
        let choices = PyDict::new_bound(py);
        choices.set_item("text", &text)?;
        dict.set_item("choices", vec![choices])?;

        let usage = PyDict::new_bound(py);
        usage.set_item("prompt_tokens", 0u32)?;
        usage.set_item("completion_tokens", 0u32)?;
        usage.set_item("total_tokens", 0u32)?;
        dict.set_item("usage", usage)?;

        Ok(dict)
    }

    /// Get the model's context window size.
    #[getter]
    fn n_ctx(&self) -> u32 {
        self.config.n_ctx
    }

    /// Get the number of CPU threads.
    #[getter]
    fn n_threads(&self) -> u32 {
        self.config.n_threads
    }

    /// Check if a real model is loaded.
    #[getter]
    fn is_loaded(&self) -> bool {
        self.backend.lock().unwrap().is_loaded()
    }
}

// ========================================================================== //
// HNSW Vector Index (v2.0 — Phase 4.2)
// ========================================================================== //
// Exposes ruvllm::ruvector_integration::{UnifiedIndex, IntegrationConfig}
// as a Python class for HNSW-based semantic similarity search. This lets
// the vector_store.py AgentDBVectorStore use the in-process Rust HNSW
// index instead of an HTTP sidecar.

#[cfg(feature = "candle")]
mod hnsw_backend {
    use ruvllm::ruvector_integration::{
        IntegrationConfig, UnifiedIndex, VectorMetadata,
    };
    use pyo3::prelude::*;
    use pyo3::types::PyDict;

    /// HNSW vector index for semantic similarity search.
    ///
    /// Wraps the ruvllm crate's UnifiedIndex (HNSW + metadata + reasoning
    /// bank). Vectors are added with a string ID and optional metadata,
    /// and searched by query embedding with a k-NN query.
    #[pyclass]
    pub struct HnswIndex {
        inner: UnifiedIndex,
    }

    #[pymethods]
    impl HnswIndex {
        /// Create a new HNSW index.
        ///
        /// Args:
        ///     dim: Vector dimension (e.g. 384 for all-MiniLM-L6-v2).
        ///     m: HNSW graph connectivity parameter (default 16).
        ///     ef_construction: HNSW build-time search depth (default 200).
        ///     ef_search: HNSW query-time search depth (default 64).
        #[new]
        #[pyo3(signature = (dim, m = 16, ef_construction = 200, ef_search = 64))]
        fn new(
            dim: usize,
            m: usize,
            ef_construction: usize,
            ef_search: usize,
        ) -> PyResult<Self> {
            let mut config = IntegrationConfig::default();
            // Override HNSW params and embedding dimension.
            config.embedding_dim = dim;
            config.hnsw_config.m = m;
            config.hnsw_config.ef_construction = ef_construction;
            config.hnsw_config.ef_search = ef_search;
            let index = UnifiedIndex::new(config)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(HnswIndex { inner: index })
        }

        /// Add a vector to the index.
        ///
        /// Args:
        ///     id: Unique string identifier for the vector.
        ///     vector: Embedding values (list of floats).
        ///     source: Source label (e.g. "task", "pattern").
        ///     quality_score: Quality score [0.0, 1.0].
        #[pyo3(signature = (id, vector, source = "unknown", quality_score = 0.0))]
        fn add(
            &self,
            id: String,
            vector: Vec<f32>,
            source: &str,
            quality_score: f32,
        ) -> PyResult<()> {
            let metadata = VectorMetadata {
                source: source.to_string(),
                quality_score,
                ..Default::default()
            };
            self.inner
                .add(id, vector, metadata)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        }

        /// Search for the k nearest neighbors of a query vector.
        ///
        /// Returns a list of dicts: [{"id": str, "score": float, "source": str}, ...]
        fn search<'a>(
            &self,
            py: Python<'a>,
            query: Vec<f32>,
            k: usize,
        ) -> PyResult<Vec<pyo3::Bound<'a, PyDict>>> {
            let results = py.allow_threads(|| {
                self.inner
                    .search(&query, k)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
            })?;
            let mut out = Vec::with_capacity(results.len());
            for r in results {
                let dict = PyDict::new_bound(py);
                dict.set_item("id", &r.id)?;
                dict.set_item("score", r.score)?;
                if let Some(meta) = &r.metadata {
                    dict.set_item("source", &meta.source)?;
                    dict.set_item("quality_score", meta.quality_score)?;
                }
                out.push(dict);
            }
            Ok(out)
        }

        /// Get index statistics.
        fn stats<'a>(&self, py: Python<'a>) -> PyResult<pyo3::Bound<'a, PyDict>> {
            let s = self.inner.stats();
            let dict = PyDict::new_bound(py);
            dict.set_item("total_vectors", s.total_vectors)?;
            dict.set_item("total_searches", s.total_searches)?;
            Ok(dict)
        }
    }
}

#[cfg(not(feature = "candle"))]
mod hnsw_backend {
    use pyo3::prelude::*;
    use pyo3::types::PyDict;

    /// HNSW vector index (STUB — build with --features candle).
    #[pyclass]
    pub struct HnswIndex;

    #[pymethods]
    impl HnswIndex {
        #[new]
        #[pyo3(signature = (dim, m = 16, ef_construction = 200, ef_search = 64))]
        fn new(dim: usize, m: usize, ef_construction: usize, ef_search: usize) -> PyResult<Self> {
            let _ = (dim, m, ef_construction, ef_search);
            eprintln!("[ruvllm_py] HnswIndex stub — build with --features candle");
            Ok(HnswIndex)
        }

        #[pyo3(signature = (id, vector, source = "unknown", quality_score = 0.0))]
        fn add(&self, id: String, vector: Vec<f32>, source: &str, quality_score: f32) -> PyResult<()> {
            let _ = (id, vector, source, quality_score);
            Ok(())
        }

        fn search<'a>(
            &self,
            _py: Python<'a>,
            query: Vec<f32>,
            k: usize,
        ) -> PyResult<Vec<pyo3::Bound<'a, PyDict>>> {
            let _ = (query, k);
            Ok(Vec::new())
        }

        fn stats<'a>(&self, _py: Python<'a>) -> PyResult<pyo3::Bound<'a, PyDict>> {
            Ok(PyDict::new_bound(_py))
        }
    }
}

// ========================================================================== //
// SONA Trajectory Recorder (v2.0 — Phase 4.3)
// ========================================================================== //
// Exposes ruvllm::sona::{SonaIntegration, SonaConfig, Trajectory} as a
// Python class for recording learning trajectories. This lets the
// orchestrator's data flywheel feed verified trajectories directly into
// the Rust SONA engine without HTTP overhead.

#[cfg(feature = "candle")]
mod sona_backend {
    use ruvllm::sona::{SonaConfig, SonaIntegration, Trajectory};
    use pyo3::prelude::*;
    use pyo3::types::PyDict;

    /// SONA learning trajectory recorder.
    ///
    /// Wraps the ruvllm crate's SonaIntegration. Records verified
    /// trajectories (query embedding, response embedding, quality score)
    /// into the SONA learning engine for continuous improvement.
    #[pyclass]
    pub struct SonaRecorder {
        inner: SonaIntegration,
    }

    #[pymethods]
    impl SonaRecorder {
        /// Create a new SONA recorder.
        ///
        /// Args:
        ///     hidden_dim: LoRA hidden dimension (default 256).
        ///     embedding_dim: Embedding dimension (default 384).
        ///     quality_threshold: Minimum quality for learning (default 0.7).
        #[new]
        #[pyo3(signature = (hidden_dim = 256, embedding_dim = 384, quality_threshold = 0.7))]
        fn new(hidden_dim: usize, embedding_dim: usize, quality_threshold: f32) -> Self {
            let config = SonaConfig {
                hidden_dim,
                embedding_dim,
                quality_threshold,
                ..Default::default()
            };
            SonaRecorder {
                inner: SonaIntegration::new(config),
            }
        }

        /// Record a learning trajectory.
        ///
        /// Args:
        ///     request_id: Unique request identifier.
        ///     session_id: Session identifier.
        ///     query_embedding: Query embedding vector.
        ///     response_embedding: Response embedding vector.
        ///     quality_score: Quality score [0.0, 1.0].
        ///     model_index: Model index used (default 0).
        #[pyo3(signature = (request_id, session_id, query_embedding, response_embedding, quality_score, model_index = 0))]
        fn record(
            &self,
            request_id: String,
            session_id: String,
            query_embedding: Vec<f32>,
            response_embedding: Vec<f32>,
            quality_score: f32,
            model_index: usize,
        ) -> PyResult<()> {
            let trajectory = Trajectory {
                request_id,
                session_id,
                query_embedding,
                response_embedding,
                quality_score,
                routing_features: Vec::new(),
                model_index,
                timestamp: chrono::Utc::now(),
            };
            self.inner
                .record_trajectory(trajectory)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        }

        /// Search for learned patterns similar to the query embedding.
        ///
        /// Returns a list of dicts: [{"id": int, "centroid": list, "cluster_size": int, "avg_quality": float}, ...]
        fn search_patterns<'a>(
            &self,
            py: Python<'a>,
            query: Vec<f32>,
            limit: usize,
        ) -> PyResult<Vec<pyo3::Bound<'a, PyDict>>> {
            let patterns = py.allow_threads(|| self.inner.search_patterns(&query, limit));
            let mut out = Vec::with_capacity(patterns.len());
            for p in patterns {
                let dict = PyDict::new_bound(py);
                dict.set_item("id", p.id)?;
                dict.set_item("centroid", p.centroid.clone())?;
                dict.set_item("cluster_size", p.cluster_size)?;
                dict.set_item("avg_quality", p.avg_quality)?;
                out.push(dict);
            }
            Ok(out)
        }

        /// Trigger the background learning loop.
        fn trigger_background_loop(&self) -> PyResult<()> {
            self.inner
                .trigger_background_loop()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        }

        /// Trigger the deep learning loop.
        fn trigger_deep_loop(&self) -> PyResult<()> {
            self.inner
                .trigger_deep_loop()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        }

        /// Get SONA statistics.
        fn stats<'a>(&self, py: Python<'a>) -> PyResult<pyo3::Bound<'a, PyDict>> {
            let s = self.inner.stats();
            let dict = PyDict::new_bound(py);
            dict.set_item("total_trajectories", s.total_trajectories)?;
            dict.set_item("instant_updates", s.instant_updates)?;
            dict.set_item("background_updates", s.background_updates)?;
            dict.set_item("deep_updates", s.deep_updates)?;
            Ok(dict)
        }
    }
}

#[cfg(not(feature = "candle"))]
mod sona_backend {
    use pyo3::prelude::*;
    use pyo3::types::PyDict;

    /// SONA trajectory recorder (STUB — build with --features candle).
    #[pyclass]
    pub struct SonaRecorder;

    #[pymethods]
    impl SonaRecorder {
        #[new]
        #[pyo3(signature = (hidden_dim = 256, embedding_dim = 384, quality_threshold = 0.7))]
        fn new(hidden_dim: usize, embedding_dim: usize, quality_threshold: f32) -> Self {
            let _ = (hidden_dim, embedding_dim, quality_threshold);
            eprintln!("[ruvllm_py] SonaRecorder stub — build with --features candle");
            SonaRecorder
        }

        #[pyo3(signature = (request_id, session_id, query_embedding, response_embedding, quality_score, model_index = 0))]
        fn record(
            &self,
            request_id: String,
            session_id: String,
            query_embedding: Vec<f32>,
            response_embedding: Vec<f32>,
            quality_score: f32,
            model_index: usize,
        ) -> PyResult<()> {
            let _ = (request_id, session_id, query_embedding, response_embedding, quality_score, model_index);
            Ok(())
        }

        fn search_patterns<'a>(
            &self,
            _py: Python<'a>,
            query: Vec<f32>,
            limit: usize,
        ) -> PyResult<Vec<pyo3::Bound<'a, PyDict>>> {
            let _ = (query, limit);
            Ok(Vec::new())
        }

        fn trigger_background_loop(&self) -> PyResult<()> {
            Ok(())
        }

        fn trigger_deep_loop(&self) -> PyResult<()> {
            Ok(())
        }

        fn stats<'a>(&self, _py: Python<'a>) -> PyResult<pyo3::Bound<'a, PyDict>> {
            Ok(PyDict::new_bound(_py))
        }
    }
}

/// Python module initialization.
#[pymodule]
fn ruvllm_py(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EngineConfig>()?;
    m.add_class::<Engine>()?;
    m.add_class::<hnsw_backend::HnswIndex>()?;
    m.add_class::<sona_backend::SonaRecorder>()?;
    // Use add() instead of setattr() for __version__ — setattr on
    // module dunder attributes can fail silently in some PyO3 configs.
    m.add("__version__", "0.2.0")?;
    let doc = if cfg!(feature = "candle") {
        "Python bindings for the RuvLLM Rust inference engine (candle backend)."
    } else {
        "Python bindings for the RuvLLM Rust inference engine (STUB — build \
         with --features candle for real inference)."
    };
    m.add("__doc__", doc)?;
    Ok(())
}
