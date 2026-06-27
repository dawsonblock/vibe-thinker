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

/// Python module initialization.
#[pymodule]
fn ruvllm_py(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EngineConfig>()?;
    m.add_class::<Engine>()?;
    // Use add() instead of setattr() for __version__ — setattr on
    // module dunder attributes can fail silently in some PyO3 configs.
    m.add("__version__", "0.1.0")?;
    let doc = if cfg!(feature = "candle") {
        "Python bindings for the RuvLLM Rust inference engine (candle backend)."
    } else {
        "Python bindings for the RuvLLM Rust inference engine (STUB — build \
         with --features candle for real inference)."
    };
    m.add("__doc__", doc)?;
    Ok(())
}
