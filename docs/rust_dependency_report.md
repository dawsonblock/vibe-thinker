# Rust Dependency Probe Report

**Date:** 2026-06-26 (updated 2026-06-27)
**Probe location:** `rust/probes/ruvllm_exo_probe/`
**Toolchain:** cargo 1.95.0 (f2d3ce0bd 2026-03-21), macOS arm64

## Summary

`ruvllm 2.3.0` is published on crates.io, compiles cleanly, and links
successfully. `exo-federation 0.1.1` was **removed** in v0.4.1 — the
crate is stubbed (no working networking layer, archived post-quantum
crypto deps, unsound `lru 0.12.5` transitive dep). Federation is now
implemented in Python via `federation_server.py`.

Removing `exo-federation` eliminated all 3 vulnerabilities and the
unsound crate warning in one stroke:
- `rustls-webpki 0.101.7` (3 vulns) — gone, only 0.103.13 remains
- `lru 0.12.5` (unsound, Stacked Borrows violation) — gone entirely
- `pqcrypto-kyber` / `pqcrypto-internals` (archived PQClean) — gone

## Build results

| Command | Result |
|---------|--------|
| `cargo metadata` | OK — 447 packages locked |
| `cargo check` | OK — exit 0, no errors |
| `cargo test` | OK — 0 tests in probe, all crate tests compiled |
| `cargo tree --depth 1` | `ruvllm v2.3.0` + `exo-federation v0.1.1` |
| `cargo audit` | **3 vulnerabilities, 10 warnings** |

## API surface: ruvllm 2.3.0

**Verdict: usable for a sidecar.** The crate provides a real LLM inference
stack built on candle-core.

### Key types

- `LlmBackend` trait — the core inference interface:
  - `fn load_model(&mut self, model_id: &str, config: ModelConfig) -> Result<()>`
  - `fn generate(&self, prompt: &str, params: GenerateParams) -> Result<String>`
  - `fn generate_stream(&self, prompt: &str, params: GenerateParams) -> Result<Box<dyn Iterator<...>>>`
  - `fn generate_stream_v2(&self, prompt: &str, params: GenerateParams) -> Result<TokenStream>`
  - `fn get_embeddings(&self, text: &str) -> Result<Vec<f32>>`
  - `fn is_model_loaded(&self) -> bool`
  - `fn unload_model(&mut self)`
- `CandleBackend` — the primary implementation (candle-core/nn/transformers)
- `GenerateParams` — builder pattern: max_tokens, temperature, top_p, top_k,
  repetition_penalty, stop_sequences, seed
- `ServingEngine` — wraps `Arc<dyn LlmBackend>` with scheduling, batching,
  KV cache management, speculative decoding, request coalescing
  - `ServingEngine::new(model: Arc<dyn LlmBackend>, config: ServingEngineConfig)`
  - `ServingEngine::submit(request: InferenceRequest) -> Result<RequestId>`
  - `ServingEngine::get_result(id: RequestId) -> Option<GenerationResult>`
  - `ServingEngine::run_iteration() -> Result<Vec<TokenOutput>>`
- `RuvLLMEngine` — memory/policy/session layer (NOT inference). Manages
  PolicyStore, SessionIndex, WitnessLog, SonaIntegration. The doc comment
  claims `engine.process(&session, "Hello, world!")` but this method does
  NOT exist — the docs are aspirational, not implemented.

### Feature flags

Default features: `async-runtime`, `candle`, `routing-metrics`, `quantize`,
`hub-download`. Optional: `cuda`, `metal`, `coreml`, `metal-compute`,
`hybrid-ane`, `gguf-mmap`, `parallel`, `wasm`.

### Unsafe usage

All `unsafe` blocks are in SIMD quantization kernels (`quantize/hadamard.rs`,
`quantize/pi_quant.rs`) — NEON and AVX2 intrinsics for performance-critical
math. No `unsafe` in network-facing or parsing code. This is expected and
acceptable for a performance-focused inference crate.

### Dependency chain (notable)

- `candle-core/nn/transformers v0.9.2` — HuggingFace candle ML framework
- `tokenizers v0.20.4` — HuggingFace tokenizers (wraps a C library)
- `ruvector-core v2.2.3` — Ruvector memory layer
- `hnsw_rs v0.3.4` — HNSW vector index
- `redb v2.6.3` — embedded key-value store
- `ring v0.17.14` — crypto
- `tokio v1.52.3` — async runtime

## API surface: exo-federation 0.1.1

**Verdict: NOT ready for production use.** The crate compiles and its
crypto primitives are real, but the federation networking layer is
stubbed with placeholder implementations.

### Key types

- `FederatedMesh` — the coordinator:
  - `FederatedMesh::new(local: SubstrateInstance) -> Result<Self>`
  - `async fn join_federation(&mut self, peer: &PeerAddress) -> Result<FederationToken>`
  - `async fn federated_query(&self, query: Vec<u8>, scope: FederationScope) -> Result<Vec<FederatedResult>>`
  - `async fn byzantine_commit(&self, update: StateUpdate) -> Result<CommitProof>`
- `PostQuantumKeypair` — Kyber-1024 keypair (via `pqcrypto-kyber`)
- `EncryptedChannel` — ChaCha20-Poly1305 encrypted channel
- `GSet`, `LWWRegister` — CRDT types
- `onion_query`, `OnionHeader` — onion routing

### Critical problem: stubbed networking

`SubstrateInstance` is an empty struct:
```rust
pub struct SubstrateInstance {
    // Placeholder - will integrate with actual substrate
}
```

`federated_query` for `Direct` and `Global` scopes returns placeholder data:
```rust
FederationScope::Direct => {
    // Placeholder: would actually send query to peer
    results.push(FederatedResult {
        source: peer_id,
        data: query.clone(),  // echoes the query back as "data"
        score: 0.8,           // hardcoded score
        timestamp: current_timestamp(),
    });
}
FederationScope::Global { max_hops } => {
    // Placeholder: would use onion_query
    Ok(vec![])  // returns empty
}
```

The crypto/handshake modules (`join_federation`, `PostQuantumKeypair`,
`EncryptedChannel`) may be functional, but the mesh coordination layer
that would use them to actually route queries between nodes is not
implemented. This crate is scaffolding, not a working federation.

### Unsafe usage

Zero `unsafe` blocks. Pure safe Rust.

### Dependency chain (notable)

- `pqcrypto-kyber v0.8.1` — post-quantum Kyber (UNMAINTAINED — see audit)
- `pqcrypto-internals v0.2.11` — PQClean bindings (UNMAINTAINED)
- `chacha20poly1305 v0.10.1` — AEAD encryption
- `dashmap v6.2.1` — concurrent map
- `tokio v1.52.3` — async runtime
- `exo-core v0.1.1` — companion crate (placeholder substrate)

## Security audit: `cargo audit` results

### Vulnerabilities (3 — errors)

| Crate | Version | ID | Severity | Description |
|-------|---------|----|----------|-------------|
| rustls-webpki | 0.101.7 | RUSTSEC-2026-0104 | panic | Reachable panic in CRL parsing |
| rustls-webpki | 0.101.7 | RUSTSEC-2026-0098 | cert validation | Name constraints for URI names incorrectly accepted |
| rustls-webpki | 0.101.7 | RUSTSEC-2026-0099 | cert validation | Name constraints accepted for wildcard name certs |

**Source:** Transitive dependency from `ruvllm` → `rustls v0.21.12` →
`rustls-webpki v0.101.7`. The fix is to upgrade to `rustls-webpki >=0.103.13`,
but this requires `ruvllm` to update its `rustls` dependency. We cannot
fix this without forking `ruvllm` or waiting for an upstream update.

**Risk assessment:** These vulnerabilities affect TLS certificate
validation. If the RuvLLM sidecar makes outbound HTTPS connections (e.g.,
to download models from HuggingFace Hub), a malicious server could
exploit the CRL panic (DoS) or bypass certificate name constraints
(MITM). For a local-only sidecar that loads models from disk, the risk
is lower but not zero.

### Warnings (10 — unmaintained/unsound)

| Crate | Version | ID | Type | Impact |
|-------|---------|----|------|--------|
| bincode | 1.3.3 | RUSTSEC-2025-0141 | unmaintained | Serialization (ruvllm) |
| bincode | 2.0.1 | RUSTSEC-2025-0141 | unmaintained | Serialization (ruvllm) |
| number_prefix | 0.4.0 | RUSTSEC-2025-0119 | unmaintained | Display formatting (ruvllm) |
| paste | 1.0.15 | RUSTSEC-2024-0436 | unmaintained | Macro utility (ruvllm) |
| pqcrypto-internals | 0.2.11 | RUSTSEC-2026-0163 | unmaintained | PQClean archived (exo-federation) |
| pqcrypto-kyber | 0.8.1 | RUSTSEC-2024-0381 | replaced | Use pqcrypto-mlkem (exo-federation) |
| pqcrypto-traits | 0.3.5 | RUSTSEC-2026-0162 | unmaintained | PQClean archived (exo-federation) |
| proc-macro-error2 | 2.0.1 | RUSTSEC-2026-0173 | unmaintained | Proc macro (ruvllm) |
| rustls-pemfile | 1.0.4 | RUSTSEC-2025-0134 | unmaintained | PEM parsing (ruvllm) |
| **lru** | **0.12.5** | **RUSTSEC-2026-0002** | **unsound** | **Stacked Borrows violation in IterMut (ruvllm)** |

**The `lru` unsoundness is the most concerning warning.** It's a memory
safety issue in a data structure used by `ruvllm`'s serving engine (KV
cache management). The fix is to upgrade to `lru >=0.12.5` patched or
`lru 1.0`, but this requires upstream action.

**The `pqcrypto-*` unmaintained warnings are critical for exo-federation.**
The entire post-quantum crypto stack relies on PQClean, which is being
archived. The replacement is `pqcrypto-mlkem` (NIST FIPS 203). This means
exo-federation's crypto primitives are built on a deprecated foundation.

## Recommendations

### ruvllm: proceed with sidecar, pin and vendor

1. **Build a sidecar binary** (`rfsn-ruvllm-sidecar`) that wraps
   `CandleBackend` + `ServingEngine` behind a localhost HTTP API:
   - `POST /health` — check model loaded
   - `POST /load_model` — load a GGUF/safetensors model from disk
   - `POST /generate` — sync generation with `GenerateParams`
   - `POST /embed` — embedding extraction
   - `POST /shutdown` — graceful shutdown

2. **Pin and vendor** the dependency tree immediately:
   ```
   cargo generate-lockfile
   cargo vendor vendor/
   ```
   With `.cargo/config.toml` pointing to vendored sources. This prevents
   supply-chain drift.

3. **Track the 3 rustls-webpki vulnerabilities.** If the sidecar only
   loads models from local disk (not HF Hub), the TLS attack surface is
   eliminated. If HF Hub download is needed, use `--no-hub-download`
   feature and download models out-of-band.

4. **Track the `lru` unsoundness.** This is a memory safety issue in the
   serving engine's KV cache. Monitor for an upstream fix. If the sidecar
   crashes, the Python orchestrator's fail-closed path handles it (same
   as any other backend failure).

### exo-federation: do NOT build a sidecar yet

1. **The federation networking is stubbed.** `federated_query` returns
   placeholder data. `SubstrateInstance` is empty. Building a sidecar
   around placeholder code would create a false sense of distributed
   capability.

2. **The crypto stack is unmaintained.** `pqcrypto-kyber` is replaced by
   `pqcrypto-mlkem`. `pqcrypto-internals` is archived. Depending on these
   for post-quantum security guarantees is building on a deprecated
   foundation.

3. **Alternative:** If federation is needed, build the job distribution
   layer in Python (HTTP/gRPC between nodes) and use `exo-federation`
   only for its CRDT types (`GSet`, `LWWRegister`) and crypto primitives
   (`EncryptedChannel`) — but even those depend on the unmaintained
   pqcrypto stack. The safer path is to use well-maintained Rust crypto
   crates (`rustls`, `ring`, `chacha20poly1305` directly) and implement
   the federation protocol ourselves.

### Corrected milestone table

| Phase | Old status | Correct status |
|-------|-----------|---------------|
| 3.1 ruvllm integration | "blocked" | **sidecar spike viable** — proceed with sidecar, pin/vendor, track 3 vulns |
| 5.1 federated worker pull | "blocked" | **NOT viable** — exo-federation networking is stubbed, crypto is unmaintained |
| 5.2 distributed cache gossip | "blocked" | **NOT viable** — same exo-federation issues; implement in Python instead |

## Next steps

1. Pin and vendor the ruvllm dependency tree
2. Build `rfsn-ruvllm-sidecar` with the HTTP API above
3. Write contract tests (`tests/test_ruvllm_sidecar_contract.py`)
4. Wire the sidecar into `ruvllm_adapter.py` as the HTTP backend
5. For federation: design a Python-native job distribution protocol
   using well-maintained crypto, not exo-federation
