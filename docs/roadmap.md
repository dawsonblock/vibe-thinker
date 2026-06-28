# Roadmap

## Frozen until stabilization is complete

No new features until:
- wheel builds;
- core tests pass;
- optional tests are properly marked;
- Mac local profile runs;
- sandbox security claims are corrected.

## Deferred

- RuvLLM production inference
- AgentDB integration
- distributed federation hardening
- web dashboard
- advanced adaptive learning
- SONA / self-optimizing runtime

## Reintroduction order (after core is green)

1. Core CLI + local model
2. Exact cache
3. Hash similarity
4. Embedding vector store
5. Logic verifier with Z3
6. Docker sandbox without network
7. Docker sandbox with enforced gateway
8. Redis federation
9. Web dashboard/PubSub
10. RuvLLM backend
11. Advanced learning/memory promotion

Do not skip this order. Features must be layered back in one at a
time with proper gates, not stacked.
