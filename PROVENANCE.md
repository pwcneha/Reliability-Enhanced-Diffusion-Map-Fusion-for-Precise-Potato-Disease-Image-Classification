# Implementation provenance

The public `src/bgf_gate.py` module is a path-independent extraction of the
regional policy and gate functions used in the locked five-fold analysis.
Private Google Drive paths, internal experiment-stage names, immutable-marker
handling and project-specific file orchestration were removed.

Preserved behaviour includes:

- first-two-coordinate DMAP regional construction;
- development-fitted global RMS scaling and k-means;
- regional sample-support, NLL-gain and ECE-drift eligibility;
- deterministic per-region expert selection;
- both as-executed and equation-consistent candidate definitions;
- their respective deterministic ranking and tie-breaking rules;
- floor-rounded per-fold intervention ceilings; and
- development-fitted post-policy temperature scaling.

## Verification

The refactored module was compiled successfully and compared with functions
extracted directly from the locked execution source on deterministic synthetic
development and held-out arrays. Region assignments, regional policy records,
accepted masks, candidate and accepted counts, maximum edit counts and final
probability arrays matched for both BGF routes.

The locked policy audit contained no regional NLL gain equal to the threshold
value `0.003`; the nearest audited value was approximately `0.001752`.

## Source hashes

- Locked execution source SHA-256:
  `02e930686e1ede7df0a9e6aa938407815ea834525e43a70105e965503780da22`
- Public path-independent module SHA-256:
  `329f8193090aca99572b92a09b02ddaa8b376d436e86b00d7a7af56e6c28cce4`
- Locked configuration SHA-256:
  `e73b197076f03c53304b16096ab74aef58207d84558aeecc34765cb18dfb8778`
- Public configuration SHA-256:
  `0debbc0f419eafbdbec93c16d18a80dec92d3489999f061b858e62b102d1c369`

The hashes differ between locked and public files because private orchestration
content and internal storage references were deliberately removed. The
behavioural equivalence check concerns the policy functions listed above.
