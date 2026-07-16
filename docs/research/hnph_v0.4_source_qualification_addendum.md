# HNPH v0.4 Source-Qualification Addendum

**Frozen v0.4 protocol SHA-256:** `401a8e47db3aefc5c549fec956254f291f71e43c6bf636c464551df552acc839`

**Status:** source qualification is `UNVERIFIED`; DOD migration, training, model selection, and external evaluation are not authorized by this document.

This addendum binds the fail-closed qualification required by Kahlus HNPH protocol v0.4. It does not report dataset qualification and does not claim that the currently downloadable archives contain every required rater stream or physical-metadata field.

## Required local qualification packet

The qualification runner must produce paired JSON and Markdown artifacts containing only subject-safe identifiers and hashes. Local raw-data paths and raw neural samples must not enter git. For both DOD-H and DOD-O the packet must bind:

1. dataset/version and license identifier;
2. immutable archive or source hash;
3. person identity field used for grouped splitting;
4. channel names, physical units, and sampling rates;
5. five separately identifiable independent rater streams and a SHA-256 for each annotation stream;
6. a leave-one-rater-out audit proving that the held-out rater is absent from the consensus target;
7. `external_opened: false` for DOD-O until protocol and model-family freeze.

Missing fields, fewer than five source raters, fewer than three contributing consensus raters, target-rater contamination, or premature external access produce a nonzero exit and `source_qualification_fail`. The gate must not fall back to consensus-only labels.

## Freeze rule

This addendum and the protocol hash above are immutable after claim-mode data access. Any change creates a new protocol version and new qualification packet. Protocol v0.3 remains preserved for audit but is superseded for future HNPH claims.
