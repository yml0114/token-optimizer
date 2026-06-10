# Protected Spans Guard

Date: 2026-06-11

## Goal

Protected spans make smart compression safer by extracting concrete fields that
must survive model compression verbatim before calling the cheap compressor.

This is a zero-cost deterministic guard. It does not add another model call.

## Protected span kinds

- `urls`
- `emails`
- `paths`
- `code_symbols`
- `numbers`
- `constraints`

Examples:

```text
https://api.example.com/v1/prices
support@unfaze.app
/app/data/project/main.py
parse_price()
500
```

## Runtime behavior

1. Extract protected spans from the original messages.
2. Add a `PROTECTED_SPANS` block to the cheap model compression prompt.
3. Keep the existing semantic-fidelity guard after compression.
4. Expose protected span count and sample items in compression metadata.

This creates a two-layer protection system:

```text
pre-compression protected span instruction
        +
post-compression semantic fidelity guard
```

## Why it matters

The previous fidelity guard could reject lossy output after the fact. Protected
spans reduce the chance that the cheap model loses critical fields in the first
place, especially in protected mode.

## Validation

- SmartCompressor unit tests: `38 passed`
- Full test suite: `98 passed`
- Fidelity regression: `30/30 safe pass`, `30/30 lossy reject`
- Competitive adapter benchmark unchanged: v5+cache cost saving `74.1%`, fidelity `11/11`
