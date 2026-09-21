# aien-local-stack

In-process composition stack integrating AEGIS and Sovereign Core via AIEN protocols.

## Architectural Mandate

```text
            aien-local-stack
             /           \
            ▼             ▼
      aegis-runtime   sovereign-core
            \             /
             \           /
              protocols
```

`aien-local-stack` composes the autonomous agent plane (`aegis-runtime`) and the accelerated inference engine (`aien-sovereign-core`) inside a single process without allowing either subsystem to directly import the other.
