# Retained temporal bindings

This optional wrapper binds the separately distributed RexTemporal runtime and its existing
exact function, program, temporal-support and finite-family executors. It needs
an installed RexTemporal SDK selected through `CMAKE_PREFIX_PATH`, Cython and a
C++20 compiler. It adds no Python dependency to the C++ library.

`NativeTemporal` owns a frozen source snapshot. Declaration and execution methods
return a new snapshot and its new object identity. Older snapshots stay valid.
Whole-function domains, original denominator exclusions, participant identities,
charts and correlated families use the native codecs and replay checks.

Build and install from a dedicated environment:

```sh
CMAKE_PREFIX_PATH=/path/to/installed/sdk python -m pip install ./bindings/temporal
python -m pip install ./rcql
```

The SDK must include the installed `rex_temporal/layer_product.hpp`. The wheel
records the SDK headers, runtime archive and binding implementation. This
optional distribution coexists with `rexgraph_cpp`'s relational-runtime binding;
RCQL provides checked complete-archive conversion between their source owners.

RexTemporal and RexLogic SDKs are supplied separately with their own terms.
The default C++ bridge builds against RFC alone. SDK prefixes may be supplied
separately through `CMAKE_PREFIX_PATH`; fingerprints retain logical header and
archive identities across installation relocation.
