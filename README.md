# simdx

Write a SIMD kernel once; build it for SSE2, AVX2 and AVX-512; run the best one the CPU has.

Header only (`include/simdx/simdx.h`) plus a Make fragment (`make/simdx.mk`). C++17, GCC 7 or newer, x86-64.

## How it works

1. A kernel file is compiled three times, with `-mavx512f -mavx2 -mfma`, `-mavx2 -mfma` and no flags.
   `simdx.mk` adds the rules; you list the files.
2. `simdx.h` looks at the compiler's ISA macros and defines `SIMDX_ISA` as `avx512`, `avx2` or `sse2`.
   `simdx::Float` becomes a 16, 8 or 4 lane vector to match.
3. Kernels live in `namespace <yours>::SIMDX_ISA`, so the three copies get different names and all
   link into the same binary. simdx itself does the same, so a baseline caller can never pick up
   an AVX-512 copy of a simdx helper.
4. At startup `simdx::dispatch` checks the CPU once and returns the matching function pointer.

## Use

Kernel, `scale_kernel.cpp`:

```cpp
#include "scale.h"

namespace dsp::SIMDX_ISA {
void scale(float* x, int n, float k)
{
    using simdx::Float;
    for (int i = 0; i < n; i += Float::kLanes)
        (Float::load(x + i) * Float::broadcast(k)).store(x + i);
}
}
```

Declaration, `scale.h`:

```cpp
#include <simdx/simdx.h>
namespace dsp { SIMDX_DECLARE(void scale(float* x, int n, float k)) }
```

Caller:

```cpp
static const auto scale = simdx::dispatch(SIMDX_VARIANTS(dsp, scale));
scale(data, n, 2.0f);
```

Makefile:

```make
SIMDX_KERNELS = scale_kernel.cpp
SIMDX_OBJDIR  = obj
include <simdx>/make/simdx.mk
# link $(addprefix $(SIMDX_OBJDIR)/,$(SIMDX_OBJ)) with the rest
```

See `examples/` for a complete build, including the scalar tail.

## API

| | |
|---|---|
| `Float::kLanes` | 4, 8 or 16 |
| `Float::load(p)`, `store(p)` | unaligned |
| `Float::broadcast(x)`, `Float::zero()` | |
| `+ - * /`, unary `-`, `+= -= *= /=` | |
| `fma(a, b, c)` | `a * b + c`; one rounding on AVX2 / AVX-512, two on SSE2 |
| `abs`, `sqrt`, `min`, `max` | `min` / `max` return `b` when either is NaN (x86) |
| `sum(a)` | horizontal add |
| `< <= > >= == !=` | return a `Mask`; NaN lanes are false except for `!=` |
| `Mask` | `uint32_t`, bit *i* = lane *i* |
| `forEachSet(mask, f)` | calls `f(lane)` for each set bit, lowest first |
| `best()`, `supported(isa)` | CPU detection, done once |
| `dispatch(sse2, avx2, avx512)` | the function for this CPU |

## Notes

- Results can differ in the last bit between ISAs because of `fma`.
- Keep simdx out of headers that other compilers see (nvcc, for one): include it from the
  kernel files and from the `.cpp` that dispatches.
- Run `make test` to check every operation against scalar code on each ISA the machine supports.
