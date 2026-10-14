#include "scale.h"

namespace dsp::SIMDX_ISA {

void scale(float* x, int n, float k)
{
    using simdx::Float;
    const Float factor = Float::broadcast(k);
    int i = 0;
    for (; i + Float::kLanes <= n; i += Float::kLanes)
        (Float::load(x + i) * factor).store(x + i);
    for (; i < n; i++)
        x[i] *= k;
}

}  // namespace dsp::SIMDX_ISA
