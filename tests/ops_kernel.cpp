#include "ops_kernel.h"

namespace ops::SIMDX_ISA {

void apply(const float* a, const float* b, const float* c, Results& out)
{
    using simdx::Float;
    Float total = Float::zero();
    for (int i = 0; i < kValues; i += Float::kLanes) {
        const Float x = Float::load(a + i), y = Float::load(b + i), z = Float::load(c + i);
        (x + y).store(out.add + i);
        (x - y).store(out.sub + i);
        (x * y).store(out.mul + i);
        (x / y).store(out.div + i);
        (-x).store(out.neg + i);
        fma(x, y, z).store(out.fma + i);
        abs(x).store(out.abs + i);
        min(x, y).store(out.min + i);
        max(x, y).store(out.max + i);
        sqrt(abs(x)).store(out.sqrt + i);
        out.lt[i] = x < y;
        out.le[i] = x <= y;
        out.gt[i] = x > y;
        out.ge[i] = x >= y;
        out.eq[i] = x == y;
        out.ne[i] = x != y;
        total += x;
    }
    out.sum = sum(total);
    out.lanes = Float::kLanes;
}

}  // namespace ops::SIMDX_ISA
