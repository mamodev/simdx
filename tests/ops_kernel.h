#pragma once

#include <simdx/simdx.h>

namespace ops {

constexpr int kValues = 64;

struct Results {
    float add[kValues], sub[kValues], mul[kValues], div[kValues], neg[kValues];
    float fma[kValues], abs[kValues], min[kValues], max[kValues], sqrt[kValues];
    simdx::Mask lt[kValues], le[kValues], gt[kValues], ge[kValues], eq[kValues], ne[kValues];
    float sum;
    int lanes;
};

using Apply = void(const float* a, const float* b, const float* c, Results& out);

SIMDX_DECLARE(Apply apply)

}  // namespace ops
