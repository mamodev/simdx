#include "scale.h"

#include <cstdio>

int main()
{
    static const auto scale = simdx::dispatch(SIMDX_VARIANTS(dsp, scale));

    float x[19];
    for (int i = 0; i < 19; i++)
        x[i] = float(i);
    scale(x, 19, 2.0f);

    const char* isa[] = {"sse2", "avx2", "avx512"};
    std::printf("running %s: x[18] = %g\n", isa[int(simdx::best())], double(x[18]));
}
