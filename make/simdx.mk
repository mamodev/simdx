# Builds every file in SIMDX_KERNELS once per ISA (see include/simdx/simdx.h).
#
#     SIMDX_KERNELS  = my_kernels.cpp
#     SIMDX_OBJDIR   = ../obj              # default: obj
#     SIMDX_CXX      = $(GPP)              # default: $(CXX)
#     SIMDX_CXXFLAGS = $(COPT) $(DEPFLAGS) # default: $(CXXFLAGS)
#     include <simdx>/make/simdx.mk
#     OBJ += $(SIMDX_OBJ)                  # three objects per kernel file, link them all
#
# Set SIMDX_OBJDIR before the include (it is part of the rule targets); the others are read
# when the rules run. Add -I<simdx>/include to the flags.

SIMDX_ISAS := sse2 avx2 avx512
SIMDX_FLAGS_sse2 :=
SIMDX_FLAGS_avx2 := -mavx2 -mfma
SIMDX_FLAGS_avx512 := -mavx512f -mavx2 -mfma

SIMDX_OBJDIR ?= obj
SIMDX_CXX ?= $(CXX)
SIMDX_CXXFLAGS ?= $(CXXFLAGS)

SIMDX_OBJ = $(foreach kernel,$(SIMDX_KERNELS),$(foreach isa,$(SIMDX_ISAS),$(notdir $(kernel:%.cpp=%_$(isa).o))))

define SIMDX_RULE
$$(SIMDX_OBJDIR)/%_$(1).o: %.cpp
	$$(SIMDX_CXX) $$(SIMDX_CXXFLAGS) $$(SIMDX_FLAGS_$(1)) -o $$@ -c $$<
endef
$(foreach isa,$(SIMDX_ISAS),$(eval $(call SIMDX_RULE,$(isa))))
