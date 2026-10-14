CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Wpedantic -Wconversion
override CXXFLAGS += -I../include

SIMDX_KERNELS = ops_kernel.cpp
SIMDX_OBJDIR = obj
include ../make/simdx.mk

test: obj/test
	./obj/test

obj/test: test.cpp $(addprefix obj/,$(SIMDX_OBJ)) | obj
	$(CXX) $(CXXFLAGS) -o $@ $^

$(addprefix obj/,$(SIMDX_OBJ)): | obj

obj:
	mkdir -p obj

clean:
	rm -rf obj

.PHONY: test clean
