CXX ?= g++
override CXXFLAGS += -O2 -std=c++17 -I../include

SIMDX_KERNELS = scale_kernel.cpp
SIMDX_OBJDIR = obj
include ../make/simdx.mk

run: obj/example
	./obj/example

obj/example: main.cpp $(addprefix obj/,$(SIMDX_OBJ)) | obj
	$(CXX) $(CXXFLAGS) -o $@ $^

$(addprefix obj/,$(SIMDX_OBJ)): | obj

obj:
	mkdir -p obj

clean:
	rm -rf obj

.PHONY: run clean
