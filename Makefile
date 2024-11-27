.PHONY: all

EXE += neuralnetwork

all: $(EXE)

clean:
	rm -f $(EXE)


%: %.cu matrix.h
	nvcc -O3 -arch=sm_75 -o $@ $< -lcublas
