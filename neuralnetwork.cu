#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include<cublas_v2.h>
#include <vector>

using namespace std;

#define THREADS_PER_BLOCK 128

#define checkError(exp) {cudaError_t err = exp; if (err != cudaSuccess) { \
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
}}

#define checkCublasError(exp) {cublasStatus_t err = exp; if (err != CUBLAS_STATUS_SUCCESS) { \
        printf("%s in %s at line %d\n", cublasGetStatusName(err), __FILE__, __LINE__); \
}}

#define checkLastError() checkError(cudaGetLastError())

static int rand_range(int min, int max) {
    return rand() % (max + 1 - min) + min;
}

// Returns a random float in [0, 1]
static float rand_float() {
    return (float) rand() / (float) RAND_MAX;
}

__device__ float sigmoid(float x) {
    return 1/(1+expf(x));
}

__device__ float sigmoid_derivative(float x) {
    return x*(1-x);
}

struct NNParams {
    uint seed = 1;
    float learning_rate = 1e-3;
    int input_size = 3;
    int output_size = 1;
    int nb_layer = 12;
    int min_nodes_per_layer = 2;
    int max_nodes_per_layer = 12;
};

struct Matrix {
    vector<float> data;
    int rows = 0, cols = 0;

    Matrix() = default;

    Matrix(int rows, int cols) : data(rows*cols), rows(rows), cols(cols) {
        for(int i = 0; i < rows  * cols; i++) {
            data[i] = 2* rand_float() - 1;
        }
    };
};

struct TrainParams {
    Matrix training_set_inputs;
    Matrix training_set_outputs;
    int number_of_training_iterations = 100000;
    int freq = 10000;
};

void print_matrix(const int &m, const int &n, const float *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%0.2f ", A[j * lda + i]);
        }
        printf("\n");
    }
}

void multiplyMatrices(cublasHandle_t& cublasHandle, cudaStream_t& stream){
    // Parameters
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    int m=2,n=2,k=2;
    const std::vector<float> A({1.0, 2.0, 3.0, 4.0}); // m x k
    const std::vector<float> B({5.0, 6.0, 7.0, 8.0}); // k x n
    std::vector<float> C(4); // m x n
    const float alpha = 1.0;
    const float beta = 0.0;
    int lda = 2; // rows of A
    int ldb = 2; // rows of B
    int ldc = 2;

    //Copy memory to GPU
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    using data_type = float;
    checkError(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
    checkError(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(data_type) * B.size()));
    checkError(cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(data_type) * C.size()));

    checkError(cudaMemcpy(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice));
    checkError(cudaMemcpy(d_B, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice));

    // Multiply matrices
    checkCublasError(cublasSgemm(cublasHandle, transa, transb, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc));
    checkError(cudaMemcpyAsync((void*) C.data(), d_C, sizeof(data_type) * C.size(), cudaMemcpyDeviceToHost,
                               stream));
    checkLastError();

    checkError(cudaStreamSynchronize(stream));


    printf("C\n");
    print_matrix(m, n, C.data(), ldc);
    printf("=====\n");

    /* free resources */
    checkError(cudaFree(d_A));
    checkError(cudaFree(d_B));
    checkError(cudaFree(d_C));
}

__global__ void hello_world() {
    int i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    printf("Hello from the GPU, index %d\n", i);
}


int main(int argc, char** argv) {
    // void* test;
    // checkError(cudaMalloc(&test, 1000000000000000000000000000000000));
    // checkLastError();
    
    // Create cublas handle and bind it to a stream
    cublasHandle_t cublasHandle = NULL;
    cudaStream_t stream = NULL;
    checkCublasError(cublasCreate(&cublasHandle));
    checkError(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    checkCublasError(cublasSetStream(cublasHandle, stream));

    NNParams params;

    srand(params.seed);

    vector<int> layers(params.nb_layer);
    for (int i = 0; i < params.nb_layer; i++) {
        layers[i] = rand_range(params.min_nodes_per_layer, params.max_nodes_per_layer);
    }

    vector<Matrix> synaptic_weights(layers.size() + 1);
    for(int i = 0; i < layers.size()+1; i++) {
        if(i == 0) {
            synaptic_weights[i] = Matrix(params.input_size, layers[0]);
        } else if(i == layers.size() ) { 
            synaptic_weights[i] = Matrix(layers[i-1], params.output_size);
        } else {
            synaptic_weights[i] = Matrix(layers[i-1], layers[i]);
        }
    }

    vector<float> error_array;

    // for(int iteration = 0; i < params.numbe)

    // hello_world<<<32, THREADS_PER_BLOCK>>>();
    multiplyMatrices(cublasHandle, stream);
    cudaDeviceSynchronize();
}