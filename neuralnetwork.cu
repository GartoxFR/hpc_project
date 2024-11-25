
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include<cublas_v2.h>
#include <vector>
#include <cassert>
#include <numeric>
#include <string>
#include <fstream>

using namespace std;

#define THREADS_PER_BLOCK 32

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

__global__ void sigmoid_kernel(const float* src, float* dest, int len){
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        dest[i] = sigmoid(src[i]);
    }
}

// dest[i] = sigmoid_derivative(A[i])*B[i]
__global__ void sigmoid_derivative_kernel(const float* A, const float* B, float* dest, int len){
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        dest[i] = sigmoid_derivative(A[i])*B[i];
    }
}

__global__ void output_error_kernel(const float* expected_outputs, const float* computed_outputs, float* err, int len){
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        err[i] = (expected_outputs[i] - computed_outputs[i]) * sigmoid_derivative(computed_outputs[i]);
    }
}


struct NNParams {
    uint seed = 1;
    float learning_rate = 5-3;
    int input_size = 3;
    int output_size = 1;
    int nb_layer = 12;
    int min_nodes_per_layer = 6;
    int max_nodes_per_layer = 12;
} nnParams;

struct Matrix {
    vector<float> data;
    int rows = 0, cols = 0;
    float* d_data = nullptr;

    Matrix() = default;

    Matrix(const Matrix& mat) = delete;
    Matrix& operator=(const Matrix& mat) = delete;

    Matrix(Matrix&& mat) : data(std::move(mat.data)), rows(mat.rows), cols(mat.cols), d_data(std::exchange(mat.d_data, nullptr)) {
    }

    Matrix& operator=(Matrix&& mat) {
        data = std::move(mat.data);
        rows = mat.rows;
        cols = mat.cols;
        d_data = std::exchange(mat.d_data, nullptr);
        return *this;
    }

    Matrix(int rows, int cols) : data(rows*cols), rows(rows), cols(cols) {}

    static Matrix random(int rows, int cols) {
        Matrix res(rows, cols);
        for(int i = 0; i < rows  * cols; i++) {
            res.data[i] = 2 * rand_float() - 1;
        }
        return res;
    };

    ~Matrix() {
        if (d_data != nullptr) {
            checkError(cudaFree(d_data));
        }
    }

    void allocateOnDevice() {
        if(d_data == nullptr) {
            checkError(cudaMalloc(&d_data, rows * cols * sizeof(float)));
        }
    }

    void copyToDevice() {
        allocateOnDevice();
        checkError(cudaMemcpy(d_data, (const void*) data.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice));
    }

    void copyFromDevice() {
        if (d_data != nullptr) {
            checkError(cudaMemcpy((void*) data.data(), d_data, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
        }
    }
};

struct TrainParams {
    Matrix training_set_inputs;
    Matrix training_set_outputs;
    int number_of_training_iterations = 100000;
    int freq = 10000;
} trainParams;

struct Input{
    Matrix x;
    Matrix y;
};

void print_matrix(const int &m, const int &n, const float *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%0.2f ", A[j * lda + i]);
        }
        printf("\n");
    }
}

void multiplyMatrices(const Matrix& a, const Matrix& b, Matrix& c, bool transposeA, bool transposeB,
    const cublasHandle_t& cublasHandle, const cudaStream_t& stream, float alpha = 1.0f, float beta = 0.0f){
    // Parameters
    cublasOperation_t transa = (transposeA ? CUBLAS_OP_T : CUBLAS_OP_N);
    cublasOperation_t transb = (transposeB ? CUBLAS_OP_T : CUBLAS_OP_N);

    int rowsA = (transposeA ? a.cols : a.rows), colsA = a.cols + a.rows - rowsA;
    int rowsB = (transposeB ? b.cols : b.rows), colsB = b.cols + b.rows - rowsB;
    
    int m = rowsA;
    int n = colsB;
    int k = colsA;
    
    assert(colsA == rowsB);
    assert(c.rows == rowsA);
    assert(c.cols == colsB);
   
    int lda = a.rows; // rows of A
    int ldb = b.rows; // rows of B
    int ldc = c.rows;


    // Multiply matrices
    checkCublasError(cublasSgemm(cublasHandle, transa, transb, m, n, k, &alpha, a.d_data, lda, b.d_data, ldb, &beta, c.d_data, ldc));

    //checkError(cudaStreamSynchronize(stream));
}

__global__ void hello_world() {
    int i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    printf("Hello from the GPU, index %d\n", i);
}

void forward_pass(const Matrix& inputs, const vector<Matrix>& synaptic_weights, vector<Matrix>& network,const cublasHandle_t& cublasHandle, const cudaStream_t& stream) {
    for(int i = 0; i < nnParams.nb_layer+1;i++) {
        if(i == 0) {
            multiplyMatrices(inputs, synaptic_weights[i], network[i], false, false, cublasHandle, stream);
            sigmoid_kernel<<<1, THREADS_PER_BLOCK, 0, stream>>>(network[i].d_data, network[i].d_data, network[i].rows*network[i].cols);
        } else {
            multiplyMatrices(network[i-1], synaptic_weights[i], network[i], false, false, cublasHandle, stream);
            sigmoid_kernel<<<1, THREADS_PER_BLOCK, 0, stream>>>(network[i].d_data, network[i].d_data, network[i].rows*network[i].cols);
        }
    }
}

void compute_errors(const vector<Matrix>& network, const Matrix& outputs, const vector<Matrix>& weights, 
    vector<Matrix>& errors, cublasHandle_t& cublasHandle, cudaStream_t& stream) {
    int output_index = errors.size() - 1;
    output_error_kernel<<<1, THREADS_PER_BLOCK, 0, stream>>>(outputs.d_data, network[output_index].d_data, errors[output_index].d_data, outputs.rows * outputs.cols);
    checkLastError();
    for (int i = output_index - 1; i >= 0; i--) {
        multiplyMatrices(errors[i + 1], weights[i + 1], errors[i], false, true, cublasHandle, stream);
        sigmoid_derivative_kernel<<<1, THREADS_PER_BLOCK, 0, stream>>>(network[i].d_data, errors[i].d_data, errors[i].d_data, errors[i].rows * errors[i].cols);
    }
}

vector<Matrix> allocate_network(const vector<int>& layers, int number_of_inputs, int output_size) {
    vector<Matrix> res;
    res.reserve(layers.size() + 1);
    for (int cols : layers) {
        res.emplace_back(number_of_inputs, cols);
        res.back().allocateOnDevice();
    }

    res.emplace_back(number_of_inputs, output_size);
    res.back().allocateOnDevice();

    return res;
}

void train(const TrainParams& trainParams, const NNParams& nnParams, const vector<int>& layers, vector<Matrix>& weights,
            const Matrix& inputs, const Matrix& outputs, cublasHandle_t& cublasHandle, cudaStream_t& stream) {
    vector<float> error_array;

    vector<Matrix> network = allocate_network(layers, inputs.rows, outputs.cols);
    vector<Matrix> errors = allocate_network(layers, inputs.rows, outputs.cols);

    for(int iteration = 0; iteration < trainParams.number_of_training_iterations; iteration++) {
        forward_pass(inputs, weights, network, cublasHandle, stream);
        compute_errors(network, outputs, weights, errors, cublasHandle, stream);

        checkLastError();
        if (iteration % trainParams.freq == 0) {
            errors[layers.size()].copyFromDevice();
            float mean = std::abs(std::accumulate(errors[layers.size()].data.begin(), errors[layers.size()].data.end(), 0.0f)) / errors[layers.size()].data.size();
            printf("Err = %e\n", mean);
        }

        for (int i = 0; i < nnParams.nb_layer+1; i++) {
            if (i == 0) {
                multiplyMatrices(inputs, errors[i], weights[i], true, false, cublasHandle, stream, nnParams.learning_rate, 1.0f);
            } else {
                multiplyMatrices(network[i - 1], errors[i], weights[i], true, false, cublasHandle, stream, nnParams.learning_rate, 1.0f);
            }
        }
    }
    
}


Input readInput(const string path){
    ifstream file;
    file.open(path);
    string line;
    std::vector<float> y;
    std::vector<std::vector<float>> x_rowMajor;
    while(getline(file, line)){
        std::vector<float> temp;
        size_t input_end = line.find(']');
        string input = line.substr(1,input_end-1);
        int pos = 0;
        while(true){
            int next_separator = input.find(',',pos);
            float x = stof(input.substr(pos,next_separator-pos));
            temp.push_back(x);


            if(next_separator == std::string::npos){
                break;
            }
            else{
                pos = next_separator+1;
            }
        }
        x_rowMajor.push_back(temp);
        float out = stof(line.substr(input_end+2));
        y.push_back(out);
    }
    Matrix x_matrix(x_rowMajor.size(), x_rowMajor[0].size()), y_matrix(y.size(),1);
    for(int i = 0;i<x_matrix.rows;i++){
        y_matrix.data[i] = y[i];
        for(int j = 0;j<x_matrix.cols;j++){
            x_matrix.data[i+x_matrix.rows*j] = x_rowMajor[i][j];
        }
    }
    file.close();
    return Input{.x=std::move(x_matrix), .y=std::move(y_matrix)};
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

    srand(nnParams.seed);

    vector<int> layers(nnParams.nb_layer);
    for (int i = 0; i < nnParams.nb_layer; i++) {
        layers[i] = rand_range(nnParams.min_nodes_per_layer, nnParams.max_nodes_per_layer);
    }

    vector<Matrix> synaptic_weights(layers.size() + 1);
    for(int i = 0; i < layers.size()+1; i++) {
        if(i == 0) {
            synaptic_weights[i] = Matrix::random(nnParams.input_size, layers[0]);
        } else if(i == layers.size() ) { 
            synaptic_weights[i] = Matrix::random(layers[i-1], nnParams.output_size);
        } else {
            synaptic_weights[i] = Matrix::random(layers[i-1], layers[i]);
        }
        synaptic_weights[i].copyToDevice();
    }

    Input input = readInput("input.txt");
    input.x.copyToDevice();
    input.y.copyToDevice();
    train(trainParams, nnParams, layers, synaptic_weights, input.x, input.y, cublasHandle, stream);

    cudaDeviceSynchronize();
    cublasDestroy(cublasHandle);
}