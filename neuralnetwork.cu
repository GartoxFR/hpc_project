
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include<cublas_v2.h>
#include <vector>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <string>
#include <fstream>
#include <sys/time.h>
#include "matrix.h"

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
    uint seed = 3;
    float learning_rate = 0.001;
    int input_size = 784;
    int output_size = 1;
    int nb_layer = 5;
    int min_nodes_per_layer = 6;
    int max_nodes_per_layer = 12;
} nnParams;



struct TrainParams {
    Matrix training_set_inputs;
    Matrix training_set_outputs;
    int number_of_training_iterations = 100;
    int freq = 10;
    int batch_size = 32;
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

void multiplyMatricesBatched(const MatrixBatched& a, const MatrixBatched& b, MatrixBatched& c, bool transposeA, bool transposeB,
    const cublasHandle_t& cublasHandle, const cudaStream_t& stream, float alpha = 1.0f, float beta = 0.0f){
    // Parameters
    cublasOperation_t transa = (transposeA ? CUBLAS_OP_T : CUBLAS_OP_N);
    cublasOperation_t transb = (transposeB ? CUBLAS_OP_T : CUBLAS_OP_N);

    int rowsA = (transposeA ? a.cols : a.rows), colsA = a.cols + a.rows - rowsA;
    int rowsB = (transposeB ? b.cols : b.rows), colsB = b.cols + b.rows - rowsB;
    
    int m = rowsA;
    int n = colsB;
    int k = colsA;
    
    assert(a.count == b.count);
    assert(a.count == c.count);
    assert(colsA == rowsB);
    assert(c.rows == rowsA);
    assert(c.cols == colsB);
   
    int lda = a.rows; // rows of A
    int ldb = b.rows; // rows of B
    int ldc = c.rows;


    // Multiply matrices
    checkCublasError(cublasSgemmBatched(
        cublasHandle, transa, transb, m, n, k, &alpha, a.d_descriptor, lda, b.d_descriptor, ldb, &beta, c.d_descriptor, ldc, a.count));

    // checkError(cudaStreamSynchronize(stream));
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

void forward_pass_batched(const MatrixBatched& inputs, const vector<MatrixBatched>& synaptic_weights, vector<MatrixBatched>& network,const cublasHandle_t& cublasHandle, const cudaStream_t& stream) {
    for(int i = 0; i < nnParams.nb_layer+1;i++) {
        if(i == 0) {
            multiplyMatricesBatched(inputs, synaptic_weights[i], network[i], true, false, cublasHandle, stream);
            sigmoid_kernel<<<1, THREADS_PER_BLOCK, 0, stream>>>(network[i].data_start, network[i].data_start, network[i].count * network[i].rows*network[i].cols);
        } else {
            multiplyMatricesBatched(network[i-1], synaptic_weights[i], network[i], false, false, cublasHandle, stream);
            sigmoid_kernel<<<1, THREADS_PER_BLOCK, 0, stream>>>(network[i].data_start, network[i].data_start, network[i].count * network[i].rows*network[i].cols);
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

void compute_errors_batched(const vector<MatrixBatched>& network, const MatrixBatched& outputs, const vector<MatrixBatched>& weights, 
    vector<MatrixBatched>& errors, cublasHandle_t& cublasHandle, cudaStream_t& stream) {
    int output_index = errors.size() - 1;
    output_error_kernel<<<1, THREADS_PER_BLOCK, 0, stream>>>(outputs.data_start, network[output_index].data_start, errors[output_index].data_start, outputs.count * outputs.rows * outputs.cols);
    checkLastError();
    for (int i = output_index - 1; i >= 0; i--) {
        multiplyMatricesBatched(errors[i + 1], weights[i + 1], errors[i], false, true, cublasHandle, stream);
        sigmoid_derivative_kernel<<<1, THREADS_PER_BLOCK, 0, stream>>>(network[i].data_start, errors[i].data_start, errors[i].data_start, errors[i].count * errors[i].rows * errors[i].cols);
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

vector<MatrixBatched> allocate_network_batched(const vector<int>& layers, int number_of_inputs, int output_size, int concurrent_batches) {
    vector<MatrixBatched> res;
    res.reserve(layers.size() + 1);
    for (int cols : layers) {
        res.emplace_back(concurrent_batches, number_of_inputs, cols);
    }

    res.emplace_back(concurrent_batches, number_of_inputs, output_size);

    return res;
}

vector<MatrixBatched> allocate_adjustments_batched(const vector<Matrix>& weights, int batch_size) {
    vector<MatrixBatched> res;
    res.reserve(weights.size());
    for (const auto& weight : weights) {
        res.emplace_back(batch_size, weight.rows, weight.cols);
    }

    return res;
}
Matrix allocate_reductions_matrices(const vector<Matrix>& weights) {
    int max_size = 0;
    for (const auto& weight : weights) {
        max_size = max(max_size, weight.rows * weight.cols);
    }

    Matrix m(max_size, 1);
    m.data = vector<float>(max_size, 1.0f);
    m.copyToDevice();

    return m;
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

void batch_train(const TrainParams& trainParams, vector<Matrix>& weights, const vector<int>& layers, const Matrix& input, const Matrix& output, cublasHandle_t cublasHandle) {
    vector<float> error_array;

    MatrixBatched input_batched[2] = {MatrixBatched(trainParams.batch_size, input.cols, 1), MatrixBatched(trainParams.batch_size, input.cols, 1)};
    MatrixBatched output_batched[2] = {MatrixBatched(trainParams.batch_size, 1, 1), MatrixBatched(trainParams.batch_size, 1, 1)};
    cudaStream_t streams[2] = {NULL, NULL};
    checkError(cudaStreamCreateWithFlags(&streams[0], cudaStreamNonBlocking));
    checkError(cudaStreamCreateWithFlags(&streams[1], cudaStreamNonBlocking));

    vector<MatrixBatched> network = allocate_network_batched(layers, 1, output.cols, trainParams.batch_size);
    vector<MatrixBatched> errors = allocate_network_batched(layers, 1, output.cols, trainParams.batch_size);
    vector<MatrixBatched> adjustments = allocate_adjustments_batched(weights, trainParams.batch_size);
    Matrix reduction_matrix = allocate_reductions_matrices(weights);

    vector<float> error_vector;

    vector<MatrixBatched> weights_batched;
    weights_batched.reserve(weights.size());
    for(int i = 0; i < weights.size(); i++) {
        weights_batched.emplace_back(weights[i], trainParams.batch_size);
    }

    if(input.rows < trainParams.batch_size) {
        fprintf(stderr, "The number of examples must be at least equal to the size of a batch (must be at least %d, is %d)\n", 
                trainParams.batch_size, input.rows);
        return;
    }

    for(int iteration = 0; iteration < trainParams.number_of_training_iterations; iteration++) {
        for(int dispatch = 0; dispatch < 2; dispatch++) {
            input_batched[dispatch].copySubmatrix(input, dispatch*trainParams.batch_size, streams[dispatch]);
            output_batched[dispatch].copySubmatrix(output, dispatch*trainParams.batch_size, streams[dispatch]);
        }
        float err = 0;
        int err_count = 0;
        for(int dispatch = 0; dispatch < (input.rows-1) / trainParams.batch_size + 1; dispatch++) {
            int dp = dispatch % 2;
            cublasSetStream(cublasHandle, streams[dp]);
            forward_pass_batched(input_batched[dp], weights_batched, network, cublasHandle, streams[dp]);
            compute_errors_batched(network, output_batched[dp], weights_batched, errors, cublasHandle, streams[dp]);

            for (int i = 0; i < nnParams.nb_layer+1; i++) {
                if (i == 0) {
                    multiplyMatricesBatched(input_batched[dp], errors[i], adjustments[i], false, false, cublasHandle, streams[dp]);
                } else {
                    multiplyMatricesBatched(network[i - 1], errors[i], adjustments[i], true, false, cublasHandle, streams[dp]);
                }
            }

            // Mean reduction
            for (int i = 0; i < nnParams.nb_layer+1; i++) {
                int m = weights[i].rows * weights[i].cols;
                int n = 1;
                int k = trainParams.batch_size;

                int lda = m;
                int ldb = k;
                int ldc = m;

                float alpha = nnParams.learning_rate / trainParams.batch_size;
                float beta = 1.0f;

                checkCublasError(cublasSgemm(
                    cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, 
                    adjustments[i].data_start, lda,
                    reduction_matrix.d_data, ldb, &beta,
                    weights[i].d_data, ldc
                ));

                // checkError(cudaStreamSynchronize(stream));
            }

            if (dispatch + 2 < (input.rows-1) / trainParams.batch_size + 1) {
                input_batched[dp].copySubmatrix(input, (dispatch + 2)*trainParams.batch_size, streams[dp]);
                output_batched[dp].copySubmatrix(output, (dispatch+2)*trainParams.batch_size, streams[dp]);
            }

            if (iteration % trainParams.freq == 0) {
                errors[errors.size() - 1].copyFromDevice(error_vector, streams[dp]);
                checkError(cudaStreamSynchronize(streams[dp]));
                std::transform(error_vector.begin(), error_vector.end(), error_vector.begin(), [](auto x) { return 
                    std::abs(x); });
                err += std::accumulate(error_vector.begin(), error_vector.end(), 0.0f);
                err_count += error_vector.size();
            }
        }

        if (iteration % trainParams.freq == 0) {
            printf("Err = %e\n", err / err_count);
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
            x_matrix.data[j+x_matrix.cols*i] = x_rowMajor[i][j];
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

    Input input = readInput("mnist.txt");
    // input.x.copyToDevice();
    // input.y.copyToDevice();

    struct timeval begin, end;

    gettimeofday(&begin, NULL);
    batch_train(trainParams, synaptic_weights, layers, input.x, input.y, cublasHandle);
    gettimeofday(&end, NULL);

    // Calculate time.
    double time = 1.0 * (end.tv_sec - begin.tv_sec) + 1.0e-6 * (end.tv_usec - begin.tv_usec);

    double time_per_iter = time / trainParams.number_of_training_iterations; 
    printf("Train time : %lfs (%les / iter)", time, time_per_iter);


    cudaDeviceSynchronize();
    cublasDestroy(cublasHandle);
}