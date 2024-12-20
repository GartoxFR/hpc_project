#pragma once

#include <vector>
#include <iostream>
#include <assert.h>


#define checkError(exp) {cudaError_t err = exp; if (err != cudaSuccess) { \
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
}}

#define checkCublasError(exp) {cublasStatus_t err = exp; if (err != CUBLAS_STATUS_SUCCESS) { \
        printf("%s in %s at line %d\n", cublasGetStatusName(err), __FILE__, __LINE__); \
}}

// Returns a random float in [0, 1]
static inline float rand_float() {
    return (float) rand() / (float) RAND_MAX;
}

using namespace std;

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

struct MatrixBatched {
    int count = 0;
    int rows = 0, cols = 0;
    vector<float*> d_data;
    float* data_start = nullptr;
    float** d_descriptor = nullptr;
    bool fake = true;

    MatrixBatched() = default;
    MatrixBatched(int count, int rows, int cols) : count(count), rows(rows), cols(cols)
    {
        allocateOnDevice();
    }

    MatrixBatched(Matrix& mat, int count) : count(count), rows(mat.rows), cols(mat.cols), data_start(mat.d_data), d_data(count, mat.d_data) {
        assert(data_start != nullptr);
        copyDescriptor();
    }

    MatrixBatched(const MatrixBatched&) = delete;
    MatrixBatched& operator=(const MatrixBatched&) = delete;

    MatrixBatched(MatrixBatched&& o) : count(o.count), rows(o.rows), cols(o.cols),
        d_data(std::move(o.d_data)), data_start(std::exchange(o.data_start, nullptr))
    {}

    MatrixBatched& operator=(MatrixBatched&& o) 
    {
        count = o.count;
        rows = o.rows;
        cols = o.cols;

        d_data = std::move(o.d_data);
        data_start = std::exchange(o.data_start, nullptr);
        return *this;
    }
    
    ~MatrixBatched() {
        if (data_start != nullptr && !fake) {
            checkError(cudaFree(data_start));
        }
        if (d_descriptor != nullptr) {
            checkError(cudaFree(d_descriptor));
        }
    }

    void allocateOnDevice() {
        fake = false;
        if(data_start == nullptr) {
            checkError(cudaMalloc(&data_start, count * rows * cols * sizeof(float)));
        }

        float* ptr = data_start;
        for (int i = 0; i < count; i++) {
            d_data.push_back(ptr);
            ptr += rows * cols;
        }

        copyDescriptor();
        
    }

    void copyDescriptor() {
        checkError(cudaMalloc(&d_descriptor, count * sizeof(float**)));
        checkError(cudaMemcpy(d_descriptor, d_data.data(), count * sizeof(float**), cudaMemcpyHostToDevice));
    }

    void copySubmatrix(const Matrix& matrix, int start_row, cudaStream_t stream) {
        assert(!fake);
        int end_row = min(start_row + count, matrix.rows);
        int size = end_row - start_row;

        // assert(matrix.cols == rows);
        // assert(size < count * cols);
        assert(size > 0);

        checkError(cudaMemcpyAsync(data_start, matrix.data.data() + start_row * rows, size * rows * sizeof(float), cudaMemcpyHostToDevice, stream));
    }

    void copyFromDevice(vector<float>& dst, cudaStream_t stream) {
        dst.clear();
        dst.resize(count * rows * cols);
        cudaMemcpyAsync(dst.data(), data_start, count * rows * cols * sizeof(float), cudaMemcpyDeviceToHost, stream);
    }

};