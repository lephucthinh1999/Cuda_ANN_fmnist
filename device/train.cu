#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>

#define HEIGHT 28
#define WIDTH 28
#define N_IN 784 
#define N1 128
#define N2 128
#define N_OUT 10
#define LEARNING_RATE 1e-3
#define EPOCHS 3
#define EPSILON 1e-3
#define NTRAINING 60000

const char *training_label_fn= "mnist/train-labels-idx1-ubyte";
const char *training_image_fn= "mnist/train-images-idx3-ubyte";
const char *model_fn= "model.dat";

#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
		fprintf(stderr, "code: %d, reason: %s\n", error,\
				cudaGetErrorString(error));\
		exit(EXIT_FAILURE);\
	}\
}

struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		cudaEventRecord(start,0);
		cudaEventSynchronize(start);
	}

	void Stop()
	{
		cudaEventRecord(stop, 0);
	}

	float Elapsed()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};

// Hàm mở file
FILE* openFile(const char *fileName, const char *mode) {
    FILE *file = fopen(fileName, mode);
    if (file == NULL) {
        printf("Cannot open file %s\n", fileName);
        exit(EXIT_FAILURE);
    }
    return file;
}

// Hàm đọc header của file
void readHeader(FILE *file, int headerSize) {
    char buffer;
    for (int i = 0; i < headerSize; i++) {
        if (fread(&buffer, sizeof(char), 1, file) != 1) {
            printf("Error reading header\n");
            fclose(file);
            exit(EXIT_FAILURE);
        }
    }
}

// Hàm đọc dữ liệu từ file ảnh và file nhãn
void readInput(FILE *imageFile , FILE *labelFile, 
               char *input, double *expected) {
    char buffer;

    // Đọc dữ liệu ảnh
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            if (fread(&buffer, sizeof(char), 1, imageFile) != 1) {
                printf("Error reading image data\n");
                fclose(imageFile);
                fclose(labelFile);
                exit(EXIT_FAILURE);
            }
            input[i * WIDTH + j] = (buffer!=0);
        }
    }

    // Đọc nhãn và chuyển đổi thành one-hot vector
    if (fread(&buffer, sizeof(char), 1, labelFile) != 1) {
        printf("Error reading label data\n");
        fclose(imageFile);
        fclose(labelFile);
        exit(EXIT_FAILURE);
    }

    // Kiểm tra giá trị nhãn hợp lệ
    if ((unsigned char)buffer >= N_OUT) {
        printf("Invalid label value: %d\n", (unsigned char)buffer);
        fclose(imageFile);
        fclose(labelFile);
        exit(EXIT_FAILURE);
    }

    // Khởi tạo giá trị cho vector `expected`
    for (int i = 0; i < N_OUT; i++) {
        expected[i] = 0.0;
    }
    expected[(unsigned char)buffer] = 1.0;
}


// CUDA kernel for softmax
__global__ void softmax(double *in_out, double *output, int n) {
    extern __shared__ double shared_mem[]; 
    
    int tid = threadIdx.x;                      
    int global_id = blockIdx.x * blockDim.x + tid; 

    if (global_id < n) {
        shared_mem[tid] = exp(in_out[global_id]);
    } else {
        shared_mem[tid] = 0.0;
    }
    __syncthreads();

   
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0 && tid + stride < blockDim.x) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }

    double block_sum = shared_mem[0];

    if (global_id < n) {
        output[global_id] = exp(in_out[global_id]) / block_sum + 1e-5;
    }
}

// CUDA kernel for ReLU
__global__ void ReLU(double *layer, double *out_layer, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out_layer[idx] = (layer[idx] > 0) ? layer[idx] : 0;
    }
}

// CUDA kernel for forward propagation
__global__ void forward_propagation(double *input, double *w, double *b, double *output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows) {
        double sum = b[idx];
        for (int j = 0; j < cols; ++j) {
            sum += w[idx * cols + j] * input[j];
        }
        output[idx] = sum;
    }
}

// CUDA kernel for delta3 calculation
__global__ void compute_delta3(double *output, double *expected, double *delta3, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        delta3[idx] = output[idx] - expected[idx];
    }
}

// CUDA kernel for delta2 and delta1 calculation
__global__ void compute_delta(double *delta_next, double *w_next, double *delta, double *layer, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < cols) {
        double sum = 0.0;
        for (int i = 0; i < rows; i++) {
            sum += w_next[i * cols + idx] * delta_next[i];
        }
        delta[idx] = sum * (layer[idx] > 0 ? 1 : 0);
    }
}

__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// CUDA kernel for backward propagation (weights and biases update)
__global__ void backward_propagation_W(double *delta, double *w, double *input, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < cols) {
        for (int i = 0; i < rows; ++i) {
            atomicAddDouble(&w[i * cols + idx], -LEARNING_RATE * delta[i] * input[idx]);
        }
    }
}

__global__ void backward_propagation_B(double *delta, double *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        atomicAddDouble(&b[idx], -LEARNING_RATE * delta[idx]);
    }
}

__global__ void cross_entropy_kernel(double* expected, double* output, double* loss, int n) {
    extern __shared__ double shared_loss[]; 

    int tid = threadIdx.x;  
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  

    if (idx < n) {
        shared_loss[tid] = expected[idx] * log(output[idx]);
    } else {
        shared_loss[tid] = 0.0;
    }
    __syncthreads();

    // Reduction to calculate block sum
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_loss[tid] += shared_loss[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAddDouble(loss, -1 * shared_loss[0]);
    }
}

// Initialize weights and biases
void init(double *weight, double *bias, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int sign = rand() % 2; // +-1
            weight[i * cols + j] = (rand() % 6) / 10.0;
            if (sign)
                weight[i * cols + j] = -weight[i * cols + j];
        }
    }

    for (int i = 0; i < rows; ++i) {
        int sign = rand() % 2;
        bias[i] = (rand() % 10 + 1) / (10.0 + cols);
        if (sign)
            bias[i] = -bias[i];
    }
}

// Write weight, bias 
void write_matrix(FILE* file, double *weight, double *bias, int rows, int cols) {
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            fprintf(file, "%lf ", weight[i * cols + j]);
        }
        fprintf(file, "\n");
    }

    for (int i = 0; i < rows; ++i)
    {
        fprintf(file, "%lf ", bias[i]);
    }
    fprintf(file, "\n");
}

// Write model
void write_model(double *h_w1, double *h_b1, double *h_w2, double *h_b2, double *h_w3, double *h_b3) {
    FILE *file = openFile(model_fn, "w");

    write_matrix(file, h_w1, h_b1, N1, N_IN);
    write_matrix(file, h_w2, h_b2, N2, N1);
    write_matrix(file, h_w3, h_b3, N_OUT, N2);

    fclose(file);
}

void train (double *h_w1, double *h_b1, double *h_w2, double *h_b2, double *h_w3, double *h_b3,
             char *h_input, double *h_expected, FILE *imageFile, FILE *labelFile){


    size_t shared_memory_size = 256 * sizeof(double);

    // Allocate device memory
    double *d_w1, *d_b1, *d_w2, *d_b2, *d_w3, *d_b3;
    CHECK(cudaMalloc(&d_w1, N1 * N_IN * sizeof(double)));
    CHECK(cudaMalloc(&d_b1, N1 * sizeof(double)));
    CHECK(cudaMalloc(&d_w2, N2 * N1 * sizeof(double)));
    CHECK(cudaMalloc(&d_b2, N2 * sizeof(double)));
    CHECK(cudaMalloc(&d_w3, N_OUT * N2 * sizeof(double)));
    CHECK(cudaMalloc(&d_b3, N_OUT * sizeof(double)));
    
    // Copy data to device
    CHECK(cudaMemcpy(d_w1, h_w1, N1 * N_IN * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b1, h_b1, N1 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_w2, h_w2, N2 * N1 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b2, h_b2, N2 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_w3, h_w3, N_OUT * N2 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b3, h_b3, N_OUT * sizeof(double), cudaMemcpyHostToDevice));

    // Implement training loop (forward + backward propagation)
    // Allocate inputs, outputs, and deltas here
    double *d_input, *d_output1, *d_output2, *d_output;
    double *d_delta1, *d_delta2, *d_delta3, *d_expected;
    double *d_loss;
    CHECK(cudaMalloc(&d_input, N_IN * sizeof(double)));
    CHECK(cudaMalloc(&d_output1, N1 * sizeof(double)));
    CHECK(cudaMalloc(&d_output2, N2 * sizeof(double)));
    CHECK(cudaMalloc(&d_output, N_OUT * sizeof(double)));
    CHECK(cudaMalloc(&d_delta1, N1 * sizeof(double)));
    CHECK(cudaMalloc(&d_delta2, N2 * sizeof(double)));
    CHECK(cudaMalloc(&d_delta3, N_OUT * sizeof(double)));
    CHECK(cudaMalloc(&d_expected, N_OUT * sizeof(double)));
    CHECK(cudaMalloc(&d_loss, sizeof(double)));

    GpuTimer timer; 
    timer.Start();

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
      printf("Epoch %d\n",epoch + 1);
        // Read header
        readHeader(imageFile, 16); // Header image (16 bytes)
        readHeader(labelFile, 8);  // Header lable (8 bytes)

        for (int i = 0; i < NTRAINING; ++i) {
            printf("Sample %d ",i + 1);
            // Read data
            readInput(imageFile, labelFile, h_input, h_expected);
            //Copy data to device
            CHECK(cudaMemcpy(d_input, h_input, N_IN * sizeof(double), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_expected, h_expected, N_OUT * sizeof(double), cudaMemcpyHostToDevice));

            // Forward pass
            forward_propagation<<<(N1 + 255) / 256, 256>>>(d_input, d_w1, d_b1, d_output1, N1, N_IN);
            CHECK(cudaGetLastError());
		    CHECK(cudaDeviceSynchronize());

            ReLU<<<(N1 + 255) / 256, 256>>>(d_output1, d_output1, N1);
            CHECK(cudaGetLastError());
		    CHECK(cudaDeviceSynchronize());

            forward_propagation<<<(N2 + 255) / 256, 256>>>(d_output1, d_w2, d_b2, d_output2, N2, N1);
            CHECK(cudaGetLastError());
		    CHECK(cudaDeviceSynchronize());

            ReLU<<<(N2 + 255) / 256, 256>>>(d_output2, d_output2, N2);
            CHECK(cudaGetLastError());
		    CHECK(cudaDeviceSynchronize());

            forward_propagation<<<(N_OUT + 255) / 256, 256>>>(d_output2, d_w3, d_b3, d_output, N_OUT, N2);
            CHECK(cudaGetLastError());
		    CHECK(cudaDeviceSynchronize());

            
            softmax<<<(N_OUT + 255) / 256, 256, shared_memory_size>>>(d_output, d_output, N_OUT);
            CHECK(cudaGetLastError());
		    CHECK(cudaDeviceSynchronize());

            // Backward pass
            compute_delta3<<<(N_OUT + 255) / 256, 256>>>(d_output, d_expected, d_delta3, N_OUT);
            CHECK(cudaGetLastError());
		    CHECK(cudaDeviceSynchronize());

            backward_propagation_W<<<(N2 + 255) / 256, 256>>>(d_delta3, d_w3, d_output2, N_OUT, N2);
            CHECK(cudaGetLastError());
		    CHECK(cudaDeviceSynchronize());

            backward_propagation_B<<<(N_OUT + 255) / 256, 256>>>(d_delta3, d_b3, N_OUT);
            CHECK(cudaGetLastError());
		    CHECK(cudaDeviceSynchronize());

            compute_delta<<<(N2 + 255) / 256, 256>>>(d_delta3, d_w3, d_delta2, d_output2, N_OUT, N2);
            CHECK(cudaGetLastError());
		    CHECK(cudaDeviceSynchronize());

            backward_propagation_W<<<(N1 + 255) / 256, 256>>>(d_delta2, d_w2, d_output1, N2, N1);
            CHECK(cudaGetLastError());
		    CHECK(cudaDeviceSynchronize());

            backward_propagation_B<<<(N2 + 255) / 256, 256>>>(d_delta2, d_b2, N2);
            CHECK(cudaGetLastError());
		    CHECK(cudaDeviceSynchronize());

            compute_delta<<<(N1 + 255) / 256, 256>>>(d_delta2, d_w2, d_delta1, d_output1, N2, N1);
            CHECK(cudaGetLastError());
		    CHECK(cudaDeviceSynchronize());

            backward_propagation_W<<<(N_IN + 255) / 256, 256>>>(d_delta1, d_w1, d_input, N1, N_IN);
            CHECK(cudaGetLastError());
		    CHECK(cudaDeviceSynchronize());

            backward_propagation_B<<<(N1 + 255) / 256, 256>>>(d_delta1, d_b1, N1);
            CHECK(cudaGetLastError());
		    CHECK(cudaDeviceSynchronize());

            CHECK(cudaMemset(d_loss, 0, sizeof(double)));
            cross_entropy_kernel<<<(N_OUT + 255) / 256, 256, shared_memory_size>>>(d_expected, d_output, d_loss, N_OUT);
            CHECK(cudaGetLastError());
		    CHECK(cudaDeviceSynchronize());

            double h_loss;
            CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(double), cudaMemcpyDeviceToHost));
            printf("Loss: %f\n", h_loss);  
            if (h_loss < EPSILON){
                break;
            }
        }

        // Copy weights and biases back to host after every epoch
        CHECK(cudaMemcpy(h_w1, d_w1, N1 * N_IN * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(h_b1, d_b1, N1 * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(h_w2, d_w2, N2 * N1 * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(h_b2, d_b2, N2 * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(h_w3, d_w3, N_OUT * N2 * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(h_b3, d_b3, N_OUT * sizeof(double), cudaMemcpyDeviceToHost));

        // Lưu mô hình sau mỗi epoch
        write_model(h_w1, h_b1, h_w2, h_b2, h_w3, h_b3);

        rewind(imageFile);
        rewind(labelFile);
    }

    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());

    // Free allocated memory on both device and host
    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output1));
    CHECK(cudaFree(d_output2));
    CHECK(cudaFree(d_output));
    CHECK(cudaFree(d_delta1));
    CHECK(cudaFree(d_delta2));
    CHECK(cudaFree(d_delta3));
    CHECK(cudaFree(d_expected));

    CHECK(cudaFree(d_w1));
    CHECK(cudaFree(d_b1));
    CHECK(cudaFree(d_w2));
    CHECK(cudaFree(d_b2));
    CHECK(cudaFree(d_w3));
    CHECK(cudaFree(d_b3));
}

int main(int argc, char ** argv)
{
    //Open image file, label file
    FILE *imageFile = openFile(training_image_fn, "rb");
    FILE *labelFile = openFile(training_label_fn, "rb");

    // Allocate and initialize weights and biases on host
    char *h_input = new char[N_IN];
    double *h_expected = new double[N_OUT];

    double *h_w1 = (double *)malloc(N1 * N_IN * sizeof(double));
    double *h_b1 = (double *)malloc(N1 * sizeof(double));
    double *h_w2 = (double *)malloc(N2 * N1 * sizeof(double));
    double *h_b2 = (double *)malloc(N2 * sizeof(double));
    double *h_w3 = (double *)malloc(N_OUT * N2 * sizeof(double));
    double *h_b3 = (double *)malloc(N_OUT * sizeof(double));

    init(h_w1, h_b1, N1, N_IN);
    init(h_w2, h_b2, N2, N1);
    init(h_w3, h_b3, N_OUT, N2);

    train(h_w1, h_b1, h_w2, h_b2, h_w3, h_b3, h_input, h_expected, imageFile, labelFile);

    free(h_input);
    free(h_expected);

    free(h_w1);
    free(h_b1);
    free(h_w2);
    free(h_b2);
    free(h_w3);
    free(h_b3);

    fclose(imageFile);
    fclose(labelFile);
    
}
