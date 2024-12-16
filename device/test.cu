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
#define EPSILON 1e-3
#define NTESTING 10000

const char * testing_label_fn="mnist/t10k-labels-idx1-ubyte";
const char * testing_image_fn="mnist/t10k-images-idx3-ubyte";
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
int readInput(FILE *imageFile , FILE *labelFile, 
               double *input, double *expected) {
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
            input[i * WIDTH + j] = (double) buffer / 255.0;
        }
    }

    // Đọc nhãn và chuyển đổi thành one-hot vector
    if (fread(&buffer, sizeof(char), 1, labelFile) != 1) {
        printf("Error reading label data\n");
        fclose(imageFile);
        fclose(labelFile);
        exit(EXIT_FAILURE);
    }

    // Khởi tạo giá trị cho vector `expected`
    for (int i = 0; i < N_OUT; i++) {
        expected[i] = 0.0;
    }
    expected[buffer] = 1.0;
    return (int) buffer;
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

void load_matrix(FILE *file, double *weight, double *bias, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (fscanf(file, "%lf", &weight[i * cols + j]) != 1) {
                printf("Error reading weight[%d][%d]\n", i, j);
                exit(EXIT_FAILURE);
            }
        }
    }

    for (int i = 0; i < rows; ++i) {
        if (fscanf(file, "%lf", &bias[i]) != 1) {
            printf("Error reading bias[%d]\n", i);
            exit(EXIT_FAILURE);
        }
    }
}

void load_model(const char *file_name, double *w1, double *w2, double *w3, double *b1, double *b2, double *b3) {
    FILE *file = fopen(file_name, "r");
    if (file == NULL) {
        printf("Cannot open model file %s\n", file_name);
        exit(EXIT_FAILURE);
    }

    load_matrix(file, w1, b1, N1, N_IN);
    load_matrix(file, w2, b2, N2, N1);
    load_matrix(file, w3, b3, N_OUT, N2);

    fclose(file);
}

void test (double *h_w1, double *h_b1, double *h_w2, double *h_b2, double *h_w3, double *h_b3,
             double *h_input, double *h_expected, FILE *imageFile, FILE *labelFile){


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
    int correct = 0;
    for (int sample = 0; sample < NTESTING; sample++) {
        // Read data
        int label = readInput(imageFile, labelFile, h_input, h_expected);
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

        int predict=0;
        double h_out1;
        double h_out2;
        for (int i = 1;i < N_OUT; i++){
            CHECK(cudaMemcpy(&h_out1, &d_output[i], sizeof(double), cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(&h_out2, &d_output[predict], sizeof(double), cudaMemcpyDeviceToHost));
            if (h_out1 > h_out2){
	            predict = i;
            }
            predict--;
            if (label == predict){
	            correct++;
            }
        }
    }

    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());

    printf("Accuracy: %0.2lf\n",(double)correct/NTESTING*100.0);

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
    FILE *imageFile = openFile(testing_image_fn, "rb");
    FILE *labelFile = openFile(testing_label_fn, "rb");

    // Read header
    readHeader(imageFile, 16); // Header image (16 bytes)
    readHeader(labelFile, 8);  // Header lable (8 bytes)

    // Allocate and initialize weights and biases on host
    double *h_input = new double[N_IN];
    double *h_expected = new double[N_OUT];

    double *h_w1 = (double *)malloc(N1 * N_IN * sizeof(double));
    double *h_b1 = (double *)malloc(N1 * sizeof(double));
    double *h_w2 = (double *)malloc(N2 * N1 * sizeof(double));
    double *h_b2 = (double *)malloc(N2 * sizeof(double));
    double *h_w3 = (double *)malloc(N_OUT * N2 * sizeof(double));
    double *h_b3 = (double *)malloc(N_OUT * sizeof(double));

    load_model(model_fn, h_w1, h_w2, h_w3, h_b1, h_b2, h_b3);

    test(h_w1, h_b1, h_w2, h_b2, h_w3, h_b3, h_input, h_expected, imageFile, labelFile);

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
