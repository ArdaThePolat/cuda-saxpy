#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// SAXPY operation function
__global__ void saxpy(int n, float a, float* x, float* y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main(void)
{
    // Print device's properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU Device Name: %s\n", prop.name);
    printf("Max Threads Per Block: %d\n\n", prop.maxThreadsPerBlock);

    int n;
    float a;

    printf("Enter the size of the arrays: ");
    scanf("%d", &n);

    // Allocate memory
    float* x, * y;
    cudaMallocManaged(&x, n * sizeof(float));
    cudaMallocManaged(&y, n * sizeof(float));

    printf("Enter the scalar value: ");
    scanf("%f", &a);

    // Initialize the vectors with random numbers
    for (int i = 0; i < n; i++) {
        x[i] = rand() / (float)RAND_MAX;
        y[i] = rand() / (float)RAND_MAX;
    }

    // Print the initialized y vector
    printf("\nInitialized y vector:\n");
    for (int i = 0; i < n; i++) {
        printf("%f ", y[i]);
    }
    printf("\n\n");

    // Print the  initialized x vector
    printf("Initialized x vector:\n");
    for (int i = 0; i < n; i++) {
        printf("%f ", x[i]);
    }
    printf("\n\n");

    // Try to find the optimum block and grid size
    int tempx = floor(sqrt(n));
    saxpy << < tempx,(n+tempx)/tempx >> > (n, a, x, y);
    cudaDeviceSynchronize();

    // Print the result
    printf("Final y vector:\n");
    for (int i = 0; i < n; i++) {
        printf("%f ", y[i]);
    }
    printf("\n\n");

    // Free allocated memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}
