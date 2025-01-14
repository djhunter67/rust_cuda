// matrix_multiply.cu
extern "C" __global__
void matrix_multiply(const float *a, const float *b, float *c, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        float value = 0.0f;
        for (int k = 0; k < width; ++k) {
            value += a[row * width + k] * b[k * width + col];
        }
        c[row * width + col] = value;
    }
}