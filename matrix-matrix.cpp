//Yanal Odeh
//12113273

#include <immintrin.h>  // portable to all x86 compilers
#include <stdio.h>
#include <time.h>

#define DATA float
#define SIZE 128

DATA __attribute__((aligned(16))) A[SIZE][SIZE];
DATA __attribute__((aligned(16))) B[SIZE][SIZE];
DATA __attribute__((aligned(16))) C[SIZE][SIZE];

DATA r[SIZE][SIZE];

//THIS Function to get the current time in seconds
double seconds()
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return now.tv_sec + now.tv_nsec / 1000000000.0;
}

// THIS Function to initialize a matrix with random values (0 or 1)
void initialize_matrix(DATA b[SIZE][SIZE], int size)
{
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++)
        {
            b[i][j] = rand() % 2;
        }
    }
}

//  transpose a matrix
void from_matrix_to_matrix(DATA b[SIZE][SIZE], DATA bt[SIZE][SIZE], int size)
{
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++)
        {
            bt[j][i] = b[i][j];
        }
    }
}


//Matrix Multiplication algorithm
void vec(DATA s1[SIZE][SIZE], DATA s2[SIZE][SIZE], int size)
{
    int i, j, k;

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            for (k = 0; k < size; k++) {
                r[i][j] += s1[i][k] * s2[k][j];
            }
        }
    }
}

// Matrix multiplication using SSE instructions
void vec_sse(DATA m1[SIZE][SIZE], DATA m2[SIZE][SIZE], int size)
{
    DATA prod = 0;
    int i, j, k, n;

    __m128 X, Y, Z;

    for (i = 0; i < size; i += 1) {
        for (j = 0; j < size; j += 1) {
            r[i][j] = 0;
            prod = 0;
            Z[0] = Z[1] = Z[2] = Z[3] = 0;
            for (k = 0; k < size; k += 4) {
                // Load 4 values from m1 and m2, multiply them, and add to the accumulator Z
                X = _mm_load_ps(&m1[i][k]);
                Y = _mm_load_ps(&m2[j][k]);
                X = _mm_mul_ps(X, Y);
                Z = _mm_add_ps(X, Z);
            }
            // Sum the 4 values in Z and store the result in prod
            for (n = 0; n < 4; n++)
            {
                prod += Z[n];
            }
            // Store the final result in the matrix r
            r[i][j] = prod;
        }
    }
}

int main()
{
    double before, after;

    // Initialize matrices A and B with random values (0 or 1)
    initialize_matrix(A, SIZE);
    initialize_matrix(B, SIZE);

    // Transpose matrix B and store the result in matrix C
    from_matrix_to_matrix(B, C, SIZE);

    //traditional matrix multiplication (vec) and measure the time
    before = seconds();
    vec(A, B, SIZE);
    after = seconds();

    // Sum all values in the matrix r
    DATA sum = 0;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            sum += r[i][j];
        }
    }

    // Print the time taken and the sum of values
    printf("Time:%f\t", after - before);
    printf("Result of summing:%f \n", sum);

    // Perform matrix multiplication using SSE instructions and measure the time
    before = seconds();
    vec_sse(A, C, SIZE);
    after = seconds();

    // Reset sum and calculate it again for the SSE version
    sum = 0;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            sum += r[i][j];
        }
    }

    // Print the time taken and the sum of values for the SSE version
    printf("Time:%f\t", after - before);
    printf("Result of summing:%f \n", sum);

    return 0;
}
