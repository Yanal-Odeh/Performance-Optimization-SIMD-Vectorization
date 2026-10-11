//Yanal Odeh
//12113273

// Libraries
#include <immintrin.h>  // portable to all x86 compilers
#include <stdio.h>
#include <time.h>

#define DATA float

#define SIZE 128

DATA A[SIZE] __attribute__((aligned(16)));
DATA B[SIZE][SIZE] __attribute__((aligned(16)));
DATA r[SIZE];

double seconds() {
    // Function to get the current time in seconds
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return now.tv_sec + now.tv_nsec / 1000000000.0;
}

void initialize_array(DATA a[SIZE], int size) {
    // Function to initialize a 1D array with random values (0 or 1)
    for (int i = 0; i < size; i++) {
        a[i] = rand() % 2;
    }
}

void initialize_matrix(DATA a[SIZE][SIZE], int size) {
    // Function to initialize a 2D matrix with random values (0 or 1)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            a[i][j] = rand() % 2;
        }
    }
}

void vec(DATA s1[SIZE], DATA s2[SIZE][SIZE], int size) {
    // Vector multiplication without SIMD optimization
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            r[i] += s2[i][j] * s1[j];
        }
    }
}

void vec_sse(DATA s1[SIZE], DATA s2[SIZE][SIZE], int size) {
    // Vector multiplication with SIMD (SSE) optimization
    DATA prod = 0;
    __m128 X, Y, Z;

    for (int i = 0; i < size; i++) {
        r[i] = 0;
        prod = 0;
        Z[0] = Z[1] = Z[2] = Z[3] = 0;

        for (int j = 0; j < size; j += 4) {
            // Load 4 single-precision floating-point values from s1 and s2 into X and Y
            X = _mm_load_ps(&s1[j]);
            Y = _mm_load_ps(&s2[i][j]);

            // Multiply corresponding elements of X and Y and add the result to Z
            X = _mm_mul_ps(Y, X);
            Z = _mm_add_ps(X, Z);
        }

        // Sum the four values in Z and store the result in prod
        for (int n = 0; n < 4; n++) {
            prod += Z[n];
        }

        // Store the final result in r[i]
        r[i] = prod;
    }
}

int main() {
    double before, after;

    // Initialize input array and matrix with random values (0 or 1)
    initialize_array(A, SIZE);
    initialize_matrix(B, SIZE);

    // Measure time before and after the non-SIMD vector multiplication
    before = seconds();
    vec(A, B, SIZE);
    after = seconds();

    // Calculate and print the time taken and the result of summing all values in the vector
    DATA sum = 0;
    for (int i = 0; i < SIZE; i++) {
        sum += r[i];
    }
    printf("Time:%f\t", after - before);
    printf("Result of summing:%f \n", sum);

    // Measure time before and after the SIMD vector multiplication
    before = seconds();
    vec_sse(A, B, SIZE);
    after = seconds();

    // Calculate and print the time taken and the result of summing all values in the vector
    sum = 0;
    for (int i = 0; i < SIZE; i++) {
        sum += r[i];
    }
    printf("Time:%f\t", after - before);
    printf("Result of summing:%f \n", sum);

    return 0;
}

