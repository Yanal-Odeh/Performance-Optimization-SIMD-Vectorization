//Yanal Odeh
//12113273

#include <immintrin.h>  // portable to all x86 compilers
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DATA float

#define SIZE 128

DATA* A;
DATA* B;

double seconds()
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return now.tv_sec + now.tv_nsec / 1000000000.0;
}

void initialize_array(DATA a[], int size)
{
    for (int i = 0 ;  i < size ; i++)
    {
        a[i] = rand()%2;
    }
}

DATA vec(DATA *s1, DATA *s2, int size) 
{
    DATA prod = 0;
    for (int i = 0; i < size; i++) {
        prod += s1[i] * s2[i];
    }
    return prod;
}

DATA vec_sse(DATA *m1, DATA *m2, int size) 
{
    DATA prod = 0;
    __m128 X, Y, Z;
    Z[0] = Z[1] = Z[2] = Z[3] = 0;

    for (int i = 0; i < size; i += 4) 
    {
        X = _mm_load_ps(&m1[i]);
        Y = _mm_load_ps(&m2[i]);
        X = _mm_mul_ps(X, Y);
        Z = _mm_add_ps(X, Z);
    }

    for (int i = 0; i < 4; i++) 
    {
        prod += Z[i];
    }

    return prod;
}

int main()
{
    DATA r;
    double before, after;

    // Dynamically allocate memory for arrays A and B
    A = (DATA*)malloc(SIZE * sizeof(DATA));
    B = (DATA*)malloc(SIZE * sizeof(DATA));

    // Check if memory allocation was successful
    if (A == NULL || B == NULL) {
        printf("Memory allocation failed.\n");
        return 1;
    }

    initialize_array(A, SIZE);
    initialize_array(B, SIZE);

    before = seconds();
    r = vec(A, B, SIZE);
    after = seconds();
    printf("Result:%f Time:%f\n", r, after - before);

    before = seconds();
    r = vec_sse(A, B, SIZE);
    after = seconds();
    printf("Result:%f Time:%f\n", r, after - before);

    // Free dynamically allocated memory
    free(A);
    free(B);

    return 0;
}

