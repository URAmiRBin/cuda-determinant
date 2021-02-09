#define _CRT_SECURE_NO_WARNINGS

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include<iostream>
#include<chrono>
#include"fileReader.h"

typedef struct {
    int* A;
    int* indices;
    int det;
    float* U;
    int n;
} DataSet;


void LUdecomposition();
void fillDataSet(DataSet* dataSet);
void printDataSet(DataSet dataSet);
void closeDataSet(DataSet dataSet);
void sort(int col);
void mul(int row, int col, int index);

DataSet dataSet;


int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("[-] Invalid No. of arguments.\n");
        printf("[-] Try -> <n> \n");
        printf(">>> ");
        scanf("%d", &dataSet.n);
    }
    else {
        dataSet.n = atoi(argv[1]);
    }
    fillDataSet(&dataSet);

    FolderReader f = FolderReader("data_in/");
    std::vector<string> files = f.getFiles();
    int i;
    for (i = 0; i < files.size(); i++) {
        std::cout << files[i] << std::endl;
    }

    FileReader fileReader = FileReader(files[0]);
    fileReader.printLines(1);
    
    //auto t_start = std::chrono::high_resolution_clock::now();

    //LUdecomposition();

    //auto t_end = std::chrono::high_resolution_clock::now();

    //printDataSet(dataSet);
    //printf("%d\n", dataSet.det);
    //double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    //std::cout << "TIME: " << elapsed_time_ms << std::endl;
    closeDataSet(dataSet);
    system("PAUSE");
    return EXIT_SUCCESS;
}

void fillDataSet(DataSet* dataSet) {
    int i, j;

    dataSet->A = (int*)malloc(sizeof(int) * dataSet->n * dataSet->n);
    dataSet->U = (float*)malloc(sizeof(float) * dataSet->n * dataSet->n);
    dataSet->indices = (int*)malloc(sizeof(int) * dataSet->n);
    srand(time(NULL));

    for (i = 0; i < dataSet->n; i++) {
        for (j = 0; j < dataSet->n; j++) {
            dataSet->A[i * dataSet->n + j] = rand() % 10;
        }
    }

    for (i = 0; i < dataSet->n; i++) {
        for (j = 0; j < dataSet->n; j++) {
            dataSet->U[i * dataSet->n + j] = dataSet->A[i * dataSet->n + j];
        }
    }

    for (j = 0; j < dataSet->n; j++) {
        dataSet->indices[j] = j;
    }

    dataSet->det = 1;
}

void printDataSet(DataSet dataSet) {
    int i, j;

    printf("[-] Matrix A\n");
    for (i = 0; i < dataSet.n; i++) {
        for (j = 0; j < dataSet.n; j++) {
            printf("%-4d", dataSet.A[i * dataSet.n + j]);
        }
        putchar('\n');
    }

    //printf("[-] Matrix L\n");
    //for (i = 0; i < dataSet.n; i++) {
    //    for (j = 0; j < dataSet.n; j++) {
    //        printf("%-4f", dataSet.L[i * dataSet.n + j]);
    //    }
    //    putchar('\n');
    //}

    printf("[-] Matrix U\n");
    for (i = 0; i < dataSet.n; i++) {
        for (j = 0; j < dataSet.n; j++) {
            printf("%.2f    ", dataSet.U[i * dataSet.n + j]);
        }
        putchar('\n');
    }

}

void closeDataSet(DataSet dataSet) {
    free(dataSet.A);
    free(dataSet.U);
}



void swapInt(int* a, int* b)
{
    int t = *a;
    *a = *b;
    *b = t;
}

void swap(float* a, float* b)
{
    float t = *a;
    *a = *b;
    *b = t;
}

/* This function takes last element as pivot, places
the pivot element at its correct position in sorted
array, and places all smaller (smaller than pivot)
to left of pivot and all greater elements to right
of pivot */
int partition(float* arr, int low, int high)
{
    int pivot = arr[high]; // pivot  
    int i = (low - 1); // Index of smaller element  

    for (int j = low; j <= high - 1; j++)
    {
        // If current element is smaller than the pivot  
        if (arr[j] > pivot)
        {
            i++; // increment index of smaller element  
            swap(&arr[i], &arr[j]);
            swapInt(&dataSet.indices[i], &dataSet.indices[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    swapInt(&dataSet.indices[i + 1], &dataSet.indices[high]);
    return (i + 1);
}


void quickSort(float* arr, int low, int high)
{
    if (low < high)
    {
        /* pi is partitioning index, arr[p] is now
        at right place */
        int pi = partition(arr, low, high);

        // Separately sort elements before  
        // partition and after partition  
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

void mul(int row, int father, int index) {
    int i;
    float z = dataSet.U[row * dataSet.n + index] / dataSet.U[father * dataSet.n + index];
    //std::cout << "ZARIB : " << dataSet.U[row * dataSet.n + index] << " / " << dataSet.U[father * dataSet.n + index] << " = " << z << std::endl;
    for (i = 0; i < dataSet.n; i++) {
        dataSet.U[row * dataSet.n + i] -= z * dataSet.U[father * dataSet.n + i];
    }
}


void sort(int col) {
    float* arr = (float*)malloc(sizeof(float) * dataSet.n);
    int j;
    for (j = 0; j < dataSet.n; j++) {
        arr[j] = abs(dataSet.U[dataSet.indices[j] * dataSet.n + col]);
    }
    int max = arr[col];
    int maxI = col;
    for (j = col + 1; j < dataSet.n; j++) {
        if (arr[j] > max) {
            max = arr[j];
            maxI = j;
        }
    }
    swapInt(&dataSet.indices[col], &dataSet.indices[maxI]);
    //quickSort(arr, col, dataSet.n - 1);
    //for (j = 0; j < dataSet.n; j++) {
    //    std::cout << dataSet.indices[j] << std::endl;
    //}
    //std::cout << std::endl;
}

void LUdecomposition() {
    int i = 0, j = 0, k = 0;
    for (i = 0; i < dataSet.n - 1; i++) {
        sort(i);
        dataSet.det *= dataSet.U[dataSet.indices[i] * dataSet.n + i];
        std::cout << "det x " << dataSet.U[dataSet.indices[i] * dataSet.n + i] << std::endl;
        for (j = i + 1; j < dataSet.n; j++) {
            mul(dataSet.indices[j], dataSet.indices[i], i);
        }
    }

    dataSet.det *= dataSet.U[dataSet.indices[dataSet.n - 1] * dataSet.n + dataSet.n - 1];
    std::cout << "det x " << dataSet.U[dataSet.indices[dataSet.n - 1] * dataSet.n + dataSet.n - 1] << std::endl;

    //for (i = 0; i < dataSet.n; i++) {
    //    std::cout << dataSet.indices[i];
    //}
    //std::cout << std::endl;

    int c = 0;
    for (i = 0; i < dataSet.n; i++) {
        if (dataSet.indices[i] != i) {
            for (j = i; j < dataSet.n; j++) {
                if (dataSet.indices[j] == i) {
                    swapInt(&dataSet.indices[i], &dataSet.indices[j]);
                    c++;
                }
            }
        }
    }
    //for (i = 0; i < dataSet.n; i++) {
    //    std::cout << dataSet.indices[i];
    //}
    //std::cout << std::endl;
    //std::cout << c << std::endl;
    if (c % 2 == 1) {
        //printf("yes\n");
        dataSet.det *= -1;
    }
        

}