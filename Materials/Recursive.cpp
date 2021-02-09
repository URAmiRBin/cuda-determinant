#define _CRT_SECURE_NO_WARNINGS

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

typedef struct {
	int* A;
	int n;
} DataSet;

void fillDataSet(DataSet* dataSet);
void printDataSet(DataSet dataSet);
void closeDataSet(DataSet dataSet);
int determinant(DataSet dataSet, int* indices, int size);
void add(DataSet dataSet);

int main(int argc, char* argv[]) {
	DataSet dataSet;
	if (argc < 3) {
		printf("[-] Invalid No. of arguments.\n");
		printf("[-] Try -> <n> \n");
		printf(">>> ");
		scanf("%d", &dataSet.n);
	}
	else {
		dataSet.n = atoi(argv[1]);
	}
	fillDataSet(&dataSet);
	int* indices = (int*)malloc(sizeof(int) * dataSet.n);
	int i = 0;
	for (i = 0; i < dataSet.n; i++) {
		indices[i] = i;
	}
	printDataSet(dataSet);
	double start = omp_get_wtime();
	int r = determinant(dataSet, indices, dataSet.n);
	double finish = omp_get_wtime();
	printf("%d\n", r);
	printf("TIME: %f seconds \n", finish - start);
	//add(dataSet);
	closeDataSet(dataSet);
	system("PAUSE");
	return EXIT_SUCCESS;
}

void fillDataSet(DataSet* dataSet) {
	int i, j;

	dataSet->A = (int*)malloc(sizeof(int) * dataSet->n * dataSet->n);
	srand(time(NULL));
	
	for (i = 0; i < dataSet->n; i++) {
		for (j = 0; j < dataSet->n; j++) {
			dataSet->A[i * dataSet->n + j] = rand() % 10;
		}
	}
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

}

void closeDataSet(DataSet dataSet) {
	free(dataSet.A);
}

int* make(int* in, int skip, int size, int t) {
	int* r = (int*)malloc(sizeof(int) * (size - 1));
	int i, j;
	//printf("size = %d\n", size);
	for (i = 0, j = 0; j < size; j++) {
		if (j != skip) {
			r[i] = in[j] + t;
			i++;
			//printf("Added %d with t = %d\n", in[i], t);
		}
		else {
			//printf("skipped %d\n", in[i]);
		}
	}
	for (i = 0; i < size - 1; i++) {
		//printf("%d	", r[i]);
	}
	//printf("\n\n");
	return r;
}

int determinant(DataSet dataSet, int* indices, int size) {
	int i;
	int result = 0;
	if (size == 2) {
		result = dataSet.A[indices[0]] * dataSet.A[indices[1] + dataSet.n] - dataSet.A[indices[1]] * dataSet.A[indices[0] + dataSet.n];
		return result;
	}
	else {
		for (i = 0; i < size; i++) {
			int* newIndices = make(indices, i, size, dataSet.n);
			if (i % 2 == 0) {
				result += dataSet.A[indices[i]] * determinant(dataSet, newIndices, size - 1);
			}
			else {
				result -= dataSet.A[indices[i]] * determinant(dataSet, newIndices, size - 1);
			}
			free(newIndices);
		}
	}
	return result;
}

void add(DataSet dataSet) {
	int i, j;
	for (i = 0; i < dataSet.n; i++) {
		for (j = 0; j < dataSet.n; j++) {
			dataSet.A[i * dataSet.n + j] = dataSet.A[i * dataSet.n + j] + dataSet.A[i * dataSet.n + j];
		}
	}
}