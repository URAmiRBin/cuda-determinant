#pragma once

#include <stdlib.h>
#include <stdio.h>
#include<string>

/*
	Implimentation of class DataSet
	This Class contains the matrix and properties of it
	The matrix is square
*/
class DataSet {
private:
	// VARIABLES
	int*	a;			// The matrix
	float*	u;			// The U(pper triangular) Matrix
	int		n;			// Dimension of matrix
public:
	/*
		Constructor
		Allocates memory for matrices
		Input: size of matrix
	*/
	DataSet(int n) {
		this->n = n;
		this->a = (int*)malloc(sizeof(int) * this->n * this->n);
		this->u = (float*)malloc(sizeof(float) * this->n * this->n);
	}

	/*
		Frees allocated memory
		This usually is done in destructor but the regular function just gives more control
	*/
	void freeDataSet() {
		free(this->u);
		free(this->a);
	}

	/*
		Prints matrix A
	*/
	void printDataSet() {
        int i, j;

        printf("Matrix A\n");
        for (i = 0; i < this->n; i++) {
            for (j = 0; j < this->n; j++) {
                printf("%-4d", this->a[i * this->n + j]);
            }
            putchar('\n');
        }
	}

	/*
		Fills matrix A and U (initially it's like A)
		Input: A pointer to a matrix with the same size
	*/
	void fillDataSet(int* matrixStr) {
        int i, j;
		for (i = 0; i < this->n; i++) {
            for (j = 0; j < this->n; j++) {
                this->a[i * this->n + j] = matrixStr[i * this->n + j];
                this->u[i * this->n + j] = this->a[i * this->n + j];
            }
        }

	}

	/*
		Gets dimenstion of the matrix
		Return: dimension of the matrix
	*/
	int getN() {
		return this->n;
	}

	/*
		Gets an element of matrix U
		Input: Row and column index of matrix U
		Return: The element of matrix U
	*/
	float getU(int i, int j) {
		return this->u[i * this->n + j];
	}

	/*
		Subtracts given value from element of given index of matrix U
		Input: Row and column index of matrix U and the value to subtract
	*/
	void subtractU(int i, int j, float value) {
		this->u[i * this->n + j] -= value;
	}
};