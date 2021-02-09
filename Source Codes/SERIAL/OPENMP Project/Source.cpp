#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include<iostream>
#include<chrono>
#include"fileReader.h"
#include"DataSet.h"

// FUNCTION IDs
double   determinantLU(DataSet* dataset);
void    columnElimination(DataSet* dataset, int index, int* indices);
int*    getDefaultIndices(int n);
int     swapMax(int col, DataSet* dataset, int* indices);
void    swapRows(int* a, int* b);

int main() {
    int i, j;                   // Iterator variables
    std::vector<string> files;  // Vector of file names in directory data_in/

    // Start counting Time
    auto t_start = std::chrono::high_resolution_clock::now();

    // Get all the files in data_in/ directory
    FolderReader f = FolderReader("data_in/");
    files = f.getFiles();


    // Do this for all files in the directory data_in/
    for (i = 0; i < files.size(); i++) {
        // LOCAL VARIABLES FOR EACH FILE
        string result           = "";           // The determinants of matrices in this files are stored here line by line
        string outputAddress    = files[i];     // The output address to write the final result in it


        // Read this file from data_in/filename.txt
        FileReader fileReader = FileReader(files[i]);
        
        // Write output in data_out/filename.txt
        FileWriter fileWriter = FileWriter(outputAddress.replace(outputAddress.find("_in"), sizeof("_in") - 1, "_out"));

        // Do this for each matrix in this file
        for (j = 0; j < fileReader.getNLines(); j++) {
            double  det;        // Determinant of this matrix
            int*    m;          // This matrix
            // Build a matrix Parser
            MatrixParser matrixParser = MatrixParser();
            
            // Parse string of numbers to matrix
            m = matrixParser.parseMatrix(fileReader.getLines(j));

            // Build a dataset from m
            DataSet dataset = DataSet(matrixParser.getSize());
            dataset.fillDataSet(m);
            free(m);
            
            // Calculate determinant using customized LU decomposition
            det = determinantLU(&dataset);
            
            // Store det to result string
            result += to_string(det) + "\n";
        }

        // write result string for this file
        fileWriter.writeToFile(result);
    }

    // Finish counting time
    auto t_end = std::chrono::high_resolution_clock::now();
    
    // Print time
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "Elapsed time: " << elapsed_time_ms << "ms" << std::endl;
    //system("PAUSE");
    return EXIT_SUCCESS;
}

/*
    Swaps biggest absolute value of the given column to location col(excluding first col values)
    Input:
        col:        the column we're working on
        dataset:    the matrix we're working on
        indices:    the indices
    Return: -1 if there was a swap and 1 if there wasn't
    NOTE: there's no real swapping, we swap virtually using the indices array
    EXAMPLE: if we swap row 0 and row 1 of a 4x4 matrix the indices array changes from [0, 1, 2, 3] to [1, 0, 2, 3]
*/
int swapMax(int col, DataSet* dataset, int* indices) {
    int j;

    // A temp array to find the max index
    float* arr = (float*)malloc(sizeof(float) * dataset->getN());

    // Fill the temp array with contents of column
    for (j = 0; j < dataset->getN(); j++) {
        arr[j] = abs(dataset->getU(indices[j], col));
    }

    // Find the max(excluding first col values)
    int max = arr[col];
    int maxI = col;
    for (j = col + 1; j < dataset->getN(); j++) {
        if (arr[j] > max) {
            max = arr[j];
            maxI = j;
        }
    }

    // Swap two rows virtually
    if (maxI != col) {
        swapRows(&indices[col], &indices[maxI]);
        return -1;
    }
    return 1;
}


/*
    Using Guassian Elimination, eliminates all rows in the matrix excluding the pivot row
    input:
        dataset:    the matrix
        index:      the column to eliminate
        indices:    current status of rows positions
*/
void columnElimination(DataSet* dataset, int index, int* indices) {
    int i, j;       // Iterators
    double z;        // The coefficient
    // Do this for each row after the pivot row
    for (j = index + 1; j < dataset->getN(); j++) {
        // Find coefficient of the row
        z = dataset->getU(indices[j], index) / dataset->getU(indices[index], index);
        // Do this for all elements in the row except the ones that are 0
        for (i = index; i < dataset->getN(); i++) {
            // Subtract z * pivot row from this element
            dataset->subtractU(indices[j], i, z * dataset->getU(indices[index], i));
        }
    }
}

/*
    Finds the determinant using a customized LU factorization
    Proof:
        PA = LU
        det(PA) = det(LU)
        det(P) x det(A) = det(L) x det(U)
        det(P) = 1 x (-1)^swaps
        det(L) = 1
        so => det(A) = det(P) x det(U)
        so we just need to calculate U using guassian elimination and partial pivoting
        and then find the number of swaps we made
    Input: dataset Matrix
    Return: determinant of given dataset
*/
double determinantLU(DataSet* dataset) {
    int i, j;       // Iterators
    double det = 1;  // Determinant to be calculated
    int* indices;   // Virtual holder of row positions

    // Load Default indices : [0, 1, ... , n]
    indices = getDefaultIndices(dataset->getN());
    
    // Do for each column
    for (i = 0; i < dataset->getN() - 1; i++) {
        // Find max absolute number of this column and call it pivot
        // Negative the result if a swapping happened
        det *= swapMax(i, dataset, indices);
        // Multiply result by the pivot element
        det *= dataset->getU(indices[i], i);
        // Do the Guassian Elimination for this column
        columnElimination(dataset, i, indices);
    }

    // Multiply result by last pivot element
    det *= dataset->getU(indices[dataset->getN() - 1], dataset->getN() - 1);
    return det;
}

/*
    Swaps two pointers to ints
    This function is used to swap rows of matrix using indices array
    Input: two pointers
*/
void swapRows(int* a, int* b)
{
    int t = *a;
    *a = *b;
    *b = t;
}

/*
    Builds default indices of a matrix
    Input: size of matrix
    Return: the indices array
*/
int* getDefaultIndices(int n) {
    int i;

    int* indices = (int*)malloc(sizeof(int) * n);

    for (i = 0; i < n; i++) {
        indices[i] = i;
    }

    return indices;
}