#pragma once

#include <string>
#include <iostream>
#include <filesystem>
#include <fstream>
namespace fs = std::filesystem;
using namespace std;

/*
    Implimentation of class FolderReader
    This Class reads all files in a given path
    IMPORTANT: FolderReader only acts like -ls command and do not read the contents of the files
*/
class FolderReader {
private:
    // VARIABLES
    string              path;           // Path of the folder to read files from
    std::vector<string> files;          // All file names in the given path


public:
    /* 
        Constructor
        Input: path of the folder to read
    */
    FolderReader(string path) {
        this->path = path;
        readFolder();
    }


    // FUNCTIONS
    /*
    Reads all file names from the path and stores it in files vector
    */
    void readFolder() {
        this->files.clear();
        // Find all files in the directory using direcotry_iterator
        // WARNING: this works on C++17 and above
        for (const auto& entry : fs::directory_iterator(path))
            this->files.push_back(entry.path().string());
    }


    /*
        Gets the vector of file addresses
        Return: files vector
    */
    std::vector<string> getFiles() {
        return this->files;
    }

    int size() {
        return this->files.size();
    }

};


/*
    Implimentation of class FileReader
    This Class reads contents of the given file
*/
class FileReader {
private:
    // VARIABLES
    string          address;            // Address of the file to read
    vector<string>  lines;              // A vector including lines in the given file
    int             nrLines;            // Number of lines in the given file

    // FUNCTIONS
    void readLines() {
        // Open the file
        fstream newfile;
        newfile.open(this->address, ios::in);

        // Read line by line and store it in lines vector
        string temp;
        while (getline(newfile, temp)) {
            this->lines.push_back(temp);
            // Update the counter
            this->nrLines++;
        }

        // Close the file
        newfile.close();
    }
public:
    /*
        Constructor
        Input: Path of the file to read
    */
    FileReader(string address) {
        this->address = address;
        this->nrLines = 0;
        readLines();
    }

    /*
        Prints the line n
        Input: line index to print
        WARNING: No error handling, throws exception if line doesn't exist
    */
    void printLines(int n) {
        cout << lines[n] << endl;
    }

    /*
        Gets the line n
        Input: line index to get
        Return: line n contents
        WARNING: No error handling, throws exception if line doesn't exist
    */
    string getLines(int n) {
        return lines[n];
    }

    /*
        Gets number of lines in the file of given address in the class constructor
        Return: number of lines
    */
    int getNLines() {
        return this->nrLines;
    }
};


/*
    Implimentation of class MatrixParser
    This Class gets a string of integer numbers and converts it to a matrix
    ASSUMPTION: strings must contain only 1 digit numbers(0 - 9)
    ASSUMPTION: strings must contain square numbers of integers to build a square matrix
*/
class MatrixParser {
private:
    // VARIABLES
    int     size;           // size of matrix to build
public:
    /*
        Constructor
    */
    MatrixParser() {
        this->size = 0;
    }

    /*
        Gets a string of numbers
        Calculates the size of matrix (No error handling, files are assumed to obey the rules)
        Input: string of numbers
        Return: Pointer to a matrix built using the input
    */
    int* parseMatrix(string matrixLine) {
        // ASSUMPTION: Each line of files ends with the last number followed by \n
        int numbers = (matrixLine.length() + 1) / 2;
        this->size  = sqrt(numbers);
        return convert(matrixLine);
    }


    /*
        Converts a string to a matrix
        Input: string of numbers
        Return: Pointer to a matrix built using the input
    */
    int* convert(string matrixLine) {
        // Allocate memory
        int* matrix         = (int*)malloc(sizeof(int) * this->size * this->size);
        string delimiter    = " ";

        size_t pos = 0;
        int i = 0;
        while ((pos = matrixLine.find(delimiter)) != std::string::npos) {
            // Splits string using whitespaces and converts to int
            matrix[i++] = stoi(matrixLine.substr(0, pos));
            matrixLine.erase(0, pos + delimiter.length());
        }
        // Do it for the last element
        matrix[i] = stoi(matrixLine);
        return matrix;
    }

    /*
        Gets size of the matrix
        Returns 4 for a 4x4 matrix
        Returns: matrix size
    */
    int getSize() {
        return this->size;
    }
};


/*
    Implimentation of class Result
    This class stores determinants of a file
*/
class Result {
private:
    // VARIABLES
    double* determinants;    // Determinants
    bool isReady;           // True if all determinants are written
    int counter;            // How many determinants are added
    int expected;           // How many determinants are supposed to be added   
    string address;         // Where to write files
public:
    /*
        Constructor
        INPUT:  number of supposed determinants to write
                address of file to write into
    */
    Result(int n, string address) {
        this->determinants = (double*)malloc(sizeof(double) * n);
        int i;
        for (i = 0; i < n; i++) {
            this->determinants[i] = 69.696969;
        }
        this->counter = 0;
        this->expected = n;
        this->isReady = false;
        this->address = address.replace(address.find("_in"), sizeof("_in") - 1, "_out");
    }

    /*
        Default Constructor
        To be modified later
    */
    Result() {
        this->counter = 0;
        this->isReady = false;

    }

    /*
        Result Modifier
        Just like the first Constructor
    */
    void modifyResult(int n, string address) {
        this->determinants = (double*)malloc(sizeof(double) * n);
        int i;
        for (i = 0; i < n; i++) {
            this->determinants[i] = 69.696969;
        }
        this->expected = n;
        this->address = address.replace(address.find("_in"), sizeof("_in") - 1, "_out");

    }

    /*
        Adds determinants to a place
        INPUT:  index(line) of the matrix
                determinant of the matrix
        RETURN :    1 if file is written
                    0 otherwise
    */
    int addResult(int index, double value) {
        this->determinants[index] = value;
        this->counter++;
        // Automatic write
        if (this->counter == this->expected) {
            this->isReady = true;
            this->writeResult();
            return 1;
        }
        return 0;
    }
   
    /*
        Automatic write
        Writes the determinants vector to file
    */
    void writeResult() {
        ofstream MyFile(this->address);
        int i;
        for (i = 0; i < this->expected; i++) {
            MyFile << to_string(this->determinants[i]) << "\n";
        }
        MyFile.close();
    }

    /*
        Checks if all answers are written
    */
    bool checkReady() {
        return this->isReady;
    }
};