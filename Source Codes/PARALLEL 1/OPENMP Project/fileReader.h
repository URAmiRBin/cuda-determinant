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

    // FUNCTIONS
    /*
        Reads all file names from the path and stores it in files vector
    */
    void readFolder() {
        // Find all files in the directory using direcotry_iterator
        // WARNING: this works on C++17 and above
        for (const auto& entry : fs::directory_iterator(path))
            this->files.push_back(entry.path().string());
    }
public:
    /* 
        Constructor
        Input: path of the folder to read
    */
    FolderReader(string path) {
        this->path = path;
        readFolder();
    }

    /*
        Gets the vector of file addresses
        Return: files vector
    */
    std::vector<string> getFiles() {
        return this->files;
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
    Implimentation of class FileWriter
    This Class writes a string to a given address
*/
class FileWriter {
private:
    // Variables
    string  address;    // Address to write in
    double*  contents;   // Answers of determinants of this file
    int n;              // Number of answers (matrices to be written in this file)
public:
    /*
        Constructor
        Input: Address to write in
    */
    FileWriter(string address) {
        this->address = address;
    }
    
    /*
        Constructor
        Input:  Address to write in
                number of lines to be written in files
        NEW : This is a modification of FileWriter that causes to answers be added dynamically
    */
    FileWriter(string address, int n) {
        this->address   = address;
        this->contents  = (double*)malloc(sizeof(double) * n);
        this->n         = n;
    }

    
    /*
        Adds value to contents with given index
        Input:  Index of given value
                Value to be written
        NEW : Add answers dynamically
    */
    void addContent(int index, double value) {
        this->contents[index] = value;
    }
    /*
        Writes given contents to address file given in constructor
        Input: contents
    */
    void writeToFile(string content) {
        // Open the file
        ofstream MyFile(this->address);

        // Write to the file
        MyFile << content;

        // Close the file
        MyFile.close();
    }


    /*
        Writes contents to adress of this file writer
        NEW : when called, it writes the contents to file, no need to pass a big string
    */
    void writeToFile() {
        ofstream MyFile(this->address);
        int i;
        for (i = 0; i < this->n; i++) {
            MyFile << to_string(this->contents[i]) << "\n";
        }
        MyFile.close();
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
