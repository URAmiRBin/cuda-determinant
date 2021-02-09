#pragma once
#include <stdlib.h>
#include <stdio.h>
#include"DataSet.h"
#include<vector>

/*
	Implementaion of class JobQueue
	This class stores a vector of Dataset(Matrix)
	And can handle simple queue operations
	NOTE: this was added because c++ default queue had some problems I don't remember
*/
class JobQueue {
private:
	// VARIABLES
	int first;					// First element of queue
	int last;					// Last element of queue
	std::vector<DataSet> jobs;	// Elements of queue
public:
	/*
		Constructor
		Sets pointers
	*/
	JobQueue() {
		this->first = 0;
		this->last = 0;
	}

	/*
		Checks if queue is empty
		Returns:	true if queue is empty
					false if it is not
	*/
	bool isEmpty() {
		return (this->first == this->last);
	}

	/*
		Adds a job
		INPUT: a dataset representing a job to calculate its' determinant
	*/
	void addJob(DataSet d) {
		this->jobs.push_back(d);
		this->last++;
	}

	/*
		Gets the first element waiting in the queue
		Return: A dataset
	*/
	DataSet getJob() {
		return this->jobs[this->first++];
	}
};