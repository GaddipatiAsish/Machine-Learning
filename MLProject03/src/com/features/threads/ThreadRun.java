package com.features.threads;

import java.util.concurrent.Callable;

import com.features.io.IOOperations;

import weka.core.matrix.Matrix;

public class ThreadRun implements Callable<Double> {
	int N;
	Matrix data;
	Matrix labels;
	String methodName;

	public ThreadRun(int N, Matrix data, Matrix labels, String methodName) {
		this.N = N;
		this.methodName = methodName;
		this.data = data;
		this.labels = labels;
	}

	public Double call() throws Exception {
		IOOperations io = new IOOperations();
		io.writeToFileKNN(data, labels, methodName);
		return 1.0;
	}

}
