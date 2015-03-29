package com.perceptron.threads;

import java.util.List;
import java.util.concurrent.Callable;

import com.perceptron.algorithms.Perceptron;
import com.perceptron.io.IOOperations;

import weka.core.matrix.Matrix;

public class EpochThreadRun implements Callable<Double> {
	int epoch;/* No of Epochs Linear Perceptron to run */
	List<Matrix> featureMatrixTrainData; /* features Matrix: Training Data */
	List<Integer> trueLabelsTrainData;/* true labels : Training data */
	List<Matrix> featureMatrixDevData; /* features Matrix : Development data */
	List<Integer> trueLabelsDevData;/* true labels : Development data */

	public EpochThreadRun(int epoch) {
		this.epoch = epoch;
	}

	/**
	 * call method hold the code that needs to be executed in a thread
	 */
	public Double call() throws Exception {
		Double accuracy = 0.0;
		int favorable = 0;
		int count = 0;
		for (int i = 0; i <= 9; i++) {
	
			/* Get the Training and development data */
			IOOperations ioTrainData = new IOOperations();
			ioTrainData.readInputData("./inputData/" + i + ".tra",true);
			this.featureMatrixTrainData = ioTrainData.getFeatureMatrix();
			this.trueLabelsTrainData = ioTrainData.getTrueLabels();

			IOOperations ioDevdata = new IOOperations();
			ioDevdata.readInputData("./inputData/" + i + ".dev",true);
			this.featureMatrixDevData = ioDevdata.getFeatureMatrix();
			this.trueLabelsDevData = ioDevdata.getTrueLabels();

			/* Set W=[0,0....0] initially */
			/* n denotes the data size N */
			int n = featureMatrixTrainData.get(0).getRowDimension();
			// System.out.println("n is " + n);
			/* Initialization of Weight Vector */
			Matrix W = new Matrix(n, 1);
			for (int j = 0; j < n; j++) {
				W.set(j, 0, 0); /* W=[0,0,0.....] */
			}

			/* learn normalized W using Training data using Linear Perceptron */
			Perceptron perceptron = new Perceptron();
			W = perceptron.computeW(W, epoch, featureMatrixTrainData,
					trueLabelsTrainData);
//			/* Normalize the Weight vector */
//			W = W.times(1 / Math.sqrt(W.transpose().times(W).get(0, 0)));
			int arr[] = perceptron.devRun(W, featureMatrixDevData,
					trueLabelsDevData);
			favorable += arr[1];
			count += arr[0];

		}
		accuracy = favorable / (new Double(count)) *100;
		return accuracy;
	}

}
