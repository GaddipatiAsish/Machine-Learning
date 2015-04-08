package com.features.test;

import java.io.IOException;


import weka.core.matrix.Matrix;

import com.features.filterMethods.PearsonCorrelationCoefficient;
import com.features.io.IOOperations;

public class Test {
	public static void main(String args[]) throws IOException {
		String trainDataFile = args[0];/* Training File */
		String trainLabelsFile = args[1];

		/* Get the training data matrix */
		IOOperations ioTrainFeatures = new IOOperations();
		Matrix trainData = ioTrainFeatures.readData(trainDataFile, 300, 20000);

		/* Get the training data labels */
		IOOperations ioTrainLabels = new IOOperations();
		Matrix trainlabels = ioTrainLabels.readLabels(trainLabelsFile, 300);

		/* computing Pearson Correlation Coefficient */
		PearsonCorrelationCoefficient pearson = new PearsonCorrelationCoefficient();
		pearson.compute(trainData, trainlabels);
	}
}
