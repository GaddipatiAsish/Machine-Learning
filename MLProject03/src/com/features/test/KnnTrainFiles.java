package com.features.test;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.ExecutorService;

import weka.core.matrix.Matrix;

import com.features.io.IOOperations;

public class KnnTrainFiles {
	public static void main(String args[]) throws Exception {
		int NValue[] = { 1, 5, 10, 20, 50, 100, 200, 300, 400, 500, 600, 700,
				800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
				10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000,
				19000, 20000 };

		String trainDataFile = args[0];/* Training File */
		String trainLabelsFile = args[1];/* Training Labels File */

		/* Get the training data matrix */
		IOOperations ioTrainFeatures = new IOOperations();
		Matrix trainData = ioTrainFeatures.readData(trainDataFile, 300, 20000);

		/* Get the training data labels */
		IOOperations ioTrainLabels = new IOOperations();
		Matrix trainlabels = ioTrainLabels.readLabels(trainLabelsFile, 300);

		/* Read the Pearson ranks from the file */
		IOOperations iop = new IOOperations();
		List<Integer> pRankedFeatureid = iop.readRanksFromFile("Pearson");

		/* Generate the KNN Training Files using Multi Threading */
		for (int i = 0; i < NValue.length; i++) {
			ExecutorService service;
		}

	}
}
