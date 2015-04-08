package com.features.test;

import java.io.IOException;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import weka.core.matrix.Matrix;

import com.features.filterMethods.PearsonCorrelationCoef;
import com.features.filterMethods.S2NRatio;
import com.features.filterMethods.TTest;
import com.features.io.IOOperations;

public class Test {
	/* Ranking the features */


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
		PearsonCorrelationCoef pearson = new PearsonCorrelationCoef();
		Map pearsonMap = pearson.compute(trainData, trainlabels);
		System.out.println(pearsonMap);

		/* compute S2N Ratio */
//		S2NRatio s2nRatio = new S2NRatio();
//		Map s2nMap = s2nRatio.compute(trainData, trainlabels);
//		System.out.println(s2nMap);

		/* Compute TTest and ranking the features*/
//		TTest ttest = new TTest();
//		Map<Integer, Double> ttestMap = ttest.compute(trainData, trainlabels);
//		System.out.println(ttestMap);
	}
}
