package com.features.filterMethods;

import java.util.HashMap;
import java.util.Map;

import weka.core.matrix.Matrix;

public class PearsonCorrelationCoefficient {

	public Map compute(Matrix trainData, Matrix trainlabels) {
		Map<Integer, Double> pearson = new HashMap<Integer, Double>();

		/* Computing the average of features */
		Matrix avgFeatures = new Matrix(1, trainData.getColumnDimension());
		for (int j = 0; j < trainData.getColumnDimension(); j++) {
			int sumup = 0;
			for (int k = 0; k < trainData.getRowDimension(); k++) {
				sumup += trainData.get(k, j);
			}
			double averageFi = sumup / (double) trainData.getRowDimension();
			avgFeatures.set(0, j, averageFi);
		}

		/* Computing the average of labels */
		int total = 0;
		for (int j = 0; j < trainlabels.getRowDimension(); j++) {
			total += trainlabels.get(j, 0);
		}
		double yBar = total / (double) trainlabels.getRowDimension();

		/* Computing the Pearson Correlation Map */
		for (int col = 0; col < trainData.getColumnDimension(); col++) {
			double numerator = 0;
			double c = 0;/* standard deviation of features */
			double d = 0;/* standard deviation of labels */
			for (int row = 0; row < trainData.getRowDimension(); row++) {
				double a = trainData.get(row, col) - avgFeatures.get(0, col);
				double b = trainlabels.get(row, 0) - yBar;
				numerator += (a * b);
				c += (a * a);
				d += (b * b);
			}
			double denominator = Math.sqrt(c) * Math.sqrt(d);

			pearson.put(col, (denominator > 0) ? (numerator / denominator) : 0);
		}
		System.out.println(pearson);
		return pearson;
	}
}
