package com.features.filterMethods;

import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.features.sort.MapUtility;

import weka.core.matrix.Matrix;

public class PearsonCorrelationCoef {

	public List<Integer> compute(Matrix data, Matrix labels) {

		/* Computing the average of features */
		Matrix avgFeatures = new Matrix(1, data.getColumnDimension());
		for (int j = 0; j < data.getColumnDimension(); j++) {
			int sumup = 0;
			for (int k = 0; k < data.getRowDimension(); k++) {
				sumup += data.get(k, j);
			}
			double averageFi = sumup / (double) data.getRowDimension();
			avgFeatures.set(0, j, averageFi);
		}

		/* Computing the average of labels */
		int total = 0;
		for (int j = 0; j < labels.getRowDimension(); j++) {
			total += labels.get(j, 0);
		}
		double yBar = total / (double) labels.getRowDimension();

		/* Computing the Pearson Correlation Keys and corresponding values */

		List<Double> values = new LinkedList<Double>();
		List<Integer> keys = new LinkedList<Integer>();

		for (int col = 0; col < data.getColumnDimension(); col++) {
			double numerator = 0;
			double c = 0;/* standard deviation of features */
			double d = 0;/* standard deviation of labels */
			for (int row = 0; row < data.getRowDimension(); row++) {
				double a = data.get(row, col) - avgFeatures.get(0, col);
				double b = labels.get(row, 0) - yBar;
				numerator += (a * b);
				c += (a * a);
				d += (b * b);
			}
			double denominator = Math.sqrt(c) * Math.sqrt(d);

			keys.add(col);/* adding key */
			values.add((denominator > 0) ? (Math.abs(numerator) / denominator)
					: 0);/* add value */
		}

		int featureSize = keys.size();

		/* Start Ranking the keys based upon the values */
		List<Integer> rankedFeatures = new LinkedList<Integer>();
		List<Double> rankedFeatureVals = new LinkedList<Double>();

		for (int featureid = 0; featureid < featureSize; featureid++) {

			Double maxValue = Collections.max(values);
			rankedFeatureVals.add(maxValue);
			int position = values.indexOf(maxValue);

			int key = keys.get(position);
			rankedFeatures.add(key);

			values.remove(position);
			keys.remove(position);
		}

		return rankedFeatures;
	}
}
