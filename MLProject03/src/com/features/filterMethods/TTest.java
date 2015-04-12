package com.features.filterMethods;

import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import com.features.sort.MapUtility;

import weka.core.matrix.Matrix;

public class TTest {

	public List<Integer> compute(Matrix trainData, Matrix trainlabels) {

		int noOfPositiveSamples = 0;
		int noOfNegativeSamples = 0;

		/* Compute the mean of positive and negative samples for each feature */
		Matrix positiveMeans = new Matrix(1, trainData.getColumnDimension());
		Matrix negativeMeans = new Matrix(1, trainData.getColumnDimension());
		for (int col = 0; col < trainData.getColumnDimension(); col++) {
			double meanPostive = 0;/* mean of positive samples */
			double meanNegative = 0;/* mean of negative samples */
			double numerator = 0;
			int count = 0;/* used by average */
			for (int row = 0; row < trainlabels.getRowDimension(); row++) {
				double sampleLabel = trainlabels.get(row, 0);
				if (sampleLabel > 0) {/* +ve samples */
					meanPostive += trainData.get(row, col);
					count++;
				} else if (sampleLabel < 0) {/*-ve samples*/
					meanNegative += trainData.get(row, col);
				}
			}

			noOfPositiveSamples = count;
			noOfNegativeSamples = trainData.getRowDimension() - count;

			meanPostive = meanPostive / (double) count;
			meanNegative = meanNegative
					/ (double) (trainData.getRowDimension() - count);

			positiveMeans.set(0, col, meanPostive);
			negativeMeans.set(0, col, meanNegative);
		}

		/* Compute the TTest Map */

		/* Original features */

		List<Double> values = new LinkedList<Double>();
		List<Integer> keys = new LinkedList<Integer>();

		for (int col = 0; col < trainData.getColumnDimension(); col++) {
			double stdPositive = 0;
			double stdNegative = 0;
			for (int row = 0; row < trainData.getRowDimension(); row++) {
				double sampleLabel = trainlabels.get(row, 0);
				if (sampleLabel > 0) {/* +ve samples */
					stdPositive += Math.pow(trainData.get(row, col)
							- positiveMeans.get(0, col), 2);
				} else if (sampleLabel < 0) {/*-ve samples*/
					stdNegative += Math.pow(trainData.get(row, col)
							- negativeMeans.get(0, col), 2);
				}
			}

			double numerator = Math.abs(positiveMeans.get(0, col)
					- negativeMeans.get(0, col));/* compute numerator */
			double stdP = stdPositive
					/ (double) Math.pow(noOfPositiveSamples, 2);
			double stdN = stdNegative
					/ (double) Math.pow(noOfNegativeSamples, 2);

			double denominator = Math.sqrt(stdP + stdN);/* compute denominator */

			keys.add(col);
			values.add((denominator > 0) ? (numerator / denominator) : 0);

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
