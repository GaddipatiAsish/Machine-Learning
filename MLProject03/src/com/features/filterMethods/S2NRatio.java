package com.features.filterMethods;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.features.sort.MapUtility;

import weka.core.matrix.Matrix;

/**
 * Signal- to Noise Ratio
 * 
 * @author AsishKumar
 *
 */
public class S2NRatio {

	public Map compute(Matrix trainData, Matrix trainlabels) {
		Map<Integer, Double> s2nRatio = new HashMap<Integer, Double>();
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
		/* Compute signal to noise ratio */
	//	System.out.println("Positive mean");
	//	negativeMeans.transpose().print(1, 6);
		
		List<Integer> keys= new ArrayList<Integer>();
		List<Double> vals= new ArrayList<Double>();
		
		List<Integer> rankedkeys= new ArrayList<Integer>();
		List<Double> rankedvals= new ArrayList<Double>();
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
			double denominator = Math.sqrt(stdPositive/noOfPositiveSamples)
					+ Math.sqrt(stdNegative/noOfNegativeSamples);/* compute denominator */
			
//			keys.add(col);
//			vals.add((denominator > 0) ? (numerator / denominator) : 0);
//			
//			
			
			s2nRatio.put(col, (denominator > 0) ? (numerator / denominator) : 0);	
			
		}
////		System.out.println(keys);
////		System.out.println(vals);
//		for(int i=0; i < 20000; i++) {
//
//			// Get the max value position to find the max feature id
//
//			int maxValuePosition = vals.indexOf(Collections.max(vals));
//
//			// Fill the corresponding max feature Id for ranking
//
//			rankedkeys.add(keys.get(maxValuePosition));
//
//			// Fill the max value as well corresponding to its id
//
//			rankedvals.add(vals.get(maxValuePosition));
//
//			// Remove the max feature value & id to find the next max elements
//
//			vals.remove(maxValuePosition);
//
//			keys.remove(maxValuePosition);
//
//			}
//		
//		System.out.println(rankedkeys);
//		System.out.println(rankedvals);
//		
		
		return MapUtility.sortByValue(s2nRatio);
	}
}
