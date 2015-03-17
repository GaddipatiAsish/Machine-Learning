package com.perceptron.algorithms;

import java.util.Iterator;
import java.util.List;

import weka.core.matrix.Matrix;

import com.perceptron.martixOperations.MatrixOperations;

/**
 * TwoClassAvgPerceptron class implements the Averaged Perceptron algorithm that
 * classify the data set in two classes. The algorithm will output the averaged
 * weight vector vector after computation over the training data.
 * 
 * references: 1. http://www.ciml.info/dl/v0_8/ciml-v0_8-ch03.pdf
 * 
 * @author AsishKumar
 *
 */
public class TwoClassAvgPerceptron {
//	/*
//	Matrix computeW(Matrix W, int noOfEpochs, List<Matrix> features,
//			List<Double> T) {
//		Matrix wBar = W; /* load initial values as same as W to avgW */
//		int toeCounter = 1; /* counts the Iterations to average W */
//		int counter = 0; /* used to count the no of Epochs */
//		boolean flag = true; /* sets to False : If Model converge */
//		/* External loop on noOfEpochs */
//		do {
//			MatrixOperations matrices = new MatrixOperations();
//			/* Internal Loop for Xi=1 to N (training data set) */
//			Iterator<Matrix> iterator1 = features.iterator();
//			Iterator<Double> iterator2 = T.iterator();
//
//			while (iterator1.hasNext() && iterator2.hasNext()) {
//				/* compute system label */
//				Matrix featuresOfXi;
//				Double yi = matrices.multiply(W.transpose(),
//						(featuresOfXi = iterator1.next())).get(0, 0);
//				Double trueLabel;
//				/* compute step function */
//				Double sysLabel = sgn(yi);
//				/* check True Label not equals system label */
//				if (sysLabel != (trueLabel = iterator2.next())) {
//					/* update W */
//					W = W.plus(featuresOfXi.times(trueLabel));
//					flag = true;
//					/* update wBar */
//
//				} else {
//					flag = false;
//				}
//				wBar = wBar.plus(W);
//				toeCounter += 1;
//			}
//			++counter; /* Increase the Epoch count by 1 */
//		} while (noOfEpochs != counter || flag);
//
//		/* return average of W vector */
//		return wBar.times(1 / toeCounter);
//	}

	/**
	 * 
	 * @param W
	 *            : initial weight vector.
	 * @param noOfEpochs
	 *            : No of Epochs to avoid infinite Iterations
	 * @param features
	 *            : list of feature vectors for every Xi.
	 * @param T
	 *            : True labels List
	 * @return
	 */
	public Matrix computeW(Matrix W, int noOfEpochs, List<Matrix> features,
			List<Integer> T) {
		/*variables to compute avgW*/
		Matrix wBar = W; /* load initial values as same as W to compute avgW */
		int toeCounter = 0; /* counts the Iterations to average W */
		
		/* Variable used to check if the data is converged */
		int maxCount = 0;
		boolean isConverged = false;
		if (features.size() == T.size()) {
			maxCount = features.size();
		}

		MatrixOperations matrices = new MatrixOperations();
		int epochCounter = 0; /* used to count the no of Epochs */
		int dataPointsCounter; /* used for convergence check. */
		/* External loop on noOfEpochs */
		do {
			/* Internal Loop for Xi=1 to N */
			Iterator<Matrix> iterator1 = features.iterator();
			Iterator<Integer> iterator2 = T.iterator();
			System.out.println("started external while loop");
			dataPointsCounter = 0;
			while (iterator1.hasNext() && iterator2.hasNext()) {
				/* compute system label */
				System.out.println("started internal while loop");
				Matrix featuresOfXi;
				featuresOfXi = iterator1.next();
				Double yi = matrices.multiply(W.transpose(), featuresOfXi).get(
						0, 0);

				Integer trueLabel;
				/* compute step function */
				Integer sysLabel = sgn(yi);
				trueLabel = iterator2.next();
				/* check True Label not equals system label */
				if (sysLabel != trueLabel) {
					W = W.plus(featuresOfXi.times(trueLabel));
				} else {
					/* incremented if there is no change in W */
					dataPointsCounter++;
				}
				/*update wBar for complete data set*/
				wBar = wBar.plus(W);
				System.out.println(toeCounter++);
				System.out.println("Updated wBar");
				matrices.displayMatrix(wBar);

			}
			if (dataPointsCounter == maxCount) {/* convergence check */
				isConverged = true;
			}
			++epochCounter; /* Increase the Epoch count by 1 */
		} while (noOfEpochs != epochCounter && !isConverged);

		/* print the results of convergence */
		if (!isConverged) {
			System.out.println("Data set doesnt Converge!");
		} else {
			System.out.println("Data set Converges");
		}
		/* return average of W vector */
		System.out.println("Final Epoch Value "+ epochCounter);
		System.out.println("Final toe value "+toeCounter);
		return wBar.times(1 / toeCounter);
	}

	/**
	 * Step function that takes yi and classify it to one of the possible two
	 * classes.
	 * 
	 * @param yi
	 * @return system Label
	 */
	private Integer sgn(Double yi) {
		Integer sysLabel = 0;
		if (yi >= 0) {
			sysLabel = +1;
		} else {
			sysLabel = -1;
		}
		return sysLabel;
	}
}
