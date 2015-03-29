package com.perceptron.algorithms;

import java.util.Iterator;
import java.util.List;

import weka.core.matrix.Matrix;

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
public class AvgPerceptron {
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
		/* variables to compute avgW */
		Matrix wBar = W; /* load initial values as same as W to compute avgW */
		int toeCounter = 1; /* counts the Iterations to average W */

		/* Variable used to check if the data is converged */
		int maxCount = 0;
		boolean isConverged = false;
		if (features.size() == T.size()) {
			maxCount = features.size();
		}

		
		int epochCounter = 0; /* used to count the no of Epochs */
		int dataPointsCounter; /* used for convergence check. */
		/* External loop on noOfEpochs */
		do {
			/* Internal Loop for Xi=1 to N */
			Iterator<Matrix> iterator1 = features.iterator();
			Iterator<Integer> iterator2 = T.iterator();

			dataPointsCounter = 0;
			while (iterator1.hasNext() && iterator2.hasNext()) {
				/* compute system label */
				
				Matrix featuresOfXi;
				featuresOfXi = iterator1.next();
				Double yi = W.transpose().times(featuresOfXi).get(
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
				/* update wBar for complete data set */
				wBar = wBar.plus(W);
				toeCounter++;

			}
			if (dataPointsCounter == maxCount) {/* convergence check */
				isConverged = true;
			}
			++epochCounter; /* Increase the Epoch count by 1 */
		} while (noOfEpochs != epochCounter && !isConverged);

		Matrix avgW= wBar.times(1/(double)toeCounter);
		avgW= avgW.times(1/Math.sqrt(avgW.transpose().times(avgW).get(0, 0)));
		return avgW;
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

	public double discriminantFn(Matrix featureOfXi, Matrix wMatrix) {
		double yi = 0.0;
		
		yi = wMatrix.transpose().times(featureOfXi).get(0, 0);

		return yi;
	}
}
