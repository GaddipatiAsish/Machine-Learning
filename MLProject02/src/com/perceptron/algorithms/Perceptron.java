package com.perceptron.algorithms;

import java.util.Iterator;
import java.util.List;


import weka.core.matrix.Matrix;

/**
 * TwoClassPerceptron class implements the Perceptron algorithm that classify
 * the data set in two classes. The algorithm will output the weight vector W
 * after computation over the training data.
 * 
 * references: 1. http://www.ciml.info/dl/v0_8/ciml-v0_8-ch03.pdf
 * 
 * @author AsishKumar
 *
 */
public class Perceptron {
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
				Double yi = W.transpose().times(featuresOfXi).get(0, 0);

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

		W=W.times(1/Math.sqrt(W.transpose().times(W).get(0, 0)));
		return W;
		
	}

	/**
	 * devRun method uses uses development data to count the favorable
	 * classification which can be used to calculate accuracy.
	 * 
	 * @param W
	 *            learnt weight vector
	 * @param features
	 *            features of each Xi of all development data
	 * @param trueLabels
	 *            true labels of the corresponding Xi of development data
	 * @return favorable classifications count.
	 */
	public int[] devRun(Matrix W, List<Matrix> features,
			List<Integer> trueLabels) {
		/*
		 * favorable[0] takes total count favorable[1] takes favorable events
		 * count
		 */
		int[] favorable = { 0, 0 };
		Iterator<Matrix> iterator = features.iterator();
		Iterator<Integer> iterator2 = trueLabels.iterator();
		while (iterator.hasNext() && iterator2.hasNext()) {
			Integer trueLabelOfXi = iterator2.next();
			Matrix featuresOfXi = iterator.next();
			if (trueLabelOfXi == 1) {
				favorable[0]++;
				/* Compute discriminant function */
				Double yi = W.transpose().times(featuresOfXi).get(0, 0);
				if (sgn(yi) > 0) { 
					favorable[1]++;
				}
			}
		}
		return favorable;
	}

	public double discriminantFn(Matrix featureOfXi, Matrix wMatrix) {
		double yi = 0.0;
		
		yi = wMatrix.transpose().times(featureOfXi).get(0, 0);

		return yi;
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
