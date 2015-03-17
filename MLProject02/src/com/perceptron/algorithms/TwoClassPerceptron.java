package com.perceptron.algorithms;

import java.util.Iterator;
import java.util.List;

import com.perceptron.martixOperations.MatrixOperations;

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
public class TwoClassPerceptron {
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

		MatrixOperations matrices = new MatrixOperations();
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
		//matrices.displayMatrix(W);
		return W;
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
