package com.linearRegression.core;

import java.util.List;

import com.linearRegression.martixOperations.MatrixOperations;

import weka.core.matrix.Matrix;
/**
 * ERMS class has methods that compute the ERMS values for a given Weight Vector and lnLambda for
 * both Linear Regression WITH Regularization and WITH OUT Regularization scenarios.
 * @author AsishKumar
 *
 */
public class ERMS {
	/**
	 * @description compute method computes ERMS values for given W in With out
	 *              Regularization case
	 * @param W
	 * @param xValues
	 * @param tValues
	 * @return ERMS
	 */
	public double compute(Matrix W, List<Double> xValues, List<Double> tValues) { /*
																				 * With
																				 * Out
																				 * regularization
																				 */
		MatrixOperations matrices = new MatrixOperations();
		/* Steps for computing Y=W^T * X */

		/* compute matrix X */
		Matrix X = new Matrix(W.getRowDimension(), xValues.size());
		for (int col = 0; col < X.getColumnDimension(); col++) {
			for (int row = 0; row < X.getRowDimension(); row++) {
				X.set(row, col, Math.pow(xValues.get(col), row));
			}
		}

		/* compute Y */
		Matrix Y = matrices.multiply(W.transpose(), X);

		/* compute ErrW */
		double ErrW = 0.0;
		for (int i = 0; i < tValues.size(); i++) {
			ErrW += Math.pow(Y.get(0, i) - tValues.get(i), 2);
		}
		ErrW = ErrW / xValues.size();
		
		/*return ERMS*/
		return Math.sqrt(ErrW / xValues.size());
	}

	/**
	 * @description compute is an overloaded method used to calculate ERMS values for
	 *              a given lnLambda in Linear Regression with Regularization
	 *              case
	 * @param W
	 * @param xValues
	 * @param tValues
	 * @param lnLambda
	 * @return ERMS
	 */
	public Double compute(Matrix W, List<Double> xValues, List<Double> tValues,
			int lnLambda) {/* with Regularization */

		MatrixOperations matrices = new MatrixOperations();
		/* Steps for computing Y=W^T * X */

		/* compute matrix X */
		Matrix X = new Matrix(W.getRowDimension(), xValues.size());
		for (int col = 0; col < X.getColumnDimension(); col++) {
			for (int row = 0; row < X.getRowDimension(); row++) {
				X.set(row, col, Math.pow(xValues.get(col), row));
			}
		}

		/* compute Y */
		Matrix Y = matrices.multiply(W.transpose(), X);

		/* Compute wNorm */
		double wNorm = matrices.multiply(W.transpose(), W).get(
				0, 0);

		/* calculate lambda from lnLambda */
		double lambda = Math.exp(lnLambda);

		/* compute ErrW */
		double ErrW = 0.0;
		for (int i = 0; i < tValues.size(); i++) {
			ErrW += Math.pow(Y.get(0, i) - tValues.get(i), 2);
		}
		
		ErrW = ErrW / xValues.size();
		
		/*return ERMS value*/
		return Math.sqrt((ErrW + (lambda * wNorm)) / xValues.size());
	}
}
