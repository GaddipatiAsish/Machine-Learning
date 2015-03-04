package com.linearRegression.core;

import java.util.List;

import com.linearRegression.martixOperations.MatrixOperations;

import weka.core.matrix.Matrix;

/**
 * WeightVector has methods that computes the W vector for Linear regression
 * with Regularization and with Regularization scenarios.
 * 
 * @author AsishKumar
 *
 */
public class WeightVector {
	/**
	 * @description getW method is used to get W in Linear Regression With out
	 *              Regularization case
	 * @param orderOfPolynomial
	 * @param xValues
	 * @param tValues
	 * @return W
	 */
	public Matrix getW(int orderOfPolynomial, List<Double> xValues,
			List<Double> tValues) { /* with out Regularization */

		/* Instantiate the Matrix Operations class */
		MatrixOperations matrixOperObject = new MatrixOperations();

		/* Compute Matrix Aij */
		Matrix A = matrixOperObject
				.computeMatrixAij(orderOfPolynomial, xValues);

		/* compute matrix Ti */
		Matrix T = matrixOperObject.computeMatrixTi(orderOfPolynomial, xValues,
				tValues);

		/* compute inverse of A matrix */
		Matrix inverseA = A.inverse();

		/* compute weights matrix W */
		Matrix W = matrixOperObject.multiply(inverseA, T);

		return W;
	}

	/**
	 * @description getW overridden method used in Linear Regression with
	 *              Regularization case
	 * @param i
	 *            : order of the polynomial
	 * @param xValues
	 * @param tValues
	 * @param lnLambda
	 * @return W
	 */
	public Matrix getW(int i, List<Double> xValues, List<Double> tValues,
			int lnLambda) { /* with Regularization */

		MatrixOperations matrices = new MatrixOperations();
		Matrix Fi = matrices.computeFi(xValues, 9);

		Matrix T = matrices.computeT(tValues);

		/* Expression: Temp = {[(Fi ^ T * Fi)] + [I*lambda*N]}^-1 */
		Matrix Temp = (matrices.multiply(Fi.transpose(), Fi)
				.plus(Matrix.identity(i + 1, i + 1).times(
						Math.exp(lnLambda) * xValues.size()))).inverse();
		
		/* Expression: W = [Temp * { (Fi) ^ T * T}] */
		Matrix W = matrices.multiply(Temp,
				matrices.multiply(Fi.transpose(), T));
		
		/*return the Weight Vector*/
		return W;
	}

}
