package com.perceptron.martixOperations;

import java.util.List;

import weka.core.matrix.Matrix;

/**
 * MatrixOperations class projects on doing the operations that are involved
 * with matrices like matrix multiplication or it could be populating a new
 * matrix upon custom definitions.
 * 
 * @author AsishKumar
 *
 */
public class MatrixOperations {
	/**
	 * @description multiplyMatrices method multiply two matrices and returns
	 *              the result
	 * 
	 * @param matrix1
	 * @param matrix2
	 * @return result of two Matrices
	 */
	public Matrix multiply(Matrix matrix1, Matrix matrix2) {
		/* Matrix Multiplication Rule check */

		/* Multiplication not possible case */
		if (matrix1.getColumnDimension() != matrix2.getRowDimension()) {
			System.exit(-1);
		}

		/* Multiplication possible case */
		Matrix outMatrix = new Matrix(matrix1.getRowDimension(),
				matrix2.getColumnDimension());
		for (int i = 0; i < matrix1.getRowDimension(); i++) {
			for (int j = 0; j < matrix2.getColumnDimension(); j++) {
				double cij = 0;
				for (int k = 0; k < matrix1.getColumnDimension(); k++) {
					cij += matrix1.get(i, k) * matrix2.get(k, j);
				}
				outMatrix.set(i, j, cij);
			}
		}
		return outMatrix;
	}

	/**
	 * @description computeT method is used to find target Matrix
	 * @param tValues
	 * @return T Matrix
	 */
	public Matrix computeT(List<Double> tValues) {
		Matrix T = new Matrix(tValues.size(), 1); /* column matrix */
		for (int i = 0; i < tValues.size(); i++) {
			T.set(i, 0, tValues.get(i));
		}
		return T;
	}

	/**
	 * @description displayMatrix method takes matrix as input and display the
	 *              contents of it to console.
	 * 
	 * @param matrix
	 */
	public void displayMatrix(Matrix matrix) {
		for (int row = 0; row < matrix.getRowDimension(); row++) {
			System.out.print("[");
			for (int column = 0; column < matrix.getColumnDimension(); column++) {
				System.out.print(matrix.get(row, column) + ",");
			}
			System.out.println("]");
		}
	}

}
