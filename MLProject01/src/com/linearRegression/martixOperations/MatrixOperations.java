package com.linearRegression.martixOperations;

import java.util.Iterator;
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
	 * @description computeMatrixAij method compute the matrix Aij and returns
	 *              it.
	 * 
	 * @param m
	 *            is the order of the polynomial
	 * @param xValues
	 *            is the input data
	 * @return Aij matrix.
	 */
	public Matrix computeMatrixAij(int m, List<Double> xValues) {
		Matrix A = new Matrix(m + 1, m + 1);
		double aij;
		for (int i = 0; i < (m + 1); i++) {
			for (int j = i; j < (m + 1); j++) {
				aij = 0;
				Iterator<Double> iterator = xValues.iterator();
				while (iterator.hasNext()) { /*
											 * finding value aij to fill ith row
											 * and jth column
											 */
					aij += Math.pow(iterator.next(), i + j);
				}
				if (i != j) {
					A.set(i, j, aij);
					A.set(j, i, aij);
				} else {
					A.set(i, j, aij);
				}
			}
		}
		return A;
	}

	/**
	 * @description computeMatrixTi method computes the Ti.
	 * 
	 * @param m
	 *            is the order of the polynomial
	 * @param xValues
	 * @param tValues
	 * @return Ti Matrix
	 */
	public Matrix computeMatrixTi(int m, List<Double> xValues,
			List<Double> tValues) {
		Matrix Ti = new Matrix(m + 1, 1);
		double ti1;
		for (int i = 0; i < m + 1; i++) {
			ti1 = 0;
			Iterator<Double> iterator1 = xValues.iterator();
			Iterator<Double> iterator2 = tValues.iterator();
			while (iterator1.hasNext() && iterator2.hasNext()) {
				ti1 += iterator2.next() * Math.pow(iterator1.next(), i);
			}
			Ti.set(i, 0, ti1);
		}

		return Ti;
	}

	/**
	 * @description computeFi method computes the Fi Matrix
	 * @param xValues
	 * @param m
	 * @return Fi Matrix
	 */
	public Matrix computeFi(List<Double> xValues, int m) {
		/* Matrix Fi dimension is [N X (m+1)] */
		Matrix Fi = new Matrix(xValues.size(), m + 1);
		for (int i = 0; i < xValues.size(); i++) {
			for (int j = 0; j <= m; j++) {
				Fi.set(i, j, Math.pow(xValues.get(i), j));
			}
		}
		return Fi;
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
