package com.features.io;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import weka.core.matrix.Matrix;

public class GenerateInputs {
	public Matrix generateSVMInputs(Matrix data, Matrix labels,
			int NoOfTopFeatures, Map<Integer, Double> map, boolean normalise) {
		List<Integer> rankedKeys = new ArrayList<Integer>();
		Set<Integer> keys = map.keySet();
		
		System.out.println(keys);
		
		rankedKeys.addAll(keys);
		System.out.println(rankedKeys);
		//System.out.println(rankedKeys);
		Matrix newFeatures = new Matrix(data.getRowDimension(),
				NoOfTopFeatures, 0);
		for (int row = 0; row < data.getRowDimension(); row++) {
			/* loop through the top ranked features */
			for (int i = 0; i < NoOfTopFeatures; i++) {
				newFeatures.set(row, i, data.get(row, rankedKeys.get(i)));
			}
			if (normalise) {/* Normalize */
				Matrix line = newFeatures.getMatrix(row, row, 0,
						NoOfTopFeatures - 1);
				double norm = line.normF();
				line = line.times((norm > 0) ? (1 / norm) : 0);
				newFeatures.setMatrix(row, row, 0, NoOfTopFeatures - 1, line);
			}
		}

		return newFeatures;
	}

}
