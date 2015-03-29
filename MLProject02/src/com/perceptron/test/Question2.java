package com.perceptron.test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import weka.core.matrix.Matrix;

import com.perceptron.algorithms.AvgKernelPerceptron;
import com.perceptron.algorithms.AvgPerceptron;
import com.perceptron.algorithms.KernelPerceptron;
import com.perceptron.algorithms.Perceptron;
import com.perceptron.io.IOOperations;

public class Question2 {

	public static void main(String args[]) throws IOException {

		/* Read the data set for Question 2 */
//		boolean lPercepFlag=true; /* True: to add 1 to features of Xi*/
		boolean lPercepFlag=false; /*to run with Polynomial kernel Perceptron */
		IOOperations io = new IOOperations();
		io.readInputData(args[0],lPercepFlag);

		/* Get target and features vector */
		List<List<Double>> feat = io.getFeatureVector();
		List<Integer> trueLabels = io.getTrueLabels();

		
		/* n denotes the data size N */
		int noOfFeatures = feat.get(0).size();
		/* Initialization of Weight Vector */
		Matrix W = new Matrix(noOfFeatures, 1);
		for (int i = 0; i < noOfFeatures; i++) {
			W.set(i, 0, 0); /* W=[0,0,0.....] */
		}
		/* Set the alfa matrix to 0*/
		Matrix alfa = new Matrix(trueLabels.size(),1);
		for (int i = 0; i < trueLabels.size(); i++) {
			alfa.set(i, 0, 0); /* W=[0,0,0.....] */
		}
		
		List<Matrix> features = new ArrayList<Matrix>();
		Iterator<List<Double>> iterator1 = feat.iterator();
		while (iterator1.hasNext()) {
			List<Double> temp = iterator1.next();
			Matrix featuresOfXi = new Matrix(noOfFeatures, 1);
			for (int i = 0; i < temp.size(); i++) {
				featuresOfXi.set(i, 0, temp.get(i));
			}
			features.add(featuresOfXi);
		}
	/*UN COMMENT THIS FOR LINEAR PERCEPTRON*/
		
//		Perceptron linearPerceptron= new Perceptron();
//		W=linearPerceptron.computeW(W, 10, features, trueLabels);
//		System.out.println("W Vector");
//		W.transpose().print(3, 2);
		
		KernelPerceptron perceptron = new KernelPerceptron();
		alfa=perceptron.computeAlfaMatrix(alfa, 3, features, trueLabels, 'b', 2);
		System.out.println("ALFA Matrix");
		alfa.transpose().print(1, 0);
		

//		// TwoClassPerceptron perceptron1 = new TwoClassPerceptron();
//		// TwoClassAvgPerceptron perceptron2 = new TwoClassAvgPerceptron();
//		Perceptron perceptron = new Perceptron();
//		// System.out.println("Simple Perceptron Algorithm");
//		// matrices.displayMatrix(perceptron1.computeW(W, 5,
//		// features,targetClasses));
//		// System.out.println("Averaged Perceptron Algorithm");
//		// matrices.displayMatrix(perceptron2.computeW(W, 5,
//		// features,targetClasses));
//		// System.out.println("Kernel Perceptron");
//		/*
//		 * a = linear kernel b = polynomial kernel c = gausian kernel
//		 */
//		// io.generateClassificationFiles(args[0]);
//		W = perceptron.computeW(W, 1, features, trueLabels);
//		W.print(4, 2);
//		// perceptron3.classify(features, targetClasses, featuresTest, alfa,
//		// 'b', 3, -1);
//		// perceptron3.classify(features, targetClasses, featuresTest, alfa,
//		// 'b', 2, -1);

	}
}
