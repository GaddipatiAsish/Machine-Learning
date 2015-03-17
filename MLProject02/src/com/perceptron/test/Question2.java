package com.perceptron.test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import weka.core.matrix.Matrix;

import com.perceptron.algorithms.TwoClassAvgPerceptron;
import com.perceptron.algorithms.TwoClassKernelPerceptron;
import com.perceptron.algorithms.TwoClassPerceptron;
import com.perceptron.io.IOOperations;
import com.perceptron.martixOperations.MatrixOperations;

public class Question2 {

	public static void main(String args[]) throws IOException {
		IOOperations io = new IOOperations();
		io.readInputData(args[0]);
		
		/*get target and features vector for training data set*/
		List<List<Double>> feat = io.getFeatureVector();
		List<Integer> trueLabels = io.getTargetVector();
		
		//System.out.println(io.getTargetVector());
		//System.out.println(io.getKClasses());
		
		MatrixOperations matrices= new MatrixOperations();
		/*n denotes the data size N*/
		int n= io.getFeatureVector().get(0).size();
		/* Initialization of Weight Vector */
		Matrix W = new Matrix(n, 1);
		for(int i=0;i<n;i++){
			W.set(i , 0, 0); /*W=[0,0,0.....]*/
		}
		System.out.println("W");
		//matrices.displayMatrix(W);
		List<Matrix> features= new ArrayList<Matrix>();
		Iterator<List<Double>> iterator1=feat.iterator();
		while(iterator1.hasNext()){
			List<Double> temp= iterator1.next();
			Matrix featuresOfXi=new Matrix(n, 1);
			for(int i=0;i<temp.size();i++){
				featuresOfXi.set(i, 0, temp.get(i));
			}
			//System.out.println("-");
			//System.out.println("fi");
			//matrices.displayMatrix(featuresOfXi);
			//System.out.println("-");
			features.add(featuresOfXi);
		}
				
		
		//TwoClassPerceptron perceptron1 = new TwoClassPerceptron();
		//TwoClassAvgPerceptron perceptron2 = new TwoClassAvgPerceptron();
		TwoClassKernelPerceptron perceptron3=new TwoClassKernelPerceptron();
		//System.out.println("Simple Perceptron Algorithm");
		//matrices.displayMatrix(perceptron1.computeW(W, 5, features,targetClasses));
		//System.out.println("Averaged Perceptron Algorithm");
		//matrices.displayMatrix(perceptron2.computeW(W, 5, features,targetClasses));
		System.out.println("Kernel Perceptron");
		/*
		 * a = linear kernel
		 * b = polynomial kernel
		 * c = gausian kernel 
		 */
		//io.generateClassificationFiles(args[0]);
		Matrix alfa=perceptron3.computeAlfaMatrix(W, 20, features, trueLabels,'b' /*kernel type*/,5/*exponent*/,-1/*sigma*/);
		//perceptron3.classify(features, targetClasses, featuresTest, alfa, 'b', 3, -1);
		//perceptron3.classify(features, targetClasses, featuresTest, alfa, 'b', 2, -1);
	
	}
}
