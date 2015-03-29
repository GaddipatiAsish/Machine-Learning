package com.perceptron.test;

import java.io.IOException;

import com.perceptron.algorithms.AvgKernelPerceptron;
import com.perceptron.algorithms.ConfMatrix;
import com.perceptron.algorithms.KernelPerceptron;
import com.perceptron.algorithms.LearnModel;
import com.perceptron.io.IOOperations;
import com.perceptron.optimalParamaters.Epochs;
import com.perceptron.optimalParamaters.Exponent;
import com.perceptron.optimalParamaters.Sigma;

public class Test {
	public static void main(String args[]) throws IOException {

		 /* Optimal T value ## Uncomment code below to find Optimal T ##*/
		 /* Send the Epochs array and get the best T value */
//		 int[] epochs = { 1, 5, 10, 20 };
//		 Epochs ep = new Epochs(epochs);
//		 System.out.println(ep.getOptimalValue());

		 /* Optimal exponent(d) Value ## Uncomment code below to find Optimal T */
//		 int[] exponent ={2,3,4,5,6};
//		 Exponent oe=new Exponent(exponent);
//		 System.out.println(oe.getOptValue());
		
		 /* Optimal Sigma Value ## Uncomment code below to find Optimal T */
		 double[] sigma ={0.5,2,5,10};
		 Sigma optSigma=new Sigma(sigma);
		 System.out.println(optSigma.getOptValue());
		
		/*Un Comment the code below to generate the model files */
		/* Generate .model files for polynomial Kernel Perceptron */
//		boolean isAvg = false;
													
//		LearnModel polyKernelModels = new LearnModel('b'/*poly K*/, 5/*degree*/, 5/*epochs*/, isAvg);
//		polyKernelModels.generateModelFileKP();
		
		/* Generate .model files for Avg polynomial Kernel Perceptron */
//		isAvg = true;
//		LearnModel polyAvgKernelModels = new LearnModel('b'/*Polynomial Kernel*/, 5 /*d value*/, 5/*Epochs*/, isAvg);
//		polyAvgKernelModels.generateModelFileAvgKP();
		
		/* Generate .model files for gausian Kernel Perceptron */
//		isAvg = false;
//		LearnModel gausianKernelModels = new LearnModel('c'/*Gau K*/, 0.5/*sigma*/, 5 /*Epochs*/, isAvg);
//		gausianKernelModels.generateModelFileKP();

		/* Generate .model files for Avg gausian Kernel Perceptron */
//		isAvg = true;
//		LearnModel gausianAvgKernelModels = new LearnModel('c'/*Gau K*/, 0.5 /*sigma*/, 5 /*Epochs*/, isAvg);
//		gausianAvgKernelModels.generateModelFileAvgKP();

		/* # Use the below codes to Generate the respective confusion Matrices
		 * a= Linear Perceptron 
		 * b= Avg Linear Perceptron 
		 * c= polynomial kernel Perceptron 
		 * d= Avg Polynomial Perceptron 
		 * e= Gausian Kernel Perceptron
		 * f= Avg Gausian Kernel Perceptron
		 */
//		ConfMatrix cMatrix = new ConfMatrix('c', 5/*Epochs for 'a'& 'b'; exponent for 'c'&'d'; sigma for 'e'&'f'*/);
//		cMatrix.readModels(); /* uncomment this line only for 'c','d','e','f'*/
		
//		cMatrix.creatConfusionMatrixPoly("./inputData/optdigits.tes");
//		cMatrix.creatConfusionMatrixPolyAvg("./inputData/optdigits.tes");
		
//		cMatrix.creatConfusionMatrixGau("./inputData/optdigits.tes");
//		cMatrix.creatConfusionMatrixGauAvg("./inputData/optdigits.tes");
		
//		cMatrix.creatConfusionMatrixP("./inputData/optdigits.tes");
//		cMatrix.creatConfusionMatrixAvgP("./inputData/optdigits.tes");

		// IOOperations io= new IOOperations();
		// io.generateClassificationFilesTest("./inputData/optdigits.tes");
//		for (int i = 0; i < 10; i++) {
//		IOOperations io = new IOOperations();
//		io.readInputData("./inputData/" + i + ".tra");
//
//		io.generateSVMDataFile(io.getFeatureMatrix(), io.getTrueLabels(), i
//				+ ".svmtra");
//	}
//		for (int i = 0; i < 10; i++) {
//			IOOperations io = new IOOperations();
//			io.readInputData("./inputData/" + i + ".tes");
//
//			io.generateSVMDataFile(io.getFeatureMatrix(), io.getTrueLabels(), i
//					+ ".svmtes");
//		}
		// io.readInputData("./inputData/optdigits.tes");
		// .generateClassificationFiles("./inputData/optdigits.tes");

	}
}
