package com.perceptron.optimalParamaters;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import com.perceptron.threads.SigmaThreadRun;

public class Sigma {
	double[] sigma;/* array of d values for optimization */
	List<Double> results = new ArrayList<Double>();
	/**
	 * Exponent constructor method used to initial the exponent array.
	 * 
	 * @param exponent
	 */
	public Sigma(double[] sigma) {
		this.sigma = sigma;
	}
	public double getOptValue() {
		double optimalSigmaCount = 0;
		/* Do Multi Threading Here to compute the best Sigma */

		/* create the threads. Each for an Epoch value */
		final ExecutorService service;
		List<Future<Double>> task = new ArrayList<Future<Double>>();

		service = Executors.newFixedThreadPool(sigma.length);
		for (int i = 0; i < sigma.length; i++) {
			task.add(service.submit(new SigmaThreadRun(sigma[i])));
		}

		/* get the task results and print them */
		for (int i = 0; i < sigma.length; i++) {
			try {

				results.add(task.get(i).get());

			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
		}
		/* shut down the service */
		service.shutdownNow();
		System.out.println("Choose optimal Sigma from % here: " + results);
		/* choose the best T from accuracies resulted */
		optimalSigmaCount=sigma[results.indexOf(Collections.max(results))];
		return optimalSigmaCount;
	}
	
}
