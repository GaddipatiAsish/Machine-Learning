package com.perceptron.optimalParamaters;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.*;

import com.perceptron.threads.EpochThreadRun;

public class Epochs {
	/**
	 * Overloaded constructor that takes an array of possible values for T to
	 * compute the best T
	 * 
	 * @param epochs
	 */
	int[] epochs;
	List<Double> results = new ArrayList<Double>();

	public Epochs(int[] epochs) {
		this.epochs = epochs;
	}

	public int getOptimalValue() {
		int optimalEpochCount = 0;
		/* Do Multi Threading Here to compute the best T */

		/* create the threads. Each for an Epoch value */
		final ExecutorService service;
		List<Future<Double>> task = new ArrayList<Future<Double>>();

		service = Executors.newFixedThreadPool(epochs.length);
		for (int i = 0; i < epochs.length; i++) {
			task.add(service.submit(new EpochThreadRun(epochs[i])));
		}

		/* get the task results and print them */
		for (int i = 0; i < epochs.length; i++) {
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
		System.out.println("Choose optimal T from % here: " + results);
		/* choose the best T from accuracies resulted */
		return epochs[results.indexOf(Collections.max(results))];
	}
}
