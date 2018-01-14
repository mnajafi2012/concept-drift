import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map.Entry;
import java.util.Queue;
import java.util.Random;
import java.util.Set;

/**
 * 
 * @author Maryam Najafi, mnajafi2012@my.fit.edu
 *
 * Mar 3, 2017
 * Course:  CSE 5693, Fall 2017
 * Project: HW3, Artificial Neural Networks
 */

// Network's configuration:
// - Single-layer perceptron with at most 4 hidden units.
// - Linear and logistic activation (sigmoid) functions
// - Multiple and Single-output network
// - Error Back-propagation algorithm for Multi-Layer Perceptron
// - Forward Propagation algorithm
// - Loss function: E = SSE = 1/2 || t - o ||^2; where t is a vector of the actual targets, and
//                                           o is a vector of the outputs from the output layer from all hidden units.
// - Loss function derivative: ∂(E)/∂(o) = ( t - y)


public class Main {
	
	
	static double[] classes;
	// classes in tennis: 0/1 = yes/no; whereas in iris: 0/1/2 = setosa versicolor virginica
	static HashMap<String, ArrayList<String>> attr_vals = new HashMap<String, ArrayList<String>>();
	private static int attrs_size = 4;
	static String[] attrs; // Tennis has 4 attrs while Iris needs to be preprocessed.
	static String[] attrs_orig = new String[4];
	static Random rnd = new Random();

	static int iter = 0, epoch, num_of_units; static int total_iter = 0;
	static List<Integer> TOTAL_ITER = new ArrayList<>();
	// the dimension of each sample
	static int input_size, output_size;
	static String name_dataset;
	static ArrayList<double[]> deltas = null;
	static ArrayList<double[]> Xs;
	static double[] sse = null;
	static ArrayList<double[][]> hidden_vals = null;
	static List<Exp> examples_train, examples_test, examples_val, examples_train_batch, examples_test_batch;
	static double[] network_error_on_train, network_error_on_val, network_error_on_test;
	
	private static double ratio = 1; // e.g. 90% goes as training, the rest as val. set
	private static int max_noise_limit = 22; // 20% given by the problem
	static int percentage = 0;
	
	static FileWriter saved_weights, hidden_values_exp1,
	hidden_values_exp2,hidden_values_exp3, hidden_values_exp4,
	hidden_values_exp5,hidden_values_exp6,hidden_values_exp7,
	hidden_values_exp8;
	static FileWriter iris_0, iris_2, iris_4,
	iris_6, iris_8, iris_10, iris_12, iris_14,
	iris_16, iris_18, iris_20;
	// last updated weights
	static double[] last_weights;
	static List<Exp> batch = new ArrayList<Exp>();
	
	public static void main (String[] args) throws FileNotFoundException, IOException, java.lang.IndexOutOfBoundsException{
		
		// 0. INITIALIZATION
		
		name_dataset =  args[0]; // name of the dataset
		num_of_units = Integer.valueOf(args[1]); // # of hidden units
		double etha = .4; // learning rate
		double init_weight = .1; // small random weights (or specify .05)
		
		// stopping criterion
		epoch = name_dataset.equalsIgnoreCase("identity")?5000:1000; // max iter
		epoch = name_dataset.equalsIgnoreCase("tennis")?100:epoch;
		epoch = name_dataset.equalsIgnoreCase("sine1")?7000:epoch;
		epoch = name_dataset.equalsIgnoreCase("circles")?1000:epoch;
		epoch = name_dataset.equalsIgnoreCase("hyperplane")?5000:epoch;
		double conv_thr = .0001; // convergence threshold
		
		// TEST TYPE either inst(instantaneous / on the fly), fixed (examples of either concepts)
		String test_type = "fixed";
		// SGD version - minibatch or fullbatch
		String SGD_version = "fullbatch";
		// batch size (number of samples in one batch)
		int batch_size = 50;
		// train to test proportion in one batch
		double batch_proportion = (test_type.equalsIgnoreCase("fixed"))?1:.75;
		
		// ADWIN properties
		int n0 = 1, n1 = 1; String drift_detection = "";
		
		// stationary (tennis, identity, iris... ) nonstationary (SINE1, CIRCLES, HYPERPLANE)
		String learning_type = "online";
		// train examples
		examples_train = new ArrayList<Exp>(); //List<Exp> examples_train = new ArrayList<Exp>();
		// test examples
		examples_test = new ArrayList<Exp>(); //List<Exp> examples_test = new ArrayList<Exp>();
		// validation set
		examples_val = new ArrayList<Exp>();
		// batch examples (comprising train and test in every batch)
		examples_train_batch = new ArrayList<Exp>();
		examples_test_batch = new ArrayList<Exp>();

		// train batch length
		int length = (int) Math.round(batch_proportion * batch_size);
		// accuracy of NN exposed to the non-stationary data set at time stamps from 0 to eof
		List<Double> ACC = new ArrayList<Double>();
		
		rnd.setSeed((long) 0);
		double momentum = .1; // momentum
		ratio = (args.length == 3)? Double.valueOf(args[2]):0; setRatio(ratio);
		
		if (args.length > 2){
			etha = Double.valueOf(args[2]); // get learning rate from user
			conv_thr = Double.valueOf(args[3]); // get convergence threshold from user
			epoch = args[4].equalsIgnoreCase("default")?epoch:Integer.valueOf(args[4]); // get maximum iterations from user
			n0 = Integer.valueOf(args[5]); // get sub-window0 size from user
			n1 = Integer.valueOf(args[6]); // get sub-window1 size from user
		}
		switch (args.length){
		case 9: {
			SGD_version = args[7]; // get fullbatch from user
			drift_detection = args[8];
			break;
		}
		case 11:{
			SGD_version = args[7]; // get minibatch from user
			batch_size = Integer.valueOf(args[8]); // get batch size from user
			test_type = args[9]; // get concept1 or concept2 from user
			drift_detection = args[10];
			break;
		}
		case 12:{
			SGD_version = args[7]; // get minibatch from user
			batch_size = Integer.valueOf(args[8]); // get batch size from user
			test_type = args[9]; // get instantaneous ("inst") test data type from user
			batch_proportion = (test_type.equalsIgnoreCase("fixed"))?1:Double.valueOf(args[10]);
			drift_detection = args[11];
			break;
		}
			default:{
				break;
			}
		}
		
		network_error_on_train = new double[epoch]; network_error_on_val = new double[epoch]; network_error_on_test = new double[epoch];
		ArrayList<Integer> list = null; // list of shuffled indices
		
		
		String txt_attrs, txt_input_train, txt_input_test, shuffled_num_address, v = getRatio()==1?"NOval":"Val";
		saved_weights = new FileWriter(System.getProperty("user.dir") + "/" + name_dataset + "weights_" + num_of_units+ "units_" + v  + ".csv");
		if (name_dataset.equalsIgnoreCase("identity")){
			hidden_values_exp1 = new FileWriter(System.getProperty("user.dir") + "/" + name_dataset + "hidden_values_exp10000000_" + num_of_units+"units"+".csv");
			hidden_values_exp2 = new FileWriter(System.getProperty("user.dir") + "/" + name_dataset + "hidden_values_exp01000000_" + num_of_units+"units"+".csv");
			hidden_values_exp3 = new FileWriter(System.getProperty("user.dir") + "/" + name_dataset + "hidden_values_exp00100000_" + num_of_units+"units"+".csv");
			hidden_values_exp4 = new FileWriter(System.getProperty("user.dir") + "/" + name_dataset + "hidden_values_exp00010000_" + num_of_units+"units"+".csv");
			hidden_values_exp5 = new FileWriter(System.getProperty("user.dir") + "/" + name_dataset + "hidden_values_exp00001000_" + num_of_units+"units"+".csv");
			hidden_values_exp6 = new FileWriter(System.getProperty("user.dir") + "/" + name_dataset + "hidden_values_exp00000100_" + num_of_units+"units"+".csv");
			hidden_values_exp7 = new FileWriter(System.getProperty("user.dir") + "/" + name_dataset + "hidden_values_exp00000010_" + num_of_units+"units"+".csv");
			hidden_values_exp8 = new FileWriter(System.getProperty("user.dir") + "/" + name_dataset + "hidden_values_exp00000001_" + num_of_units+"units"+".csv");
		}
		if (name_dataset.equalsIgnoreCase("irisnoisy")){
			/*
			iris_0 = new FileWriter(System.getProperty("user.dir") + "/" + name_dataset + "_0"+".txt");
			iris_2 = new FileWriter(System.getProperty("user.dir") + "/" + name_dataset + "_2"+".txt");
			iris_4 = new FileWriter(System.getProperty("user.dir") + "/" + name_dataset + "_4"+".txt");
			iris_6 = new FileWriter(System.getProperty("user.dir") + "/" + name_dataset + "_6"+".txt");
			iris_8 = new FileWriter(System.getProperty("user.dir") + "/" + name_dataset + "_8"+".txt");
			iris_10 = new FileWriter(System.getProperty("user.dir") + "/" + name_dataset + "_10"+".txt");
			iris_12 = new FileWriter(System.getProperty("user.dir") + "/" + name_dataset + "_12"+".txt");
			iris_14 = new FileWriter(System.getProperty("user.dir") + "/" + name_dataset + "_14"+".txt");
			iris_16 = new FileWriter(System.getProperty("user.dir") + "/" + name_dataset + "_16"+".txt");
			iris_18 = new FileWriter(System.getProperty("user.dir") + "/" + name_dataset + "_18"+".txt");
			iris_20 = new FileWriter(System.getProperty("user.dir") + "/" + name_dataset + "_20"+".txt");
			*/
		}
		
		// 1. READ DATASET & PREPROCESS
		if (name_dataset.equalsIgnoreCase("identity")) {
			// IDENTITY
			System.out.printf("Identity Data Set: %n-------------%nThe results are saved in *.csv files.%n");
			txt_attrs = System.getProperty("user.dir").concat("/identity-attr.txt");
			txt_input_train = System.getProperty("user.dir").concat("/identity-train.txt");
			txt_input_test = System.getProperty("user.dir").concat("/identity-test.txt");
			
			// a) read from input text file for the identity dataset
			setAttrs_size(8);
			readAttributes(txt_attrs);
			attrs_orig = attrs.clone();
			examples_train = readExamples(txt_input_train);
			
			examples_test = readExamples(txt_input_test);
			
			// keep track of hidden values of each example
			hidden_vals = new ArrayList<double[][]>();
			
			max_noise_limit = 0;
			learning_type = "offline";

		} else if (name_dataset.equalsIgnoreCase("tennis")) {
			// TENNIS
			System.out.printf("Tennis Data Set: %n-------------%n");
			txt_attrs = System.getProperty("user.dir").concat("/tennis-attr.txt");
			txt_input_train = System.getProperty("user.dir").concat("/tennis-train.txt");
			txt_input_test = System.getProperty("user.dir").concat("/tennis-test.txt");

			// b) read from input text file for the tennis dataset
			setAttrs_size(4);
			readAttributes(txt_attrs);
			attrs_orig = attrs.clone();

			// pre-process discrete input/output attributes
			preprocess_Tennis(examples_train);

			examples_train = readExamples(txt_input_train);
			examples_test = readExamples(txt_input_test);

			max_noise_limit = 0;
			learning_type = "offline";
			
		} else if (name_dataset.equalsIgnoreCase("iris")) {
			// IRIS
			System.out.printf("Iris Data Set: %n-------------%n");
			txt_attrs = System.getProperty("user.dir").concat("/iris-attr.txt");
			txt_input_train = System.getProperty("user.dir").concat("/iris-train.txt");
			txt_input_test = System.getProperty("user.dir").concat("/iris-test.txt");

			// c) read from input text file for the iris dataset
			setAttrs_size(4);
			readAttributes(txt_attrs);
			attrs_orig = attrs.clone();

			examples_train = readExamples(txt_input_train);
			examples_test = readExamples(txt_input_test);
			
			max_noise_limit = 0;
			if (args.length > 2){
				learning_type = "online";
			}else{
				learning_type = "offline";
			}

		} else if (name_dataset.equalsIgnoreCase("irisnoisy")){
			// IRIS
			System.out.printf("Iris Noisy Data Set: %n-------------%n");
			txt_attrs = System.getProperty("user.dir").concat("/iris-attr.txt");
			txt_input_train = System.getProperty("user.dir").concat("/iris-train.txt");
			txt_input_test = System.getProperty("user.dir").concat("/iris-test.txt");

			// c) read from input text file for the iris dataset
			setAttrs_size(4);
			readAttributes(txt_attrs);
			attrs_orig = attrs.clone();

			examples_train = readExamples(txt_input_train);
			
			// split training data into training and validation sets
			Pair<List<Exp>, List<Exp>> train_val = split_into_train_val(examples_train, getRatio());
			examples_train = train_val.getfirst();
			examples_val = train_val.getsecond();
			
			
			examples_test = readExamples(txt_input_test);
			
			
			// IRIS NOISY
			// inject noise (do it once and record noisy data labeled with the percentage of noise)
			list = getShuffledList(examples_train);
			
			/*
			// CREATE CORRUPT IRIS DATA
			// Inject noise from 0% to 20% by 2% increment
			for (int percentage = 0; percentage <= max_noise_limit; percentage = percentage + 2){
				// CORRPUT THE TRAINING EXAMPLES
				examples_train = corrupt(examples_train, list, percentage);
				
				System.out.printf("%n%d %% of noise%n", percentage);
				
				// RECORD NOISY DATA ON DISK
				write_to_file(examples_train, percentage);
				
			}			
			*/
			
			/*
			shuffled_num_address = System.getProperty("user.dir").concat("/shuffledList.txt");
			list = readList_fromFile(shuffled_num_address);
			*/
			
			learning_type = "offline";
			
		}else if (name_dataset.equalsIgnoreCase("toy")) {
			// IRIS
			System.out.printf("TOY Data Set: %n-------------%n");
			txt_attrs = System.getProperty("user.dir").concat("/toy-attr.txt");
			txt_input_train = System.getProperty("user.dir").concat("/toy-train.txt");
			txt_input_test = System.getProperty("user.dir").concat("/toy-test.txt");

			// c) read from input text file for the iris dataset
			setAttrs_size(2);
			readAttributes(txt_attrs);
			attrs_orig = attrs.clone();

			examples_train = readExamples(txt_input_train);
			examples_test = readExamples(txt_input_test);
			
			max_noise_limit = 0;
			learning_type = "online";

		}else if (name_dataset.equalsIgnoreCase("hyperplane")) {
			// IRIS
			System.out.printf("HYPERPLANE Data Set: %n-------------%n");
			txt_attrs = System.getProperty("user.dir").concat("/hyperplane-attr.txt");
			txt_input_train = System.getProperty("user.dir").concat("/hyperplane-train.txt");
			txt_input_test = System.getProperty("user.dir").concat("/hyperplane-test.txt");

			// c) read from input text file for the iris dataset
			setAttrs_size(2);
			readAttributes(txt_attrs);
			attrs_orig = attrs.clone();

			examples_train = readExamples(txt_input_train);
			examples_test = readExamples(txt_input_test);
			
			max_noise_limit = 0;
			learning_type = "online";
			
			
		}else if (name_dataset.equalsIgnoreCase("circles")) {
		
			// IRIS
			System.out.printf("CIRCLES Data Set: %n-------------%n");
			txt_attrs = System.getProperty("user.dir").concat("/circles-attr.txt");
			txt_input_train = System.getProperty("user.dir").concat("/circles-train.txt");
			txt_input_test = System.getProperty("user.dir").concat("/circles-test.txt");

			// c) read from input text file for the iris dataset
			setAttrs_size(2);
			readAttributes(txt_attrs);
			attrs_orig = attrs.clone();

			examples_train = readExamples(txt_input_train);
			examples_test = readExamples(txt_input_test);
			
			max_noise_limit = 0;
			learning_type = "online";
		}else if (name_dataset.equalsIgnoreCase("sine1")) {
			// IRIS
			System.out.printf("SINE1 Data Set: %n-------------%n");
			txt_attrs = System.getProperty("user.dir").concat("/sine1-attr.txt");
			txt_input_train = System.getProperty("user.dir").concat("/sine1-train.txt");
			txt_input_test = System.getProperty("user.dir").concat("/sine1-test.txt");

			// c) read from input text file for the iris dataset
			setAttrs_size(2);
			readAttributes(txt_attrs);
			attrs_orig = attrs.clone();

			examples_train = readExamples(txt_input_train);
			examples_test = readExamples(txt_input_test);
			
			max_noise_limit = 0;
			//learning_type = "online";
		}

		do {
			// GET THE NUMBER OF INPUT TO THE NETWORK
			input_size = examples_train.get(0).size();

			// GET THE NUMBER OF OUTPUT TO THE NETWORK
			output_size = classes.length;

			// CREATE A FEED-FORWARD NETWORK
			Network network = createNetwork(init_weight);

			// TRAIN THE NETWORK AGAINST TRAIN EXAMPLES (or VAL SET for
			// Irisnoisy)
			double weights[] = network.getWeights(0, 0);
			write_to_file(weights, "weights");
			saved_weights.append("\n");

			// DO BATCH TRAIN AND TEST ONLY FOR NON-STATIONARY DATA SETS
			if (learning_type.equalsIgnoreCase("online")) {

				try{
					
					// FOR TIME STAMP t
					/*
					for (int t = 0; t < 1000; t++) {
						// CREATE THE BATCH OF SIZE e.g. 50 DATA POINTS
						//examples_train_batch = createBatch(examples_train, t, batch_size, length );
						//examples_test_batch = createBatch(examples_test, t, batch_size, batch_size - length );
						System.out.println(t);
						Pair<List<Exp>, List<Exp>> examples_train_test = createBatch(t, batch_size, length );

						examples_train_batch = examples_train_test.getfirst();
						examples_test_batch = examples_train_test.getsecond();
						
						// TRAIN
						iter = 0;
						network = train(network, examples_train_batch, etha);

						// TEST (ON TRAIN SET)
						double acc;
						// acc = test(network, examples_train_batch);
						//System.out.printf(
						//		"The accuracy of %d-unit ANN on " + name_dataset
						//				+ " train data after %d number of iterations: %.1f%%%n",
						//		num_of_units, epoch, acc * 100);

						// TEST (ON TEST SET)
						acc = test(network, examples_test);
						System.out.printf(
								"The accuracy of %d-unit ANN on " + name_dataset
										+ " test data after %d number of iterations: %.1f%%%n",
								num_of_units, epoch, acc * 100);
						
						// RECORD ACCURACIES IN EVERY ITERATION (FOR TIME STAMP t)
						ACC.add(acc * 100);
						
					}// end of for (index t)
				
				*/
					ADWIN adwin = new ADWIN(n0, n1);
					adwin.setDim(examples_train.get(0).size());
					List<Exp> train_test_examples = new ArrayList<Exp>();
					
					// FOR TIME STAMP t
					for (int t = 0; t < 10000; t++){
						
						//System.out.printf("t: %d%n", t);
						// stream the data point to the adwin to estimate the concept drift
						if (name_dataset.equalsIgnoreCase("SINE1")){
							// hard coded drift detection at the time being (due to false positive for distribution-based approach)
							adwin.addElement(examples_train.get(t), "SINE1");
						}else{
							adwin.addElement(examples_train.get(t));
						}
						
						// read from the user whether detector detects the drift or not
						if (!drift_detection.equalsIgnoreCase("drift")){
							adwin.y_or_n = false;
						}
						
						
						// the context is detected. Train the network from beginning with new examples.
						if (adwin.y_or_n){
							//System.out.printf("in y_or_n %s%n", adwin.y_or_n);
							network = createNetwork(init_weight);
							examples_train_batch = new ArrayList<Exp>();
							
							// get the train-test data for the next context (purified data)
							examples_train_batch = adwin.get_batch();
							
						}else{
							// stream the data point into the batch
							train_test_examples = addToData(examples_train.get(t), train_test_examples);
							
							//adwin.y_or_n = false;
						}
						
						// MINI BATCH STOCHASTIC GRADIENT DESCENT
						// pick unique sequence of batches
						if  (((t + 1)%batch_size == 0) && (!train_test_examples.isEmpty())){
							
							assert (!train_test_examples.isEmpty()):"empty batch!";
							

							Pair<List<Exp>, List<Exp>> examples_train_test = createBatch(train_test_examples, batch_proportion, SGD_version);
							examples_train_batch = examples_train_test.getfirst();
							examples_test_batch = examples_train_test.getsecond();
							
							// TRAIN
							iter = 0;
							if (SGD_version.equalsIgnoreCase("fullbatch")){
								network = createNetwork(init_weight);
							}
							
							network = train(network, examples_train_batch, etha, conv_thr);
							// TEST
							double acc = .0;
							switch (test_type.toLowerCase()) {
							case "inst": { // instantaneous (subset from the batch)
								// gets the test data from within the batch itself. (train and test data are mutually exclusive)
								acc = test(network, examples_test_batch);
								break;
							}
							case "fixed": { // e.g. 100 fixed samples from context1/2 (generated previously)
								acc = test(network, examples_test);
								break;
							}
							default: {
								acc = test(network, examples_test_batch);
								break;
							}
							}
							
							
							System.out.printf(
									"The accuracy of %d-unit ANN on " + name_dataset
											+ " test data after %d number of iterations: %.1f%%%n%n",
									num_of_units, iter, acc * 100);
							
							// RECORD ACCURACIES IN EVERY ITERATION (FOR TIME STAMP t)
							ACC.add(acc * 100);
						}
					}

				}
				catch(java.lang.IndexOutOfBoundsException e){
					System.out.printf("done!%n%n");
				}

			} else { // if NOT non-stationary

				//for (int w = 0; w < 4; w++){ // to demonstrate we're bug free
				if (name_dataset.equalsIgnoreCase("irisnoisy") && getRatio() != 1) {
					network = train(network, examples_train, examples_val, etha);
				} else {
					network = train(network, examples_train, etha, conv_thr);
				}

				double acc = .0;
				if (!name_dataset.equalsIgnoreCase("identity")) {
					// TEST THE NETWORK ON TRAIN SET
					acc = test(network, examples_train);
					System.out.printf(
							"The accuracy of %d-unit ANN on " + name_dataset
									+ " train data after %d number of iterations: %.1f%%%n",
							num_of_units, epoch, acc * 100);
					
					// TEST THE NETWORK ON TEST SET
					acc = test(network, examples_test);
					System.out.printf(
							"The accuracy of %d-unit ANN on " + name_dataset
									+ " test data after %d number of iterations: %.1f%%%n",
							num_of_units, iter, acc * 100);

				}

				if (name_dataset.equalsIgnoreCase("irisnoisy")) {

					percentage += 2; // Inject noise from 0% to 20% by 2%
										// increment
					iter = 0;

					if (percentage < max_noise_limit) {
						System.out.printf("%n%d %% of noise%n", percentage);
					}

					// CREATE CORRUPT IRIS DATA
					// CORRPUT THE TRAINING EXAMPLES
					examples_train = corrupt(examples_train, list, percentage);

					// PRINT NETWORK ERROR ON BOTH TRAIN AND VAL DATASETS
					// AGAINST ITERATIONS (FOR REPORT)
					/*
					 * for (double a : network_error_on_train){
					 * System.out.printf("%.2f\n",a); } System.out.println();
					 * for (double a : network_error_on_val){
					 * System.out.printf("%.2f\n",a); }
					 */
				}
				
			//} // end of loop over w (to show there's no bug)
			} // end of if non-stationary
		} while (percentage < max_noise_limit);
		
		if (name_dataset.equalsIgnoreCase("identity")){
		hidden_values_exp1.close();
		hidden_values_exp2.close();
		hidden_values_exp3.close();
		hidden_values_exp4.close();
		hidden_values_exp5.close();
		hidden_values_exp6.close();
		hidden_values_exp7.close();
		hidden_values_exp8.close();}
		saved_weights.close();
		
		System.out.println("The error rate for each training");
		for (double a : ACC){
			System.out.printf("%.1f%n",100-a);
		}

		//System.out.printf("%ntotal iterations for each training%n");
		//for (int it : TOTAL_ITER){
			//System.out.printf("%d%n",it);
		//}
		System.out.printf("%nTotal # of iterations to train is %d%n", total_iter);
	}
	
	
	private static List<Exp> addToData(Exp exp, List<Exp> train_test_examples) {
		// stream data
		train_test_examples.add(exp);
		
		return train_test_examples;
	}


	/**
	 * 
	 * @param examples_train all training examples (are observed by the algorithm at each time stamp consecutively)
	 * @param t the time stamp at which a new batch is injected to the NN
	 * @param batch_size the number of samples in one batch
	 * @param batch_proportion 
	 * @return
	 */
	private static Pair<List<Exp>, List<Exp>> createBatch(int t, int batch_size, int length) {
		// fetches batch_size number of samples at time stamp t from train examples
		
		if ((t*batch_size%1000) == 0){
			//System.out.println();
			examples_train_batch.clear();
		}
		examples_test_batch.clear();
		
		for (int i = batch_size * t; i < (batch_size * t) + (length); i++){
			examples_train_batch.add(examples_train.get(i));
			//sentinel++;
		}
		
		for (int i = (batch_size * t ) + length; i < batch_size * (t + 1); i++){
			examples_test_batch.add(examples_train.get(i));
		}

		return new Pair<List<Exp>, List<Exp>> (examples_train_batch, examples_test_batch);
	}
	
	private static Pair<List<Exp>, List<Exp>> createBatch(List<Exp> examples, double proportion, String SGD_version) {
		// fetches batch_size number of samples at time stamp t from train examples
		int sz = examples.size();
		
		if (SGD_version.equalsIgnoreCase("minibatch")){
			examples_train_batch.clear();
			
			for (int i = 0; i < (int) (Math.ceil(sz * proportion)) ; i++){
				examples_train_batch.add(examples.get(i));
				//sentinel++;
			}
		}else if (SGD_version.equalsIgnoreCase("fullbatch")){
			//examples_train_batch.add(examples.get(examples.size() - 1));
			for (int i = 0; i < (int) (Math.ceil(sz * proportion)) ; i++){
				examples_train_batch.add(examples.get(i));
				//sentinel++;
			}
		}

		// if we use fullbatch SGD there is no test data for the incremental learning in this case!
		examples_test_batch.clear();
		
		for (int i = (int) (Math.ceil(sz * proportion)); i < sz; i++){
			examples_test_batch.add(examples.get(i));
		}
		
		
		examples.clear(); // shallow copied of train-test-examples

		return new Pair<List<Exp>, List<Exp>> (examples_train_batch, examples_test_batch);
	}

	private static ArrayList<Integer> readList_fromFile(String filepath) throws IOException {
			// reads input txt file (shuffled numbers)
			
			ArrayList<Integer> list = new ArrayList<Integer>();
			String[] tmp = null;
			
			try (BufferedReader reader = new BufferedReader(new FileReader(filepath))) {
				String line = reader.readLine();
				
				while (line != null){
					
					// Write to the array
					tmp = line.split(", ");
					
					line = reader.readLine();
				}
				
				for (String in : tmp){
					list.add(Integer.valueOf(in));
				}
			}

			return list;
		
	}

	private static Network train(Network network, List<Exp> examples_train, List<Exp> examples_val, double etha)
			throws FileNotFoundException, IOException{
		double err_val = 1, min_err_val = Double.MAX_VALUE; Network best_network = null;
		// TERMINATION CONDITION:
		while (iter < epoch){
			
			int counter = 0;
			double[][] hid_tmp = new double[input_size][num_of_units];
			
			// LOOP OVER THE EXAMPLES:
			for (Exp e : examples_train){
				
				//double[] weights = network.printWeights(0, 0);
				// print ingoing weights from input of hidden layer (0)
				// to the first hidden unit (0)
				
				
				// 3_1. FORWARD PROPAGATION ALGORITHM
				// INDUCE AN INPUT SAMPLE
				Xs = network.forwardPropagation(e);
				
				
				// 3_2. BACKPROPAGATION ALGORITHM
				// (propagate the errors backward)
				deltas = network.backpropagation(e);
				

				// 3_3. UPDATE WEIGHTS: GRADIENT-DESCENT OPTIMIZATION ALGORITHM
				network.updateWeights(etha, deltas, Xs );
				
				// CALCULATE SSE TO PLOT
				//sse = network.SSE(network.L.get(1).getZ(), e.getTarget());

				if (name_dataset.equalsIgnoreCase("identity")){
					// hidden unit values
					hid_tmp[counter] = Xs.get(1).clone();
				}
				//System.out.printf("%.3f, %.3f, %.3f %n", Xs.get(1)[0],Xs.get(1)[1], Xs.get(1)[2] );
			
				counter++;
				
			}
			
			double weights[] = network.getWeights(0, 0);
			write_to_file(weights, "weights");
			
			if (name_dataset.equalsIgnoreCase("identity")){
			hidden_vals.add(hid_tmp); // hidden unit values for distinct examples
			write_to_file(hidden_vals.get(iter));
			//write_to_file(sse, "layer_values");
			
			// RECORD HIDDEN UNIT ENCODING FOR THE FIRST THREE EXAMPLES
			hidden_values_exp1.append("\n");
			hidden_values_exp2.append("\n");
			hidden_values_exp3.append("\n");
			hidden_values_exp4.append("\n");
			hidden_values_exp5.append("\n");
			hidden_values_exp6.append("\n");
			hidden_values_exp7.append("\n");
			hidden_values_exp8.append("\n");}
			saved_weights.append("\n");
			//save.flush();
			
			iter++;
			//System.out.printf("iter: %d%n" ,iter);
			
			// TRACE THE PERFORMANCE OF NETWORK AFTER EACH WEIGHT UPDATE
			// TEST THE NETWORK ON BOTH TRAIN SET
			double acc = test(network, examples_train);
			//System.out.printf("The accuracy of %d-unit ANN on "+name_dataset+" train data after %d number of iterations: %.1f%%%n", num_of_units, iter, acc * 100);
			network_error_on_train[iter - 1] = 1 - acc;
			
			// TEST THE NETWORK ON BOTH TEST SET
			acc = test(network, examples_val);
			//		System.out.printf("The accuracy of %d-unit ANN on "+name_dataset+" test data after %d number of iterations: %.1f%%%n", num_of_units, iter, acc * 100);

			network_error_on_val[iter - 1] = 1 - acc;
			
			// OVERFITTING VERIFICATION
			if (!examples_val.isEmpty()){
				// FOR EVERY UPDATE TEST OVER VALIDATION SET
				double acc_val = test(network, examples_val); err_val = 1 - acc_val;
			}
			if (err_val < min_err_val){
				min_err_val = err_val;
				best_network = network.clone();
			}
		}
		
		return best_network;
	}

	private static double test(Network network, List<Exp> examples) {
		
		double num_of_matched = .0;
		// ITERATE OVER EXAMPLES TO TEST
		for (Exp e: examples){
			
			// DECLARE AND INITIALIZE ACTUAL TARGET
			double[] target = new double[e.getTarget().length];// size of 1 for tennis and iris
			target = e.getTarget();
			
			// GO THROUGH THE NETWORK
			network.forwardPropagation(e);
			
			// FETCH PREDICTED NETWORK
			double[] target_network = network.get_networkOutput();
			
			// COMPARE TARGET TO TARGET_NETWORK IF THEY MATCH
			boolean match = compare(target, target_network);
			
			// count the number of matched targets
			num_of_matched = match?(num_of_matched+1):(num_of_matched);
			
		}
		
		assert !examples.isEmpty():"Empty test set!";
		
		double acc = num_of_matched / examples.size();
		return acc;
	}


	private static boolean compare(double[] target, double[] target_network) {
		boolean match = true;
		for (int i = 0 ; i < target.length; i ++){
			match = target[i]== target_network[i]? true:false;
			
			if (!match) break;
		}
		return match;
	}

	private static Network train(Network network, List<Exp> examples, double etha, double thr)
			throws FileNotFoundException, IOException{
		boolean converged = false;
		
		// TERMINATION CONDITION:
		while ((iter < epoch) && (!converged)){
			
			int counter = 0;
			double[][] hid_tmp = new double[input_size][num_of_units];
			
			// LOOP OVER THE EXAMPLES:
			for (Exp e : examples){
				
				//double[] weights = network.printWeights(0, 0);
				// print ingoing weights from input of hidden layer (0)
				// to the first hidden unit (0)
				
				
				// 3_1. FORWARD PROPAGATION ALGORITHM
				// INDUCE AN INPUT SAMPLE
				Xs = network.forwardPropagation(e);
				
				
				// 3_2. BACKPROPAGATION ALGORITHM
				// (propagate the errors backward)
				deltas = network.backpropagation(e);
				

				// 3_3. UPDATE WEIGHTS: GRADIENT-DESCENT OPTIMIZATION ALGORITHM
				network.updateWeights(etha, deltas, Xs );
				
				// CALCULATE SSE TO PLOT
				//sse = network.SSE(network.L.get(1).getZ(), e.getTarget());

				if (name_dataset.equalsIgnoreCase("identity")){
					// hidden unit values
					hid_tmp[counter] = Xs.get(1).clone();
				}
				//System.out.printf("%.3f, %.3f, %.3f %n", Xs.get(1)[0],Xs.get(1)[1], Xs.get(1)[2] );
			
				counter++;
			}
			
			double weights[] = network.getWeights(0, 0);
			// UPDATE CONDITIONS TO TERMINATE THE WHILE LOOP (iter and converged)
			converged = check_convergence(weights, thr); iter++;
			write_to_file(weights, "weights");
			

			if (name_dataset.equalsIgnoreCase("identity")){
			hidden_vals.add(hid_tmp); // hidden unit values for distinct examples
			write_to_file(hidden_vals.get(iter));
			//write_to_file(sse, "layer_values");
			
			// RECORD HIDDEN UNIT ENCODING FOR THE FIRST THREE EXAMPLES
			hidden_values_exp1.append("\n");
			hidden_values_exp2.append("\n");
			hidden_values_exp3.append("\n");
			hidden_values_exp4.append("\n");
			hidden_values_exp5.append("\n");
			hidden_values_exp6.append("\n");
			hidden_values_exp7.append("\n");
			hidden_values_exp8.append("\n");}
			saved_weights.append("\n");
			//save.flush();
			
			//System.out.printf("iter: %d%n" ,iter);
			
			// FOR REPORT SHOW THE GRADUAL IMPROVEMENT IN PREDICTION
			/*
			if (!name_dataset.equalsIgnoreCase("identity")){
				// TRACE THE PERFORMANCE OF NETWORK AFTER EACH WEIGHT UPDATE
				// TEST THE NETWORK ON TRAIN SET
				double acc = test(network, examples_train);
				//System.out.printf("The accuracy of %d-unit ANN on "+name_dataset+" train data after %d number of iterations: %.1f%%%n", num_of_units, iter, acc * 100);
				network_error_on_train[iter - 1] = 1 - acc;
				
				// TEST THE NETWORK ON TEST SET
				acc = test(network, examples_test);
				//		System.out.printf("The accuracy of %d-unit ANN on "+name_dataset+" test data after %d number of iterations: %.1f%%%n", num_of_units, iter, acc * 100);
	
				network_error_on_test[iter - 1] = 1 - acc;
			}*/
		}
		
		// check which condition is met for the loop termination
		//System.out.printf("with %d iterations the network is converged.%n", iter);
		TOTAL_ITER.add(iter);
		total_iter += iter;
		
		return network;
	}
	
	private static boolean check_convergence(double[] weights, double thr) {
		// read the last saved weights and compare it to the current weights
		
		double diff = .0;
		boolean conv = true;
		// check if there is a negligible difference between current weights and the last updated weights
		for (int i= 0; i < weights.length; i++){
			diff = Math.abs(weights[i] - last_weights[i]);
			if (diff > thr){
				return false;
			}else{
				conv = true;
			}
		}
		
		return conv;
	}

	private static Network createNetwork(double init_weight) {
		// 2_0. CREATE A NETWORK
		Network network = new Network(output_size);
				
		// 2_2. CREATE A PERCEPTRON
		Perceptron p;
		
		// 2_3. CREATE A HIDDEN LAYER OF $num PERCEPTRONS
		Layer hid = new Layer("hidden");
		for (int i = 0; i < num_of_units; i++){
			p = new Perceptron();
			
			// add the perceptron to the layer
			hid.add(p);
		}
		
		// GET THE SIZE OF HIDDEN LAYER
		int num_hid_unit = hid.size();
		
		// 2.4. INIALIZE WEIGHTS WITH RANDOM NUMBERS
		// & SET THE SIZES FOR OUTPUTS BEFORE AND AFTER THRESHOLD
		hid.init(input_size, num_hid_unit, init_weight);

		// 2_5. ADD THE HIDDEN LAYER TO THE NETWORK
		network.add(hid);
		
		// 2_6. CREATE THE OUTPUT LAYER of SIZE $output (classes)
		Layer out = new Layer("output");

		for (int i = 0; i < output_size; i++){
			p = new Perceptron();
			
			// add the perceptron to the layer
			out.add(p);
		}
		
		// 2_7. INIALIZE WEIGHTS WITH RANDOM NUMBERS
		// & SET THE SIZES FOR OUTPUTS BEFORE AND AFTER THRESHOLD
		out.init(num_hid_unit, output_size, init_weight);
		
		// 2_8. ADD THE HIDDEN LAYER TO THE NETWORK
		network.add(out);
		
		return network;
		
	}
	
	private static ArrayList<Integer> getShuffledList(List<Exp> examples_train) {
		ArrayList<Integer> list = new ArrayList<Integer>();
		for( int i = 0; i < examples_train.size(); i++){
			list.add(i);
		}
		Collections.shuffle(list);
		
		return list;
	}
	
	private static List<Exp> corrupt(List<Exp> examples, ArrayList<Integer> list, int percentage) {
		// corrupt training examples by "percentage"%
		if (percentage == 0) {return examples;}
		for (int i = percentage - 2; i< percentage; i++) {
            double[] target_actual_double = examples.get(list.get(i)).getTarget();
            //String target_actual_string = replace(target_actual_double);
            double[] target_random = getRandomTarget(target_actual_double);
            
            //System.out.println(target_actual + " " + target_random);
            //System.out.print(" " + list.get(i));
            examples.get(list.get(i)).setTarget(target_random);
        }
		
		return examples;
	}


	private static double[] getRandomTarget(double[] argin) {// argin could be either: [1 0 0], [0 1 0], or [0 0 1]
		
		double[] output = new double[argin.length]; // the size of 3
		
		if (argin[0] == 1){
			
			double[] idx = new double[2]; // the size of 2
			idx[0] = 1; idx[1] = 2;
			
			int t = (rnd.nextDouble() < 0.5)? 0: 1;
			output[ (int) idx[t] ] = 1;
			return output;
			
		}else if (argin[1] == 1){
			double[] idx = new double[2]; // the size of 2
			idx[0] = 0; idx[1] = 2;
			
			int t = (rnd.nextDouble() < 0.5)? 0: 1;
			output[ (int) idx[t] ] = 1;
			return output;
			
		}else if (argin[2] == 1){
			double[] idx = new double[2]; // the size of 2
			idx[0] = 01; idx[1] = 1;
			
			int t = (rnd.nextDouble() < 0.5)? 0: 1;
			output[ (int) idx[t] ] = 1;
			return output;
			
		}else{
			System.out.println("No good input to 'replace' in Iris noisy!");
			return null;
		}
		
	}

	private static void write_to_file(double[] content, String content_title) throws IOException {
		
		if (content_title.equalsIgnoreCase("weights")){

			// keep as a record for convergence check
			last_weights = new double[content.length];
			last_weights = content.clone();
			
			for (int i =0 ; i < content.length; i++){
				saved_weights.append(String.valueOf(content[i]));
				if (i < content.length - 1){
					saved_weights.append(",");
					
				}
			}
		}
		
	}
	
	@SuppressWarnings("unused")
	private static void write_to_file(List<Exp> examples, int percentage) throws IOException{
		
		for (Exp e : examples) {
			double[] content = new double[e.getData().length];
			content = e.getData();

			for (int i = 0; i < e.getData().length; i++) {
				switch (percentage) {
				case 2: {
					iris_2.append(String.valueOf(content[i])); iris_2.append(" "); break;
				}
				case 4: {
					iris_4.append(String.valueOf(content[i])); iris_4.append(" "); break;
				}
				case 6: {
					iris_6.append(String.valueOf(content[i])); iris_6.append(" "); break;
				}
				case 8: {
					iris_8.append(String.valueOf(content[i])); iris_8.append(" "); break;
				}
				case 10: {
					iris_10.append(String.valueOf(content[i])); iris_10.append(" "); break;
				}
				case 12: {
					iris_12.append(String.valueOf(content[i])); iris_12.append(" "); break;
				}
				case 14: {
					iris_14.append(String.valueOf(content[i])); iris_14.append(" "); break;
				}
				case 16: {
					iris_16.append(String.valueOf(content[i])); iris_16.append(" "); break;
				}
				case 18: {
					iris_18.append(String.valueOf(content[i])); iris_18.append(" "); break;
				}
				case 20: {
					iris_20.append(String.valueOf(content[i])); iris_20.append(" "); break;
				}
				default: // no corruption
				{
					if (percentage <= 20){
					iris_0.append(String.valueOf(content[i])); iris_0.append(" ");} break;
				}

				}

				// APPEND THE TARGETS TO EACH EXAMPLE IN THE FILE (LINE BY LINE)
				if (i == e.getData().length - 1) {
					switch (percentage) {
					case 2: {
						iris_2.append(String.valueOf(replace(e.getTarget())));
						iris_2.append("\n");
						break;
					}
					case 4: {
						iris_4.append(String.valueOf(replace(e.getTarget())));
						iris_4.append("\n");
						break;
					}
					case 6: {
						iris_6.append(String.valueOf(replace(e.getTarget())));
						iris_6.append("\n");
						break;
					}
					case 8: {
						iris_8.append(String.valueOf(replace(e.getTarget())));
						iris_8.append("\n");
						break;
					}
					case 10: {
						iris_10.append(String.valueOf(replace(e.getTarget())));
						iris_10.append("\n");
						break;
					}
					case 12: {
						iris_12.append(String.valueOf(replace(e.getTarget())));
						iris_12.append("\n");
						break;
					}
					case 14: {
						iris_14.append(String.valueOf(replace(e.getTarget())));
						iris_14.append("\n");
						break;
					}
					case 16: {
						iris_16.append(String.valueOf(replace(e.getTarget())));
						iris_16.append("\n");
						break;
					}
					case 18: {
						iris_18.append(String.valueOf(replace(e.getTarget())));
						iris_18.append("\n");
						break;
					}
					case 20: {
						iris_20.append(String.valueOf(replace(e.getTarget())));
						iris_20.append("\n");
						break;
					}
					default: {
						if (percentage <= 20){
						iris_0.append(String.valueOf(replace(e.getTarget())));
						iris_0.append("\n");
						}
						break;
					}
					}
				}

			}
			
			

		}
		
		switch (percentage){
			case 2: iris_2.close(); break;
			case 4: iris_4.close(); break;
			case 6: iris_6.close(); break;
			case 8: iris_8.close(); break;
			case 10: iris_10.close(); break;
			case 12: iris_12.close(); break;
			case 14: iris_14.close(); break;
			case 16: iris_16.close(); break;
			case 18: iris_18.close(); break;
			case 20: iris_20.close(); break;
			default: iris_0.close(); break;
		}
	}
	
	private static String replace(double[] argin) {
		
		if (argin[0] == 1){
			return "Iris-setosa";
		}else if (argin[1] == 1){
			return "Iris-versicolor";
		}else if (argin[2] == 1){
			return "Iris-virginica";
		}else{
			System.out.println("no good input");
			return null;
		}
		
	}



	private static void write_to_file(double[][] content) throws IOException {
		
		for (int i =0 ; i < content[0].length; i++){
			hidden_values_exp1.append(String.valueOf(content[0][i]));
			if (i < content[0].length - 1){
				hidden_values_exp1.append(",");
				
			}
		}
		
		for (int i =0 ; i < content[1].length; i++){
			hidden_values_exp2.append(String.valueOf(content[1][i]));
			if (i < content[1].length - 1){
				hidden_values_exp2.append(",");
				
			}
		}
		
		for (int i =0 ; i < content[2].length; i++){
			hidden_values_exp3.append(String.valueOf(content[2][i]));
			if (i < content[2].length - 1){
				hidden_values_exp3.append(",");
				
			}
		}
		
		for (int i =0 ; i < content[3].length; i++){
			hidden_values_exp4.append(String.valueOf(content[3][i]));
			if (i < content[3].length - 1){
				hidden_values_exp4.append(",");
				
			}
		}
		
		for (int i =0 ; i < content[4].length; i++){
			hidden_values_exp5.append(String.valueOf(content[4][i]));
			if (i < content[4].length - 1){
				hidden_values_exp5.append(",");
				
			}
		}
		for (int i =0 ; i < content[5].length; i++){
			hidden_values_exp6.append(String.valueOf(content[5][i]));
			if (i < content[5].length - 1){
				hidden_values_exp6.append(",");
				
			}
		}
		for (int i =0 ; i < content[6].length; i++){
			hidden_values_exp7.append(String.valueOf(content[6][i]));
			if (i < content[6].length - 1){
				hidden_values_exp7.append(",");
				
			}
		}
		for (int i =0 ; i < content[7].length; i++){
			hidden_values_exp8.append(String.valueOf(content[7][i]));
			if (i < content[7].length - 1){
				hidden_values_exp8.append(",");
				
			}
		}
	
	}

	private static void setAttrs_size(int i) {
		attrs_size = i;

	}
	
	private static List<Exp> readExamples(String filepath) throws FileNotFoundException, IOException {
		// reads input txt file line by line
		// every line contains a row of a value for all possible attributes plus a target class at the end.
		// Exp is a class containing only one example. (line)
		// we have an array of Exp comprises with the entire training dataset.

		List<Exp> examples = new ArrayList<Exp>();
		
		try (BufferedReader reader = new BufferedReader(new FileReader(filepath))) {
			String line = reader.readLine();
			
			while (line != null){
				
				// create an example
				Exp e = new Exp(line, attrs, attrs_orig, classes);
				
				// add an example to the list
				examples.add(e);
				
				line = reader.readLine();
			}
			
		}

		return examples;
	}
	
	private static void readAttributes(String filepath) throws FileNotFoundException, IOException {
		// reads input txt file line by line
		// There are 4 attributes: Outlook, Temperature, Humidity, Wind, plus target PlayTennis
		
		// Outlook Sunny Overcast Rain
		// Temperature Hot Mild Cool
		// Humidity High Normal
		// Wind Weak Strong

		// PlayTennis Yes No
		
		attrs = new String[attrs_size];
		
		try (BufferedReader reader = new BufferedReader(new FileReader(filepath))) {
			String line = reader.readLine();
			int counter = 0;
			
			// read the first 4 lines
			while (!line.isEmpty()){
				
				String[] tmp = line.split(" ");
				ArrayList<String> tmp_vals = new ArrayList<String>();
				
				for (int i = 1; i < tmp.length; i++){
					tmp_vals.add(tmp[i]);
				}
				
				attr_vals.put(tmp[0], tmp_vals);
				
				attrs[counter] = tmp[0];
				
				counter++;
				
				line = reader.readLine();
			}
			
			// For assigning possible target values
			// I hard coded this part.
			if (attrs.length == 8) { // identity
				classes = new double[attrs.length];
				return;
			}
			
			// now read PlayTennis Yes No
			line = reader.readLine();
			String[] tmp = line.split(" ");
			if (tmp.length == 3) { // tennis
				classes = new double[tmp.length - 2];
			} else if (tmp.length == 4){ // iris
				classes = new double[tmp.length - 1];
			}

			// classes contains 0 or 1 for tennis dataset meanning yes or no
			// classes contains [1, 0, 0] for setosa
			//                  [0, 1, 0] for versicolor
			//               or [0, 0, 1] for virginica
			// for iris dataset.
			// classes contains [ 0/1 0/1 0/1 0/1 0/1 0/1 0/1 0/1 ] for identity
			
		}
		
	}
	
	private static void preprocess_Tennis(List<Exp> examples_train) {
		
		ArrayList<String> attrs_tmp = new ArrayList<String>();
		
		
		// CONVERT TENNIS EXAMPLES TO 1-OF-N REPRESENTATION
		for (int i = 0; i < attrs_orig.length; i++){
			 Iterator<String> itr = attr_vals.get(attrs_orig[i]).iterator(); // outlook
			 
			 while (itr.hasNext()){ // sunny, overcast, rain
				 attrs_tmp.add(attrs_orig[i]+"-"+itr.next());
			 }
	
		}
     	// ADD ALL NEWLY FORMED ATTRIBUTES TO OUR KNOWLEDGE
		updateAttributes(attrs_tmp);
		
	}

	private static void updateAttributes(ArrayList<String> argin) {
		
		setAttrs_size(argin.size());
		attrs = new String[attrs_size];
		
		for (int i = 0; i < argin.size(); i++){
			attrs[i] = argin.get(i);
		}
		
	}
	
	@SuppressWarnings({ "rawtypes", "unchecked" })
	private static Pair<List<Exp>, List<Exp>> split_into_train_val(List<Exp> examples, double ratio) {
		// split the training data into two sections
		// the hold-out (10%) set and the training (90%) set itself
		
		int sz = examples.size(); int thr = (int) (sz * ratio);
		List<Exp> train_set = new ArrayList<Exp>();
		List<Exp> validation_set = new ArrayList<Exp>();
		
		if (thr != 0){
			for (int i = 0; i < thr; i++){
				train_set.add(examples.get(i));
			}
			
			for (int i = thr; i < sz; i++){
				validation_set.add(examples.get(i));
			}
			Pair<List<Exp>, List<Exp>> output = new Pair<List<Exp>, List<Exp>>(train_set, validation_set);
			return output;
		}else{
			Pair<List<Exp>, List<Exp>> output = new Pair<List<Exp>, List<Exp>>(examples, examples);
			return output;
		}
		
		
	}
	
	private static double getRatio(){
		return ratio;
	}
	
	private static void setRatio(double r){
		ratio = 1 - r;
	}
	
}
