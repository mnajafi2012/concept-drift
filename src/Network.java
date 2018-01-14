import java.util.*;

/**
 * 
 * @author Maryam Najafi, mnajafi2012@my.fit.edu
 *
 * Mar 5, 2017
 * 
 * Course:  CSE 5693, Fall 2017
 * Project: HW3, Artificial Neural Networks
 * 
 * @category
 * A Network is literally an ANN containing Layers.
 * I keep input instances and targets for updating the weights.
 * A line search algorithm has been employed here: 
 * Gradient-Descent Optimization algorithm.
 * 
 * The accuracy of the network after done with the training is computed here.
 * In this project we usually use only one hidden layer with multiple units.
 * 
 */
public class Network {

	protected ArrayList<Layer> L;
	public double[] O; // a vector of network's predicted outputs
	protected double acc; // accuracy
	
	Network (){
		this.L = new ArrayList<Layer>();
		this.setAcc(.0);
		
	}
	
	Network (int output_size){
		this.L = new ArrayList<Layer>();
		this.setAcc(.0);
		this.O = new double[output_size];
	}
	
	protected void add (Layer l){
		this.L.add(l.clone());
	}
	
	protected void setAcc (double argin){
		this.acc = argin;
	}
	
	/**
	 * FORWARD PROPAGATION ALGORITHM
	 * @param e an example - only data is being used.
	 * @return two sets of inputs to our hidden and output layers.
	 * The first set is the same input sample.
	 * The second set is the output from our hidden layer.
	 */
	protected ArrayList<double[]> forwardPropagation (Exp e){
		
		ArrayList<double[]> Xs = new ArrayList<double[]>();
		double[] Z = null;
		O = new double[O.length];
		
		// RECORD INPUT TO THIS LAYER
		Xs.add(e.getData().clone()); // input to hidden layer is recorded.
	
		// GO THROUGH LAYERS (we have only 1 hidden layer in this problem)
		for (int i = 0; i < L.size(); i++){ // forward
			if (L.get(i).getType().equalsIgnoreCase("output")){
				// Z: RETURN SUMMATION FOR EACH PERCEPTRON OF THE OUTPUT LAYER
				L.get(i).summation(Z);
				
				// O: RETURN OUTPUTS AFTER THRESHOLD FOR EACH PERCEPTRON OF iTH LAYER
				Z = L.get(i).activation();
				
				// SET THE OUTPUT OF THE NETWORK
				this.set_networkOutput(Z);
				
			}else{
				// Z: RETURN SUMMATION FOR EACH PERCEPTRON OF HIDDEN LAYERS
				L.get(i).summation(e.getData());
				
				// O: RETURN OUTPUTS AFTER THRESHOLD FOR EACH PERCEPTRON OF iTH LAYER
				Z = L.get(i).activation();
				
				// RECORD INPUTS TO THIS LAYER
				Xs.add(Z.clone()); // input to output layer is recorded.
				
			}
			
		}
			
		
		return Xs;
		
	}

	public void set_networkOutput(double[] z) {
		for (int i = 0 ; i < z.length; i++){
			O[i] = z[i];
		}
		
	}
	
	public double[] get_networkOutput(){
		
		for (int i = 0 ; i < this.O.length; i++){
			this.O[i] = this.O[i] < .5? 0 : 1;
		}
		return this.O;
	}

	/**
	 * BACKPROPAGATION ALGORITHM
	 * @param e an example - only target is being used.
	 * @return all errors (deltas) from all layers
	 */
	protected ArrayList<double[]> backpropagation (Exp e){
		// (propagate the errors backward)
		
		ArrayList<double[]> deltas = new ArrayList<double[]>();
		
		double[] delta_k = null, delta_h = null;
		double[][] weights_out_layer = L.get(L.size() - 1).getWeights();
		
		for (int i = L.size() - 1; i >= 0; i--){ // backward
			
			if (L.get(i).getType().equalsIgnoreCase("output")) {
				// CALCULATE OUTPUT LAYER ERRORS
				delta_k = L.get(i).cal_delta_output(e);
				
			}else{
				// CALCULATE HIDDEN LAYER ERRORS
				delta_h = L.get(i).cal_delta_hidden(weights_out_layer, delta_k);
			}
		}
		
		// RECORD ERRORS
		deltas.add(delta_h.clone());
		deltas.add(delta_k.clone());

		return deltas;
		
	}

	/**
	 * 
	 * @param etha learning rate (defined by user)
	 * @param getfirst (deltas) errors
	 * @param getsecond (inputs to each layer)
	 */
	protected void updateWeights(double etha, 
			ArrayList<double[]> deltas, ArrayList<double[]> Xs) {
		// UPDATE WEIGHTS FOR EACH LAYER
		for (int l = 0 ; l < L.size(); l++){
			// shallow copy
			double[][] W = L.get(l).getWeights();
			double[] w_bias = L.get(l).getw_bias();
			
			double[][] delta_w = delta_W(etha, deltas.get(l), Xs.get(l));
			//System.out.printf("%.3f, %.3f, %.3f, %.3f %n", 
					//delta_w[0][0], delta_w[0][1],delta_w[0][2],delta_w[0][3]);
			// update : shallow copied
			W = sum(L.get(l), W, delta_w);
			
			//L.get(l).setWeights(W);
		}
		
	}
	

	protected double[][] delta_W (double etha, double[] delta, double[] X){
		
		double[][] delta_w = new double[delta.length][X.length + 1]; 
		
		for (int j = 0; j < delta.length; j++){
			for (int i = 0; i < X.length; i++){
				delta_w[j][i] = etha * delta[j] * X[i];
			}
		}
		
		// for updating the bias weight
		for (int j = 0; j < delta.length; j++){
			delta_w[j][delta_w[j].length - 1] = etha * delta[j] * 1;
		}
		
		return delta_w;
	}
	private double[][] sum(Layer l, double[][] w, double[][] delta_w) {
		
		for (int i = 0; i < delta_w.length ; i++){
			for (int j = 0; j < delta_w[i].length - 1; j++){
				w[i][j] = w[i][j] + delta_w[i][j];
				
			}
		}
		
		// update bias weight
		for (int i = 0; i < l.getw_bias().length; i++){
			l.setw_bias(i, l.getw_bias(i) + delta_w[i][delta_w[i].length - 1]);
		}

		return w;
	}
	
	public void printWeights() {
		
		for (Layer l: this.L){
			System.out.print("\nLayer's weights:\n");
			for (double[] W: l.getWeights()){
				System.out.println();
				for (double w: W){
					System.out.printf("%.3f ", w);
				}
			}
		}
		
	}
	
	public double[] printWeights(int idx, int unit) {
		
		Layer l = L.get(idx);
			//System.out.print("\nLayer's weights:\n");
			double[] W = l.getWeights()[unit];
			System.out.println();
			for (double w: W){
				System.out.printf("%.3f,", w);
			}
			
			return W;
	}
	
	public double[] getWeights(int idx, int unit){
		Layer l = L.get(idx);
		double[] W = l.getWeights()[unit];
		
		double[] tmp = new double[W.length + 1]; // to add bias w
		
		for (int i = 0; i < W.length; i++){
			tmp[i] = W[i];
		}

		tmp[tmp.length-1] = l.getw_bias(idx);
		
		return tmp;
	}

	public double[] getLayerVals (int idx){
		
		Layer l = L.get(idx);
		double[] vals = l.getZ();
		
		return vals;
	}
	
	protected double[] SSE (double[] o, double[] t){
		double[] sum = new double[o.length];
		for (int i = 0 ; i < o.length; i ++){
			sum[i] += Math.pow((o[i] - t[i]), 2);
			System.out.println(o[i]);
		}
		
		return sum;
	}
	
	protected Network clone (){
		Network newNetwork = new Network(this.get_networkOutput().length);
		
		// copy layers
		for (Layer l : this.L){
			Layer tmp = l.clone();
			newNetwork.add(tmp);
		}
		
		// copy O
		double [] newO = new double [this.O.length];
		for (int i = 0 ; i < this.O.length; i++){
			newO[i] = this.O[i];
		}
		
		newNetwork.set_networkOutput(newO);
		
		// copy acc
		double new_acc = this.acc;
		newNetwork.setAcc(new_acc);
		
		return newNetwork;
		
	}

}
