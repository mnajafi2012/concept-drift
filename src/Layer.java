import java.util.ArrayList;
import java.util.Random;

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
 * Since the activation function and in-going weights are the same for all the neurons
 * I put them in the Layer class; they are inherited from the hidden layer.
 * Perceptrons of a hidden layer share a common activation function and weights.
 */
public class Layer {

	private ArrayList<Perceptron> P;
	private double[][] weights;
	private double[] w_bias;
	private String activationFunc = "sigmoid"; // and "sign"
	private double[] Z; // all neuron's outputs (after threshold)
	private double[] net; // all neuron's outputs (before threshold)
	private int size;
	private static Random rnd = new Random();
	private String type; // hidden or output layer

	Layer(String argin) {

		this.P = new ArrayList<Perceptron>();
		this.type = argin;
		this.weights = new double[0][0];
		this.w_bias = new double[0];
		this.Z = new double[this.size()];
		this.net = new double[this.size()];

	}

	/**
	 * 
	 * @param e
	 * @return a vec. of neuron's outputs before threshold
	 */
	protected double[] summation(double[] x) {

		// FOR EACH PERCEPTRON
		for (int p = 0; p < this.size(); p++) {
			double sum = .0;
			for (int i = 0; i < x.length; i++) {

				double[] unit_p_weights = this.getWeights()[p];

				sum += ((double) x[i] * unit_p_weights[i]);

			}

			sum += this.getw_bias(p);

			// update net with the sum of unit p
			this.setnet(p, sum);

		}

		return this.net;
	}
	
	protected double[] activation() {
		double[] tmp = new double[this.size()];

		// FOR EACH PERCEPTRON
		for (int p = 0; p < this.size(); p++) {

			double alpha = this.get(0).getAlpha();

			tmp[p] = (g(this.getnet(p), alpha));

		}

		this.setZ(tmp);

		return this.Z;
	}
	
	
	/**
	 * @category
	 * g(.) is the activation function of activation signal Y
	 * 
	 * @param net
	 * @return the output Z, after threshold
	 */
	private double g(double x, double alpha) {

		double tmp = Double.MIN_NORMAL;
		// ITERATE OVER EACH BEFORE-THRESHOLD OUTPUT IN net
		// for (int i = 0; i < net.length; i++){
		// double net_i = net[i];
		switch (activationFunc.toLowerCase()) {
		case "sign": {
			tmp = x > alpha ? 1 : 0;
			// System.out.println("Here goes the sgn function.");
			break;
		}
		case "sigmoid": {
			tmp = 1 / (1 + Math.pow(Math.E, -x));
			// System.out.println("Here goes the sigmoid g(.)");
			break;
		}
		default: {
			System.out.println("run hyperbolic func.");
			break;
		}
		}
		// }

		return tmp;
	}
	
	/**
	 * For the output layer we use: <p>
	 * error_k = g'(.)*∂(E)/∂(o) = o_k (1 - o_k) (t_k - o_k)<p>
	 * For the hidden layer we use:
	 * error = g'(.)*SUM(w_kh * error_k)<p>
	 * Here the input to the activation function is the sigmoid func.<p>
	 * g(u) = 1/[1 + e^(-alpha * u)]; g'(u) = alpha*g(u)*[1-g(u)] 
	 * @param e
	 * @return a vector of layer's errors
	 */
	protected double[] cal_delta_output(Exp e) {

		// IF THE LAYER FOR WHICH THE ERRORS ARE BEING CALCULATED IS THE OUTPUT
		// LAYER
		// ... IS THE HIDDEN LAYER.

		double[] delta_k = new double[P.size()];
		int h;

		// ITERATE OVER EACH PERCEPTRON IN THIS LAYER
		for (int k = 0; k < P.size(); k++) {
			delta_k[k] = cal_error_output(e.getTarget()[k], this.getZ(k));
			P.get(k).setError(delta_k[k]);
		}
		return delta_k;
	}

	private double cal_error_output(double t_k, double o_k) {

		double error = o_k * (1 - o_k) * (t_k - o_k);

		// check for divergence
		assert error < Double.MAX_EXPONENT : "Divergence!";

		return error;
	}

	/**
	 * 
	 * @param w_out_l
	 *            the weights from hidden to output layer
	 * @param delta_k
	 *            the errors of the output layer
	 * @return the errors of the hidden layer
	 */
	protected double[] cal_delta_hidden(double[][] w_out_l, double[] delta_k) {

		// IF THE LAYER FOR WHICH THE ERRORS ARE BEING CALCULATED IS THE OUTPUT
		// LAYER
		// ... IS THE HIDDEN LAYER.

		double[] delta_h = new double[P.size()];

		// ITERATE OVER EACH PERCEPTRON IN THIS LAYER
		for (int h = 0; h < P.size(); h++) {
			delta_h[h] = cal_error_hidden(this.getZ()[h], h, w_out_l, delta_k);
			P.get(h).setError(delta_h[h]);
		}
		return delta_h;
	}

	protected double cal_error_hidden(double o_h, int h, double[][] w, double[] delta_k) {

		double sum = .0;
		for (int k = 0; k < delta_k.length; k++) {
			sum += (w[k][h] * delta_k[k]);
		}

		// add bias
		// for (int k = 0; k < delta_k.length; k++){
		// sum += this.getw_bias(h);
		// }

		double error = o_h * (1 - o_h) * sum;

		// check for divergence
		assert error < Double.MAX_EXPONENT : "Divergence!";

		return error;
	}

	protected void setSize(int sz) {
		this.size = sz;
	}

	protected void add(Perceptron p) {
		this.setSize(this.size() + 1);
		this.P.add(p.clone());
	}
	
	/**
	 * 
	 * @return the number of hidden units
	 */
	protected int size() {
		return this.size;
	}

	protected void init(int num_input, int num_hid_unit, double limit) {

		// INIALIZE WEIGHTS WITH RANDOM NUMBERS
		rnd.setSeed((long) .0);
		// set random initial weights to the layer weights [-.05, .05]
		double min = -limit, max = limit;
		double range = max - min;

		assert num_input != 0;

		this.weights = new double[this.size()][num_input];
		this.w_bias = new double[this.size()];

		for (int i = 0; i < this.weights.length; i++) {
			for (int j = 0; j < this.weights[i].length; j++) {
				this.weights[i][j] = min + rnd.nextDouble() % range;
				//this.weights[i][j] = .01;
			}

		}

		// BIAS weight
		for (int i = 0; i < this.w_bias.length; i++) {
			this.w_bias[i] = min + rnd.nextDouble() % range;
			//this.w_bias[i] = .01;
		}

		// SET THE SIZES FOR OUTPUTS BEFORE AND AFTER THRESHOLD
		this.net = new double[num_hid_unit];
		this.Z = new double[num_hid_unit];

	}

	protected Perceptron get(int idx) {
		return this.P.get(idx);
	}

	@Override
	protected Layer clone() throws NullPointerException {
		Layer new_layer = new Layer(this.getType());

		new_layer.setWeights(this.getWeights());
		new_layer.setw_bias(this.getw_bias());

		// SET THE SIZES FOR OUTPUTS BEFORE AND AFTER THRESHOLD
		int num_hid_unit = this.size;
		new_layer.net = new double[num_hid_unit];
		new_layer.Z = new double[num_hid_unit];

		new_layer.setZ(this.getZ());
		new_layer.setnet(this.getnet());

		for (Perceptron p : this.P) {
			new_layer.add(p);
		}

		return new_layer;
	}

	protected void setZ(double[] argin) {
		for (int i = 0; i < argin.length; i++) {
			this.Z[i] = argin[i];
		}
	}

	protected void setZ(int idx, double argin) {
		this.Z[idx] = argin;
	}

	protected double[] getZ() {
		return this.Z;
	}

	protected double getZ(int idx) {
		return this.getZ()[idx];
	}

	protected void setnet(double[] argin) {
		for (int i = 0; i < argin.length; i++) {
			this.net[i] = argin[i];
		}
	}

	protected void setnet(int idx, double argin) {
		this.net[idx] = argin;
	}

	protected double[] getnet() {
		return this.net;
	}

	protected double getnet(int idx) {
		return this.getnet()[idx];
	}

	protected void setWeights(double[][] argin) {
		this.weights = new double[argin.length][argin[0].length];
		for (int i = 0; i < argin.length; i++) {
			for (int j = 0; j < argin[i].length; j++) {
				this.weights[i][j] = argin[i][j];
			}
		}
	}

	protected double[][] getWeights() {
		return this.weights;
	}

	protected void setw_bias(double[] argin) {
		this.w_bias = new double[argin.length];
		for (int i = 0; i < argin.length; i++) {
			this.w_bias[i] = argin[i];
		}
	}

	protected void setw_bias(int idx, double val) {
		this.w_bias[idx] = val;
	}

	protected double[] getw_bias() {
		return this.w_bias;
	}

	protected double getw_bias(int idx) {
		return this.w_bias[idx];
	}

	protected ArrayList<Perceptron> getEntry() {
		return this.P;
	}

	protected String getType() {
		return this.type;
	}

}
