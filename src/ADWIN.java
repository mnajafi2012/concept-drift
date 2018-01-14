import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

/**
 * 
 * @author Maryam Najafi, mnajafi2012@my.fit.edu
 *
 * Apr 16, 2017
 * 
 * ADWIN stands for Adaptive Windowing
 * Reference: Learning from Time-Changing Data with Adaptive Windowing
 * Authors: Albert Bifet Ricard Gavaldà
 * 
 * Problem setting:
 * The inputs to ADWIN is the confidence interval which is a decimal number [0 - 1].
 * A sequence of positive infinite numbers that are shown as x's as the input data point.
 * Each x_t comes at the time t, with an unknown distribution D_t.
 * Each x is assumed to be normalized and located between 0 and 1 region in the Cartesian coordinate.
 * 
 * The algorithm keeps tracking the expected averages of two sub-windows.
 * Once the distribution changes in the data points, these two averages get far enough from each other.
 * A threshold is assigned to trigger the concept drift detection by monitoring the averages difference.
 * 
 * eps_cut:
 * (no Hoeffding bound due to its overestimation of large deviations for distributions of small variance)
 * Hoeffding: eps_cut = sqrt[(1/2m) . ln (4/delta_prime)]
 * alternatively eps_cut = sqrt[(2/m) . sigma^2 . ln(2/delta_prime)] + [(2/3m) . ln (2/delta_prime)] 
 * 
 */
public class ADWIN {
	
	// parent window
	private Queue<Exp> W;
	
	// sub-windows
	private Queue<Exp> W0, W1;
	
	// W0, W1 and W sizes
	private int n0 = 30, n1 = 30; final int n = n0 + n1;
	
	// observed/estimated arithmetic means of W0 and W1 
	private double[] mu_hat_W0, mu_hat_W1;
	
	// expected arithmetic means of W0 and W1
	private double[] mu_W0, mu_W1;
	
	// the Harmonic mean of n0 and n1
	private double m;
	
	// confidence value (user-defined) - between [0 and 1]
	private final double delta = .95;
	
	// confidence value to avoid problems with multiple hypothesis testing
	private final double delta_prime = delta/(double)Math.log(n);
	
	// observed variance of elements in window W
	private double[] variance;
	
	// threshold between the means of two sub-windows
	private double[] eps_cut;
	
	// feature dimension
	private int dim;
	
	// batch containing either W or W0
	private List<Exp> batch = new ArrayList<Exp>();
	
	private static int capacity;
	
	protected boolean y_or_n = false;
	
	Queue<Exp> W_context;
	
	// constructor
	public ADWIN(int n0, int n1){
		// init.
		W = new LinkedList<Exp>();
		W0 = new LinkedList<Exp>();
		W1 = new LinkedList<Exp>();
		W_context = new LinkedList<Exp>();
		setMu_hat_W0 (new double[dim]);
		setMu_hat_W1 (new double[dim]);
		eps_cut = new double[dim];
		this.set_n0(n0);
		this.set_n1(n1);
		setHarmonic_mean(n0, n1);
	
	}


	private void set_n0(int argin) {
		this.n0 = argin;	
	}

	private void set_n1(int argin) {
		this.n1 = argin;	
	}
	
	public void addElement(Exp exp){
		capacity++;
		W.add(exp);
		y_or_n = false;
		
		// if W1 is filled up remove the last element and move it to W0
		if (W1.size() == this.get_n1()) {

			
			if (W0.size() < this.get_n0()) {
				W0.add(W1.peek());
				W1.remove();
				W1.add(exp);

			} else {// shift to the left
				W0.remove();
				W0.add(W1.peek());
				W1.remove();
				W1.add(exp);
			}

			// if both sub-windows are filled
			if (W0.size() + W1.size() == this.get_n()) {
				update_statistics();
			}
		} else {
			W1.add(exp);
		} // end of if
		
		// streaming data point exp through windows from head (rightmost)
		if (W.size() >= this.get_n()) {
			
			if (W.size() > this.get_n()){
				W.poll();
			}
			
			// check if you should shrink the window into two.
			//y_or_n = (capacity%160 == 0)?true:false;
			//if (capacity == 120){
			//	System.out.println();
			//}
			
			update_statistics();
			y_or_n = check_to_shrink();
			
			if (y_or_n){
				// shrink and return W1 as for train
				W.clear(); W0.clear();W_context.clear();
				set_batch(W1);
				
				Iterator<Exp> itr = W1.iterator();
				while (itr.hasNext()){
					W.add(itr.next());
				}
				
			}else{
				// return examples of W0 as train set
				// add examples that are useful and belong to the current context
				//Iterator<Exp> itr = W0.iterator();
				//while (itr.hasNext()){
				//	W_context.add(itr.next());
				//}
				//set_batch(W_context);
				//////set_batch(W0);
				//W.poll();
				y_or_n = false;
			}
			
		}// end of if (outer)

	}
	public void addElement(Exp exp, String name_dataset){
		capacity++;
		W.add(exp);
		y_or_n = false;
		
		// if W1 is filled up remove the last element and move it to W0
		if (W1.size() == this.get_n1()) {

			
			if (W0.size() < this.get_n0()) {
				W0.add(W1.peek());
				W1.remove();
				W1.add(exp);

			} else {// shift to the left
				W0.remove();
				W0.add(W1.peek());
				W1.remove();
				W1.add(exp);
			}

			// if both sub-windows are filled
			if (W0.size() + W1.size() == this.get_n()) {
				update_statistics();
			}
		} else {
			W1.add(exp);
		} // end of if
		
		// streaming data point exp through windows from head (rightmost)
		if (W.size() >= this.get_n()) {
			
			if (W.size() > this.get_n()){
				W.poll();
			}
			
			// check if you should shrink the window into two.
			//y_or_n = (capacity%160 == 0)?true:false;
			//if (capacity == 120){
			//	System.out.println();
			//}
			
			update_statistics();
			y_or_n = check_to_shrink();
			
			y_or_n = (capacity%1000 == 0)?true:false;
			
			if (y_or_n){
				// shrink and return W1 as for train
				W.clear(); W0.clear();W_context.clear();
				set_batch(W1);
				
				Iterator<Exp> itr = W1.iterator();
				while (itr.hasNext()){
					W.add(itr.next());
				}
				
			}else{
				// return examples of W0 as train set
				// add examples that are useful and belong to the current context
				//Iterator<Exp> itr = W0.iterator();
				//while (itr.hasNext()){
				//	W_context.add(itr.next());
				//}
				//set_batch(W_context);
				//////set_batch(W0);
				//W.poll();
				y_or_n = false;
			}
			
		}// end of if (outer)

	}

	private void set_batch(Queue<Exp> win) {
		batch.clear();
		for (Exp exp: win){
			batch.add(exp);
		}
	}

	public List<Exp> get_batch(){
		return this.batch;
	}
	
	private boolean check_to_shrink() {
		
		// 1. calculate the norm of absolute value of the difference
		// between the observed mean of both sub-windows
		double[] diff = new double[dim];
		for (int i = 0; i < dim; i++) {
			diff[i] = Math.abs(mu_hat_W0[i] - mu_hat_W1[i]);
		}
		// norm for the difference
		double subwindows_norm = norm(diff);
		//System.out.printf("sub norm: %.3f ", subwindows_norm);
		
		// 2. calculate the norm of the threshold since it is a vector
		// L-2 norm of the threshold
		double eps_norm = norm(eps_cut);
		//System.out.printf("eps norm: %.3f%n", eps_norm);
		
		// 3. compare these two norms whether the former is less than the
		// threshold
		//if (subwindows_norm > eps_norm/2) {
			if (subwindows_norm > eps_norm) {
			return true;
		} else {
			return false;
		}
	}

	private double norm(double[] argin){
		double norm = .0;
		// L-2 norm of the input argument
		for (int i = 0; i < dim; i++){
			norm += Math.pow(argin[i], 2);
		}
		
		norm = Math.sqrt(norm);
		return norm;
	}

	private void update_statistics(){
		// if W0 is also filled up calculate the statistics
		double[] mu;

		// 1. W0
		mu = statistics(W0);
		setMu_hat_W0(mu);

		// 2. W1
		mu = statistics(W1);
		setMu_hat_W1(mu);

		// 3. SHRINKING THRESHOLD eps
		mu = statistics(W);
		set_eps(cal_eps(W, mu, m, delta_prime));
		
	}

	private void setMu_hat_W0(double[] mu) {
		this.mu_hat_W0 = mu;
		
	}
	
	private void setMu_hat_W1(double[] mu) {
		this.mu_hat_W1 = mu;
		
	}

	private double[] cal_eps(Queue<Exp> window, double[] mu, double m, double delta_prime) {
		// calculate the epsilon_cut at this moment
		// this is a threshold that determines the shrinking
		
		double[] var = cal_variance(window, mu);
		setVariance(var);
		
		double[] eps = new double[dim];
		
		/*for (int i = 0; i < dim; i++){
			eps[i] += Math.sqrt(((2 / m) * getVariance()[i] * Math.log(2 / delta_prime))) 
					+ ((2/(3 * m)) * Math.log(2/ delta_prime));
		}*/
		for (int i = 0; i < dim; i++){
			eps[i] += Math.sqrt(((1 / 2*m) * Math.log(2 / delta_prime)));
		}
		return eps;
	}
	
	
	private double[] cal_variance(Queue<Exp> window, double[] mu) {
		// calculates the variance of elements in W
		int N = window.size(); // number of data points
		
		double[] sum = new double[dim];
		
		Iterator<Exp> itr = window.iterator();
		
		while (itr.hasNext()){
			
			double[] datapoint = itr.next().getData();
			
			// over features
			double[] tmp = new double[dim];
			
			for (int i = 0; i < dim; i++){
				
				// L-2 Norm of datapoint and the mean
				tmp[i] = Math.pow(datapoint[i] - mu[i], 2);
			}
			
			for (int i = 0; i < dim; i++){
				// sum up norms
				sum[i] += tmp[i];
			}	
			
		} // end of while
		
		for (int i = 0; i <dim; i++){
			sum[i] = sum[i]/(N-1);
		}
		
		
		double[] variance = sum;
		
		return variance;
	}

	private void setVariance(double[] var) {
		// this is to assign the standard deviation of elements in W
		
		this.variance = var;
	}
	
	private double[] getVariance(){
		return this.variance;
	}

	/**
	 * 
	 * @param subwindow
	 * @return the observed mean average of the sub-window W0 or W1
	 */
	private double[] statistics(Exp[] subwindow) {
		
		// reserve the spot for multiple-feature data points
		double[] sum = new double[dim]; // 2 for SINE1
		double[] mu = new double[dim]; // 2 for SINE1
		double sz = subwindow.length;
		double[] features;
		
		for (Exp exp: subwindow){
			// for all features of the data example (x and y for SINE1)
			features = exp.getData();
			
			for (int i = 0; i < features.length; i++){
				sum[i] += features[i];
			}
		}
		
		for (int i = 0; i < sum.length; i++){
			mu[i] = sum[i]/sz;
		}
		
		return mu;
	}
	
	/**
	 * 
	 * @param window
	 * @return the arithmetic mean of window W (queue)
	 */
	private double[] statistics(Queue<Exp> window) {
		
		// reserve the spot for multiple-feature data points
		double[] sum = new double[dim]; // 2 for SINE1
		double[] mu = new double[dim]; // 2 for SINE1
		int sz = window.size();
		
		double[] features;
		
		Iterator<Exp> itr = window.iterator();
		
		while (itr.hasNext()){
			// for all features of one example
			features = itr.next().getData();
			
			for (int i = 0; i < features.length; i++){
				sum[i] += features[i];
			}
		}// end of while
		
		for ( int i = 0; i < sum.length; i++){
			mu[i] = sum[i] / sz;
		}
		
		return mu;
	}

	protected int get_n(){
		return this.n;
	}
	
	protected int get_n0(){
		return this.n0;
	}
	
	protected int get_n1(){
		return this.n1;
	}
	
	protected void setDim (int argin){
		this.dim = argin;
	}
	
	private int getDim (){
		return this.dim;
	}
	
	private void set_eps(double[] argin){
		this.eps_cut = argin;
	}
	
	private void setHarmonic_mean(int n0, int n1) {
		// m = [1 / (1/n0) + (1/n1)]
		double tmp = 1/ (double) this.get_n0();
		double tmp2 = 1/(double) this.get_n1();
		m = 1/ (tmp + tmp2);
	}
}
