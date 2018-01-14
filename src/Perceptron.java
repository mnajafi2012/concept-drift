/**
 * 
 * @author Maryam Najafi, mnajafi2012@my.fit.edu
 *
 * Mar 4, 2017
 * Course:  CSE 5693, Fall 2017
 * Project: HW3, Artificial Neural Networks
 * 
 * @category
 * This class indicates only one perceptron (neuron).
 * A perceptron has input weights (by convention in-going weights to a neuron
 * belong to the hidden layer in which the neuron is.)
 * A perceptron owns a summation part named SIGMA. SIGMA is we output net output.
 * A perceptron has an activation signal alpha.
 * Since the activation function and in-going weights are the same for all the neurons
 * I put them in the Layer class; they are inherited from the hidden layer.
 * Perceptrons of a hidden layer share a common activation function and weights.
 * 
 */
public class Perceptron {
	
	private final double alpha = .5; // Perceptron's activation signal
	private double error;
	//private double y; // net; neuron's output before threshold
	//private double z; // neuron's output after threshold
	
	
	@Override
	protected Perceptron clone (){
		Perceptron new_perc = new Perceptron();
		/*
		new_perc.setY (this.getY());
		new_perc.setZ(this.getZ());
		*/
		return new_perc;
	}
	

    protected double getAlpha (){
		return this.alpha;
	}
    
    protected void setError(double e){
    	this.error = e;
    }
    
    protected double getError(){
    	return this.error;
    }
	
    /*
	protected void setY (double argin){
		this.y = argin;
	}
	
	protected double getY (){
		return this.y;
	}
	
	protected void setZ(double argin){
		this.z = argin;
	}
	
	protected double getZ (){
		return this.z;
	}
	
	 */
}
