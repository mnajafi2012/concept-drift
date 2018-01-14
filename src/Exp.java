import java.util.Arrays;

/**
 * 
 * @author Maryam Najafi, mnajafi2012@my.fit.edu
 *
 * @date
 * Mar 4, 2017
 * 
 * Course:  CSE 5693, Fall 2017
 * Project: HW3, Artificial Neural Networks
 * 
 * @category
 * Exp is a class containing only one example; a line from the input txt file.
 * Each line contains a row of a value for all possible attributes plus a target class(or a vector of output) at the end.
 * Identity function dataset has 8 attributes, each of which has 1 possible spot for its value. (0 or 1)
 * Identity function dataset has 8 outputs, that is the same input vector.
 */
public class Exp {

	private double[] data; // e.g. [0 1 0 0 0 0 0 0]
	private double[] target; // e.g [0 1 0 0 0 0 0 0]
	
	// constructor
	// argin could be for Identity dataset is like:
	//           [0 1 0 0 0 0 0 0   0 1 0 0 0 0 0 0]
	// argin for tennis dataset is like:
	// [Sunny Hot High Weak No] that we change into a binary code as:
	// [1001001001 0]
	Exp (String argin, String[] attrs, String[] attrs_orig, double[] classes){	 
		// form a 1-of-n representation of an example
		// for Tennis dataset
		
		// SET INPUT
		
		this.data = new double[attrs.length];
		this.target = new double[classes.length];

		String[] tmp = argin.split(" ");
		for (int i = 0; i < attrs_orig.length; i++){

			String temp = attrs_orig[i]; // outlook
			String dummy = temp + "-" + tmp[i]; // outlook-sunny
			
			int idx = Arrays.asList(attrs).indexOf(dummy); // 0
			if (idx == -1){ // identity and iris
				// SET DATA
				this.setData(i, Double.valueOf(tmp[i]));
				// SET OUTPUT
				if (classes.length >= 8){ // for identity
					this.setTarget(Double.valueOf(tmp[i]), i);
				}else if (classes.length == 3){ // for iris
					if (i == 0){ // do it only once
						this.setTarget(tmp[tmp.length-1], classes);
					}
				}else if (classes.length == 1){ // for SINE1 - non-stationary data set
					this.setTarget(tmp[tmp.length-1], classes);
				}
				
			}else { // tennis
				// SET DATA
				this.setData(idx, 1);
				if (i == 0) { // do it only once
					// SET OUTPUT
					this.setTarget(tmp[tmp.length-1], classes);
				}
				
			}
		}
		
	}
	
	protected void setTarget(double val, int idx){
		this.target[idx] = val;
	}
	
	protected void setTarget(double[] argin){
		for (int i = 0; i < argin.length; i++){
			this.target[i] = argin[i];
		}
	}
	
	protected void setTarget (String argin, double[] classes){
		this.target = replace(argin, classes);
	}
	
	protected double[] getTarget(){
		return this.target;
	}
	
	protected void setData (int idx, double val){
		this.data[idx] = val;
	}
	
	private double[] replace(String argin, double[] classes){
		// yes or no  will be replaces with 1 or 0
		// setosa, versicolor, virginica with 100, 010, 001
		double[] tmp = new double[classes.length];
		
		if (tmp.length == 3) { // iris
			switch (argin){
				case "Iris-setosa": tmp[0] = 1; tmp[1] = 0; tmp[2] = 0; break;
				case "Iris-versicolor": tmp[0] = 0; tmp[1] = 1; tmp[2] = 0; break;
				case "Iris-virginica": tmp[0] = 0; tmp[1] = 0; tmp[2] = 1; break;
				default: System.out.print("no valid targets!");
			}
		}else if (tmp.length == 1){ // tennis or SINE1
			
			if (Integer.valueOf(argin) instanceof Integer){ // SINE1
				tmp[0] = Integer.valueOf(argin);
			}else{ // tennis
				tmp[0] = argin.equals("Yes")?1:0; 
			}
			
		}else if (tmp.length == 8) { // identity
			for (int i = 0; i < classes.length; i++){
				tmp[i] = classes[i];
			}
		}

		return tmp;
		
	}
	
	protected double[] getData(){
		return this.data;
	}
	
	
	protected void setData(int[] d){
		for (int i = 0; i < d.length; i++){
			this.data[i] = d[i];
		}
	}
	
	protected double get(int idx){
		// takes the index of data and returns the particular element of the string array
		return this.getData()[idx];
	}
	
	protected int size(){
		return this.getData().length;
	}
	
}
