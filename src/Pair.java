/**
 * 
 * @author Maryam Najafi, mnajafi2012@my.fit.edu
 *
 * Mar 6, 2017
 * 
 * Course:  CSE 5693, Fall 2017
 * Project: HW3, ANN
 */
public class Pair<F , S> {
	
	private F first;
	private S second;
	
	Pair(F first, S second){
		this.first = first;
		this.second = second;
		
	}
	
	protected F getfirst(){
		return this.first;
	}
	protected S getsecond() {
		return this.second;
	}

}
