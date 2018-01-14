 * Author: Maryam Najafi
 * Mar 10, 2017
 * Course:  CSE 5693, Fall 2017
 * Project: HW3, Artificial Neural Networks
 * The following is the manual to run the ANN algorithm.
---
java Main [OPTIONS]

OPTIONS:
java Main [dataset] [# of hidden units] [validation]

VALUES:
[dataset]:           identity/tennis/iris/irisnoisy
[# of hidden units]: 1/ 2/ 3/ 4/ 5.. any discrete numbers greater than 1
[validation]:        [0 _ .7] any continuous number in [0 _ .7] interval.
                     Usually I select 20% of data as for the validation and the rest for the training set.
                     Thus .2 would be assigned for the value of this option.

---

Examples:

Using command line arguments:
Open a terminal and...

java Main identity 3
Allows you to run identity function auto-encoder and learn the weights after 5000 epochs. The hidden unit encoding for all 8 examples will be recorded in *.csv files. (e.g. identityhidden_values_exp01000000_4units.csv" for the input 01000000 with a 4-unit ANN).

java Main tennis 4
Allows you to run tennis data set with one hidden layer of 4 units. The weights will be stored in "tennisweights_4units.csv" under the "results" folder.

java Main iris 3
Allows you to run iris data set and learn weights using a 3-hidden-unit network. The weights will be stored in "irisweights_3units.csv" under "results".

java Main irisnoisy 3
Allows you to train the network for the iris data with NO validation set for 1000 iterations. A 3-hidden-unit ANN is chosen for this example. The maximum limit for injected noise is 20% of the train set. The learned weights are recorded in "irisnoisyweights_3units_NOVal.csv". This file is saved in the "results" folder.

java Main irisnoisy 3 .2
Allows you to train the network for the iris data WITH validation set for 1000 iterations. The proportion of the validation set to the training set is 2:8. The network has one hidden layer of 3 hidden units. The learned weights are recorded in "irisnoisyweights_3units_Val.csv" in the "results" folder.

---

Required files:
Main.java
Perceptron.java
Layer.java
Network.java
Exp.java
Pair.java
shuffledList.txt
identity-attr.txt
identity-train.txt
tennis-attr.txt
tennis-train.txt
tennis-test.txt
iris-attr.txt
iris-train.txt
iris-test.txt
