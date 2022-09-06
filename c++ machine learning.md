# C++ Machine Learning

## Introduction
Following this [tutorial](https://www.section.io/engineering-education/an-introduction-to-machine-learning-using-c++/), we implement linear regression to predict integers:

```c++
#include <iostream>
#include <vector>
#include <algorithm> //for std::sort

bool minAbsValue(double a, double b){
    double a1 = abs(a-0); 
    double b1 = abs(b-0);
    return a1<b1;
}

int main(){
    //training values
    double x[] = {1,2,3,4,5};
    double y[] = {1,3,3,2,5};
    std::vector<double> error; //for storing error values
    double devi; //for calculating error on each stage
    double b0 = 0; double b1 = 0; //initialize values
    double learnRate = .01; //initialize learn rate

    //training phase
    for(int i = 0; i < 20; i++){ //20 because there are  values and 4 epochs needed
        int index = i % 5; //for accessing index after each epoch
        double p = b0 + b1 * x[index]; //calculating prediction
        devi = p - y[index]; //calculating error
        b0 = b0 - learnRate * devi; //updating b0
        b1 = b1 - learnRate * devi * x[index]; //updating b1
        //print values after every update
        std::cout << "b0= " << b0 << " b1= " << b1 << ' ' << "error= " << devi << std::endl;
        error.push_back(devi); 
    }
    std::sort(error.begin(), error.end(), minAbsValue);
    std::cout << "Optimal end values are: b0= " << b0 << " b1 = " << b1 << " error= " << error[0] << std::endl; 

    //testing phase
    std::cout << "Enter a test x value: ";
    double test;
    std::cin >> test;
    double pred = b0 + b1 * test;
    std::cout << "\nThe value predicted by the model= " << pred << std::endl;

    return 0;
}
```
