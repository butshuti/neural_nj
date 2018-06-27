package learners.perceptrons;

import java.io.Serializable;

/**
 * This interface models an activation function.
 */
public abstract class ActivationFunction implements Serializable{

    private static final long serialVersionUID = 4301596837L;

    /**
     * Sigmoid activation function.
     */
    public static ActivationFunction SIGMOID = new ActivationFunction() {
        private static final long serialVersionUID = 43015968371L;
        @Override
        public double computeOutput(double input) {
            return 1.0 / (1.0 + Math.exp(-input));
        }

        @Override
        public double computeGradient(double fOfX) {
            return fOfX * (1.0 - fOfX);
        }
    };

    /**
     * tahn activation function
     */
    public static ActivationFunction TANH = new ActivationFunction() {
        private static final long serialVersionUID = 43015968372L;
        @Override
        public double computeOutput(double input) {
            return (Math.exp(input) - Math.exp(-input))/(Math.exp(input) + Math.exp(-input));
        }

        @Override
        public double computeGradient(double fOfX) {
            return 1 - (fOfX*fOfX);
        }
    };

    /**
     * Rectified linear unit (ReLU) activation function
     */
    public static ActivationFunction RELU = new ActivationFunction() {
        private static final long serialVersionUID = 43015968373L;
        @Override
        public double computeOutput(double input) {
            return Math.log(1 + Math.exp(input));
        }

        @Override
        public double computeGradient(double fOfX) {
            return 1.0 / (1 + Math.exp(-fOfX));
        }
    };

    /**
     * Gaussian activation function
     */
    public static ActivationFunction GAUSSIAN = new ActivationFunction() {
        private static final long serialVersionUID = 43015968374L;
        @Override
        public double computeOutput(double input) {
            return Math.exp(-(input*input));
        }

        @Override
        public double computeGradient(double fOfX) {
            return -2 * Math.exp(-(fOfX*fOfX));
        }
    };

    /**
     * Compute the node's output based on its value.
     * @param input the node's value
     * @return the node's output
     */
    public abstract double computeOutput(double input);

    /**
     * Compute the gradient of a given function
     * @param x the function
     * @return the gradient
     */
    public abstract double computeGradient(double x);
}
