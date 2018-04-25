package learners.perceptrons;

import java.util.Arrays;
import java.util.Random;

public class Perceptron extends NeuralNetwork {

    private double[][] weights; /*the network's current weights*/
    private double[] currentInputs; /*the pending input*/
    private double[] currentOutputs; /*the output from the last computation*/
    private double[] errors; /*the errors from the last computation*/
    private double[] deltas; /*the weight updates for the next adjustment*/
    private Random random;
    private String name;

    private int numInputs, numOutputs;
    private double threshold = -2;

    private ActivationFunction activationFunction;

    protected double output(int i) {return currentOutputs[i];}
    protected double delta(int i) {return deltas[i];}
    private double input(int i) {return currentInputs[i];}
    private double error(int i) {return errors[i];}

    protected void setError(int i, double error) {errors[i] = error;}

    public int numInputNodes() {return numInputs;}
    public int numOutputNodes() {return numOutputs;}
    public int threshold() {return numInputNodes();}

    /**
     * Construct a new perceptron.
     *
     * @param name the unit's name
     * @param numIn the number of input nodes
     * @param numOut the number of output nodes
     * @param activationFunction the activation function for this unit.
     */
    public Perceptron(String name, int numIn, int numOut, ActivationFunction activationFunction) {
        this.name = name;
        numInputs = numIn;
        numOutputs = numOut;
        this.activationFunction = activationFunction;
        random = new Random();
        random.setSeed(1);
        initialize();
        for (int i = 0; i < numOutputNodes(); ++i) {
            for (int j = 0; j < numInputNodes(); ++j) {
                weights[j][i] = (2 * random.nextDouble() - 1);
            }
            weights[threshold()][i] = 1;
        }
    }

    /**
     * Initialize network's internal state.
     */
    private void initialize() {
        weights = new double[numInputNodes()+1][numOutputNodes()];
        currentOutputs = new double[numOutputNodes()];
        errors = new double[numOutputNodes()];
        deltas = new double[numOutputNodes()];
        currentInputs = new double[numInputNodes()+1];
    }

    public double getWeightFromTo(int inputNode, int outputNode) {
        return weights[inputNode][outputNode];
    }

    protected void setActivationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    private void resetInputs(){
        for(int i = 0; i<numInputNodes(); ++i){
            currentInputs[i] = 0;
        }
    }

    /**
     * {@See {@link NeuralNetwork#compute(double[])}}
     * @param inputs the normalized instance
     * @return
     */
    @Override
    public double[] compute(double[] inputs) {
        System.arraycopy(inputs, 0, currentInputs, 0, inputs.length);
        for(int i=0; i<numOutputNodes(); ++i){
            double output = 0.0;
            for(int j=0; j<numInputNodes(); ++j){
                output += currentInputs[j] * weights[j][i];
            }
            output += threshold * weights[threshold()][i];
            currentOutputs[i] = activationFunction.computeOutput(output);
        }
        double ret[] = new double[currentOutputs.length];
        System.arraycopy(currentOutputs, 0, ret, 0, ret.length);
        return ret;
    }

    /**
     * Update the network's weights.
     *
     * @param rate the current learning rate
     */
    public void updateWeights(double rate) {
        for(int i = 0; i<numOutputNodes(); ++i){
            deltas[i] = rate * error(i) * activationFunction.computeGradient(output(i));
            for(int j = 0; j<numInputNodes(); ++j){
                weights[j][i] += deltas[i] * input(j);
            }
            weights[threshold()][i] += deltas[i] * threshold;
        }
        resetInputs();
    }

    /**
     * Iterate with one training episode.
     *
     * @param inputs an input feature vector
     * @param targets target value corresponding to the input
     * @param rate the current learning rate
     */
    @Override
    public void train(double[] inputs, double[] targets, double rate) {
        compute(inputs);
        for (int i = 0; i < numOutputNodes(); ++i) {
            setError(i, targets[i] - output(i));
        }
        updateWeights(rate);
    }

    private String printR(double[][] arr){
        String ret = "{";
        for(double[] r : arr){
            ret += Arrays.toString(r) + ", ";
        }
        ret += "}";
        return ret;
    }

    public void printErrors(){
        System.out.println(name + ": " + printR(weights) + "; => " + Arrays.toString(currentInputs));
    }
}
