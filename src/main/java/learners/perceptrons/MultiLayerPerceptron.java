package learners.perceptrons;

import java.io.Serializable;

public class MultiLayerPerceptron extends NeuralNetwork implements Serializable {
    private Perceptron[] layers;


    /**
     * Construct a new multilayer perceptron
     *
     * @param networkShape the perceptron's shape, in term of its composing layers:
     *                     {N1, N2,...,Nl} means l-1 layers, where Nj and Nj+1 are respectively the number of input and output nodes in layer Nj.
     * @param activationFunctions the alternating activation functions for layers other than the output layer, which is always a SIGMOID.
     */
    public MultiLayerPerceptron(int[] networkShape, ActivationFunction[] activationFunctions) {
        layers = new Perceptron[networkShape.length - 1];
        for(int i=0; i<layers.length; ++i){
            layers[i] = new Perceptron("L"+i, networkShape[i], networkShape[i+1], activationFunctions[i%activationFunctions.length]);
        }
        layers[layers.length-1].setActivationFunction(ActivationFunction.SIGMOID);
    }


    /**
     * @See {link {{@link NeuralNetwork#compute(double[])}}}
     * <p>
     *     Compute and propagate outputs through the network's layers and calculate the error in the output layer.
     * </p>
     * @param inputs the normalized instance
     * @return
     */
    @Override
    public double[] compute(double[] inputs) {
        double[] result = inputs;
        for(int i=0; i<layers.length; ++i){
            result = layers[i].compute(result);
        }
        return result;
    }

    /**
     * Propagate the current weight deltas back to previous layers.
     *
     * @param rate the current learning rate.
     */
    protected void backpropagate(double rate) {
        for(int layerIdx=layers.length-2; layerIdx >= 0; --layerIdx){
            for(int i=0; i<layers[layerIdx].numOutputNodes(); ++i){
                double error = 0;
                for(int j=0; j<layers[layerIdx+1].numOutputNodes(); ++j){
                    error += layers[layerIdx+1].delta(j) *  layers[layerIdx+1].getWeightFromTo(i, j);
                }
                layers[layerIdx].setError(i, error);
            }
            layers[layerIdx].updateWeights(rate);
        }
    }


    /**
     * @see {link {{@link NeuralNetwork#train(double[], double[], double)}}}
     *<p>
     *     Iterate the current training episode through all layers and backpropagate the weight deltas.
     *</p>
     * @param inputs an input feature vector
     * @param targets target value corresponding to the input
     * @param rate the current learning rate
     */
    @Override
    public void train(double[] inputs, double[] targets, double rate) {
        double[] curLayerInput = inputs;
        for(int i=0; i<layers.length - 1; ++i){
            curLayerInput = layers[i].compute(curLayerInput);
        }
        layers[layers.length - 1].train(curLayerInput, targets, rate);
        backpropagate(rate);
    }

    public void printErrors(){
        layers[layers.length-1].printErrors();
    }
}
