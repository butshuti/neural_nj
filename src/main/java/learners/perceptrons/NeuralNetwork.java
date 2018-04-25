package learners.perceptrons;

import learners.core.normalization.*;

import java.io.Serializable;

import static learners.perceptrons.NeuralNetwork.OutputMap.RAW;
import static learners.perceptrons.NeuralNetwork.OutputMap.SCALED;

/**
 * An interface for a simple neural network.
 */
public abstract class NeuralNetwork implements Serializable {

    public enum FeatureDescriptionOptions {
        RESCALED /*scale attributes based on ranges*/,
        NORMALIZED /*normalize attributes around the mean*/,
        STANDARDIZED /*standardize attributes*/,
        RAW /*use the raw attributes with no preprocessing*/
    }
    public enum OutputMap {
        SCALED /*output class probabilities/likelihoods for the result*/,
        RAW /*output the unchanged numerical value of the result*/
    }

    private static final long serialVersionUID = -4382934L;
    private FeatureDescriptors featureDescriptors = new RawDescriptors();
    private OutputMap outputMap = RAW;

    /**
     * Compute the result given a normalized input vector
     * @param inputs the normalized instance
     * @return the result
     */
    abstract protected double[] compute(double[] inputs);

    /**
     * Iterate with one training episode.
     *
     * @param inputs an input feature vector
     * @param targets target value corresponding to the input
     * @param rate the current learning rate
     */
    abstract public void train(double[] inputs, double[] targets, double rate);
    abstract public void printErrors();

    /**
     * Normalize a raw feature instance.
     * @param instance the instance to preprocess
     * @return a normalized version of the instance.
     */
    protected double[] regularizeInstance(double[] instance){
        return featureDescriptors.regulariseInstance(instance);
    }

    /**
     * Train the network
     *
     * @param inputs the training dataset
     * @param targets target values corresponding to input vectors in the dataset
     * @param numEpochs number of training epochs
     * @param learningRate the network's learning rate
     * @param attributesOptions an optional attribute selection
     * @param outputMap the shape of the output result
     */
    public void trainNetwork(double[][] inputs, double[][] targets, int numEpochs, double learningRate, FeatureDescriptionOptions attributesOptions, OutputMap outputMap) {
        this.outputMap = outputMap;
        if(attributesOptions.equals(FeatureDescriptionOptions.RESCALED)){
            featureDescriptors = new RangeScaledDescriptors();
        }else if(attributesOptions.equals(FeatureDescriptionOptions.NORMALIZED)){
            featureDescriptors = new MeanNormalizedDescriptors();
        }else if(attributesOptions.equals(FeatureDescriptionOptions.STANDARDIZED)){
            featureDescriptors = new StandardizedDescriptors();
        }else{
            featureDescriptors = new RawDescriptors();
        }
        double[][] attributes = featureDescriptors.normalize(inputs);
        for (int i = 0; i < numEpochs; ++i) {
            for (int j = 0; j < attributes.length; ++j) {
                train(attributes[j], targets[j], learningRate);
            }
        }
    }

    /**
     * Compute the result given an input vector.
     *
     * @param inputs the input vector
     * @return the result
     */
    public final double[] process(double[] inputs){
        double[] ret = compute(regularizeInstance(inputs));
        if(outputMap.equals(SCALED) && ret.length > 1) {
            double acc = 0;
            for (double val : ret) {
                acc += val;
            }
            if (acc > 0) {
                for (int i = 0; i < ret.length; i++) {
                    ret[i] /= acc;
                }
            }
        }
        return ret;
    }
}
