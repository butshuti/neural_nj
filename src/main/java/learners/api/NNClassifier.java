package learners.api;

import learners.core.eval.LabelDist;
import learners.perceptrons.NeuralNetwork;

import java.io.Serializable;

public class NNClassifier implements Serializable{
    private NeuralNetwork neuralNetwork;
    private LabelDist labels;
    private int selectedAttributes[];
    int featureLen;

    public NNClassifier(NeuralNetwork neuralNetwork, LabelDist labels, int selectedAttributes[], int featureLen){
        this.neuralNetwork = neuralNetwork;
        this.labels = labels;
        this.selectedAttributes = selectedAttributes;
        this.featureLen = featureLen;
    }

    public NeuralNetwork getNN() {
        return neuralNetwork;
    }

    public LabelDist getLabels() {
        return labels;
    }

    public double[] selectFeatures(double input[]){
        if(selectedAttributes != null && input.length == featureLen){
            double filteredInput[] = new double[selectedAttributes.length];
            for(int i=0; i<selectedAttributes.length && selectedAttributes[i] < input.length; i++){
                filteredInput[i] = input[selectedAttributes[i]];
            }
            return filteredInput;
        }else if(input.length == selectedAttributes.length){
            return input;
        }
        return new double[selectedAttributes.length];
    }

    public String classify(double input[]){
        double output[] = neuralNetwork.process(selectFeatures(input));
        return labels.getBestMatch(output);
    }
}