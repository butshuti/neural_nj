package main;

import learners.core.eval.Dataset;
import learners.core.eval.CSVDataset;
import learners.core.eval.Score;
import learners.perceptrons.ActivationFunction;
import learners.perceptrons.MultiLayerPerceptron;
import learners.perceptrons.NeuralNetwork;

import java.io.FileNotFoundException;
import java.io.IOException;

import static learners.utils.Serialization.loadModel;
import static learners.utils.Serialization.saveModel;

public class Main {

    private static void testPerceptron() throws IOException {
        boolean nominal = true;
        Dataset training_dataset = new CSVDataset(nominal);
        Dataset testing_dataset = new CSVDataset(nominal);
        Score score = new Score(training_dataset, 0.2);
        Score score2 = new Score(training_dataset, 0.2);
        Score score3 = new Score(training_dataset, 0.2);
        //int[] selectedAttributes = new int[]{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
        //int[] selectedAttributes = new int[]{1,2,3,4,5,6,7,8,9};
        MultiLayerPerceptron mlp = null, mlp2 = null, savedMlp = null;
        training_dataset.fromInput("/home/butshuti/on_recoil/nn/data/recoil/manual_training.csv", null);
        testing_dataset.fromInput("/home/butshuti/on_recoil/nn/data/recoil/manual_testing.csv", null);
        if (training_dataset.hasNext()){
            double[][] training_inputs = training_dataset.getInputs();
            double[][] training_targets = training_dataset.getTargets();
            int[] shape1 = new int[]{training_inputs[0].length, 4, training_targets[0].length};
            int[] shape2 = new int[]{training_inputs[0].length, 4, 4, 4, training_targets[0].length};
            ActivationFunction[] activations = new ActivationFunction[]{ActivationFunction.RELU};
            ActivationFunction[] activations2 = new ActivationFunction[]{ActivationFunction.TANH};
            mlp = new MultiLayerPerceptron(shape1, activations);
            mlp2 = new MultiLayerPerceptron(shape2, activations2);
            try {
                savedMlp = (MultiLayerPerceptron) loadModel(training_dataset.getDataPath());
            } catch (ClassNotFoundException e) {
                e.printStackTrace();
            }catch (FileNotFoundException e2){
                e2.printStackTrace();
            }
            mlp.trainNetwork(training_inputs, training_targets, 10000, 0.3, NeuralNetwork.FeatureDescriptionOptions.STANDARDIZED, NeuralNetwork.OutputMap.SCALED);
            mlp2.trainNetwork(training_inputs, training_targets, 10000, 0.3, NeuralNetwork.FeatureDescriptionOptions.STANDARDIZED, NeuralNetwork.OutputMap.SCALED);
            mlp.printErrors();
            mlp2.printErrors();
            if(savedMlp != null){
                savedMlp.printErrors();
            }
        }
        if (testing_dataset.hasNext() && mlp != null){
            double[][] testing_inputs = testing_dataset.getInputs();
            double[][] testing_targets = testing_dataset.getTargets();
            testResult(score, mlp, testing_inputs, testing_targets, 0.2, false);
            testResult(score2, mlp2, testing_inputs, testing_targets, 0.2, false);
            if(savedMlp != null){
                testResult(score3, savedMlp, testing_inputs, testing_targets, 0.2, false);
            }
        }
        saveModel(mlp, training_dataset.getDataPath());
    }

    public static void testResult(Score score, NeuralNetwork p, double[][] inputs, double[][] targets, double margin, boolean verbose) {
        if(score == null){
            return;
        }
        System.out.println("=============");
        for (int i = 0; i < inputs.length; ++i) {
            double[] result = p.process(inputs[i]);
            score.recordResult(targets[i], result, verbose);
        }
        System.out.println(score.printSummary());
        System.out.println("=============");
    }

    public static void main(String[] args){
        try {
            testPerceptron();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
