package learners.core.eval;

import java.util.*;

public class Score {
    private int pos, neg;
    private LabelDist labelDist;
    private Map<String, List<String>> predictionMatrix;
    private Map<String, Integer> predictions;
    private double margin, mse = 0;

    public Score(LabelDist labelDist, double margin){
        pos = 0;
        neg = 0;
        this.labelDist = labelDist;
        this.margin = margin;
        predictionMatrix = new HashMap<>();
        predictions = new HashMap<>();
    }

    void incr(){
        pos++;
    }
    void decr(){
        neg++;
    }

    public double getAccuracy(){
        return ((double)pos) / (pos + neg);
    }

    public void recordResult(double[] expected, double[] prediction, boolean verbose){
        double dist = labelDist.dist(expected, prediction);
        mse += dist;
        if(dist <= margin){
            incr();
        }else{
            decr();
        }
        String expectedLabel = labelDist.getBestMatch(expected);
        String predictedLabel = labelDist.getBestMatch(prediction);
        if(!predictionMatrix.containsKey(expectedLabel)){
            predictionMatrix.put(expectedLabel, new ArrayList<String>());
        }
        if(!predictions.containsKey(predictedLabel)){
            predictions.put(predictedLabel, 0);
        }
        predictionMatrix.get(expectedLabel).add(predictedLabel);
        predictions.put(predictedLabel, predictions.get(predictedLabel) + 1);
        if(verbose){
            System.out.println("Expected: " + Arrays.toString(expected) + "; predicted: " + Arrays.toString(prediction) + "; error: " + dist);
        }
    }

    public String printSummary(){
        int total = pos + neg;
        double AvgPrecision = 0, AvgRecall = 0;
        int globalTP = 0, globalFP = 0, globalTN = 0, globalFN = 0;
        StringBuilder sb = new StringBuilder("\n");
        sb.append("=================================\n");
        sb.append("====Contingency Matrix : ========\n");
        sb.append("---<EXPECTED => {PREDICTED}>-----\n");
        sb.append("---------------------------------\n");
        for(String actual : predictionMatrix.keySet()){
            if(!predictions.containsKey(actual)){
                predictions.put(actual, 0);
            }
        }
        for(String actual : predictionMatrix.keySet()){
            Map<String, Integer> expected = new HashMap<>();
            int tp = 0, fn = 0;
            for(String val : predictionMatrix.keySet()){
                expected.put(val, 0);
            }
            for(String val : predictionMatrix.get(actual)){
                expected.put(val, expected.get(val) + 1);
                if(val.equals(actual)){
                    tp++;
                }else{
                    fn++;
                }
            }
            int fp = predictions.get(actual) - tp;
            int tn = total - predictions.get(actual) - fn;
            globalTP += tp;
            globalFP += fp;
            globalTN += tn;
            globalFN += fn;
            AvgPrecision += 1.0*tp / (tp + fp);
            AvgRecall += 1.0*tp / (tp + fn);
            sb.append(actual);
            sb.append(" => ");
            sb.append(expected.toString());
            sb.append("\n");
            sb.append("---------------------------------\n");
        }
        AvgPrecision /= predictionMatrix.keySet().size();
        AvgRecall /= predictionMatrix.keySet().size();
        double score = 2 * ((AvgPrecision * AvgRecall)/(AvgPrecision + AvgRecall));
        score = Math.round(100 * score)/100.0;
        double accuracy = 1.0 * (globalTP + globalTN) / (globalTN + globalFN + globalFP + globalTP);
        accuracy = Math.round(100 * accuracy)/100.0;
        mse = Math.sqrt(mse / total);
        double computational_accuracy = Math.round(100 * getAccuracy())/100.0;
        sb.append("---------------------------------\n");
        sb.append("Evaluation 1: regression");
        sb.append("\n.................................\n");
        sb.append(String.format("** RMSE: %.2f\n", mse));
        sb.append(String.format("** Accuracy within %.2f: \t %.2f\n", margin, computational_accuracy));
        sb.append("\n---------------------------------\n");
        sb.append("Evaluation 2: classification");
        sb.append("\n.................................\n");
        sb.append(String.format("Accuracy: %.2f", accuracy));
        sb.append("\t;  ");
        sb.append(" F-Score: ");
        sb.append(String.valueOf(score));
        sb.append("\n---------------------------------\n");
        return sb.toString();
    }
}
