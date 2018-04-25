package learners.core.eval;

import java.util.*;

public class Score {
    private int pos, neg;
    private LabelDist labelDist;
    private Map<String, List<String>> predictionMatrix;
    private Map<String, Integer> predictions;
    private double margin;

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

    double getAccuracy(){
        return ((double)pos) / (pos + neg);
    }

    public void recordResult(double[] expected, double[] prediction, boolean verbose){
        double dist = labelDist.dist(expected, prediction);
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
        double score = 0;
        StringBuilder sb = new StringBuilder("\n====Contingency Matrix : ====\n");
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
            double precision = 1.0*tp / (tp + fp);
            double recall = 1.0*tp / (tp + fn);
            score += 2 * ((precision * recall)/(precision + recall));
            sb.append(actual);
            sb.append(" => ");
            sb.append(expected.toString());
            sb.append("\n");
            sb.append("---------------------------------\n");
        }
        score /= predictionMatrix.keySet().size();
        score = Math.round(100 * score)/100.0;
        double accuracy = Math.round(100 * getAccuracy())/100.0;
        sb.append("Accuracy: ");
        sb.append(String.valueOf(accuracy));
        sb.append("\t; ");
        sb.append("F-Score: ");
        sb.append(String.valueOf(score));
        sb.append("\n---------------------------------\n");
        return sb.toString();
    }
}
