package learners.core.eval;

import java.io.IOException;
import java.io.InputStream;
import java.io.Serializable;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import static learners.core.eval.Dataset.NominalCategory.UNKNOWN_CATEGORY;

/**
 * Interface for loading and operating on datasets.
 */
public abstract class Dataset{

    private double[][] inputs; /*Set of input vectors in the dataset*/
    private double[][] targets; /*Set of target values (vectors) corresponding to the input vectors*/
    private Set<NominalCategory> labelSet; /*Set of pivot target values used in projecting predictions on the training set.*/
    private LabelSet labelMatcher;
    private int index;
    private boolean isNominal; /*Whether categories are nominal*/


    /**
     * Construct a new dataset
     * @param nominal whether target values are nominal.
     */
    public Dataset(boolean nominal){
        inputs = null;
        targets = null;
        index = 0;
        isNominal = nominal;
        labelMatcher = new LabelSet();
    }

    /**
     * Check whether target values are nominal.
     * @return true if nominal, and false if numeric.
     */
    public final boolean isNominal() {
        return isNominal;
    }

    /**
     * Check if the dataset's cursor can be advanced.
     * @return false if cursor at the end.
     */
    public final boolean hasNext(){
        return inputs != null && index < inputs.length;
    }

    /**
     * Get all input vectors.
     * @return
     */
    public final double[][] getInputs() {
        return inputs;
    }

    /**
     * Get all target vectors.
     * @return
     */
    public final double[][] getTargets() {
        return targets;
    }

    /**
     * Get label mappings
     * @return
     */
    public LabelDist getLabels(){
        return labelMatcher;
    }

    /**
     * Initialize the dataset.
     * @param inputs a set of input vectors.
     * @param targets a set of target values corresponding to the input vectors.
     */
    protected final void reset(double[][] inputs, String[] targets){
        this.inputs = inputs;
        this.targets = new double[targets.length][];
        labelSet = new HashSet<>();
        Map<String, Integer> targetsMap = new HashMap<>();
        for(String s : targets){
            if(!targetsMap.containsKey(s)){
                targetsMap.put(s, targetsMap.keySet().size());
            }
        }
        for(int i=0; i<targets.length; i++){
            if(isNominal()) {
                this.targets[i] = new double[targetsMap.keySet().size()];
                this.targets[i][targetsMap.get(targets[i])] = 1;
            }else{
                this.targets[i] = new double[]{targetsMap.get(targets[i])};
            }
            labelSet.add(new NominalCategory(targets[i], this.targets[i]));
        }
        labelMatcher = labelMatcher.setLabelSet(labelSet);
        index = 0;
    }

    protected final void reset(double[][] inputs, double[][] targets, Dataset labelDist){
        this.labelSet = new HashSet(labelDist.labelSet);
        this.inputs = inputs;
        this.targets = targets;
        labelMatcher = labelMatcher.setLabelSet(labelSet);
        index = 0;
    }

    /**
     * Initialize the dataset from a data file.
     *
     * @param path a path to a data file/directory
     * @param selectedAttributes a list of attribute indices to include
     * @throws IOException
     */
    public abstract void fromFile(String path, int[] selectedAttributes) throws IOException;

    /**
     * Initialize the dataset from a data file.
     *
     * @param inputStream an input stream
     * @param selectedAttributes a list of attribute indices to include
     * @throws IOException
     */
    public abstract void fromInputStream(InputStream inputStream, int[] selectedAttributes) throws IOException;

    public abstract String getDataPath();

    public static final class InheritedDataset extends Dataset{
        String path;
        public InheritedDataset(double[][] inputs, double[][] targets, Dataset labelDist){
            super(labelDist.isNominal());
            reset(inputs, targets, labelDist);
            path = (labelDist.getDataPath() != null ? labelDist.getDataPath() : "") + "_inherited";

        }
        @Override
        public void fromFile(String path, int[] selectedAttributes) throws IOException {
            return;
        }

        @Override
        public void fromInputStream(InputStream inputStream, int[] selectedAttributes) throws IOException {
            return;
        }

        @Override
        public String getDataPath() {
            return path;
        }
    }

    /**
     * Wrapper for named/labelled feature points.
     */
    protected static final class NominalCategory implements Serializable{
        public static final String UNKNOWN_CATEGORY = "UNKNOWN";
        private String label;
        private double[] numerciValue;
        NominalCategory(String l, double[] val){
            label = l;
            numerciValue = val;
        }
    }

    private static final class LabelSet implements LabelDist, Serializable{
        private Set<NominalCategory> labelSet;

        private LabelSet(){}

        private LabelSet setLabelSet(Set<NominalCategory> labelSet){
            this.labelSet = labelSet;
            return this;
        }

        @Override
        public double dist(double[] p1, double[] p2){
            if(p1 != null && p2 != null && p1.length == p2.length && p1.length > 0){
                double diff = 0;
                for(int i=0; i<p1.length; i++){
                    diff += (p1[i] - p2[i]) * (p1[i] - p2[i]);
                }
                return Math.sqrt(diff/p1.length);
            }
            return Double.MAX_VALUE;
        }

        @Override
        public String getBestMatch(double[] target){
            NominalCategory bestMatch = null;
            double diff = 0;
            for(NominalCategory categ : labelSet){
                double curDiff = dist(target, categ.numerciValue);
                if(bestMatch == null || curDiff < diff){
                    diff = curDiff;
                    bestMatch = categ;
                }
            }
            if(bestMatch != null){
                return String.valueOf(bestMatch.label);
            }
            return String.valueOf(UNKNOWN_CATEGORY);
        }
    }
}
