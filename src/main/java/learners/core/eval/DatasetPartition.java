package learners.core.eval;

import java.util.Arrays;
import java.util.Random;

/**
 * This class implements a k-fold partitioning of datasets for cross-validation.
 */
public class DatasetPartition {

    private static final class Instance implements Comparable<Instance>{
        double[] inputs;
        double[] targets;
        int index;

        public int getIndex() {
            return index;
        }

        public void setIndex(int index) {
            this.index = index;
        }

        Instance(double[] inputs, double[] targets){
            this.inputs = inputs;
            this.targets = targets;
        }

        @Override
        public int compareTo(Instance instance) {
            return index - instance.index;
        }
    }

    private Dataset trainingSet, testingSet;

    private DatasetPartition(Dataset trainingSet, Dataset testingSet){
        this.trainingSet = trainingSet;
        this.testingSet = testingSet;
    }

    public Dataset getTrainingSet(){
        return trainingSet;
    }

    public Dataset getTestingSet(){
        return testingSet;
    }

    public static DatasetPartition[] stratifyDataset(Dataset dataset, int folds, long seed){
        Random random = new Random(seed);
        DatasetPartition[] ret = new DatasetPartition[Math.max(2, folds)];
        double[][] inputs = dataset.getInputs();
        double[][] targets = dataset.getTargets();
        LabelDist labels = dataset.getLabels();
        Instance[] instances = new Instance[inputs.length];
        for(int i=0; i<instances.length; ++i){
            instances[i] = new Instance(inputs[i], targets[i]);
            instances[i].setIndex((int)labels.dist(instances[0].targets, instances[i].targets) + (instances.length/ret.length));
            instances[i].setIndex(instances[i].getIndex() + random.nextInt(instances.length));
        }
        Arrays.sort(instances);
        int[][] partitions = new int[ret.length][];
        int partitionSize = instances.length/ret.length;
        int lastPartitionSize = instances.length - (partitionSize*(partitions.length-1));
        for(int i=0, count=0; i<partitions.length; ++i){
            int sz = i < partitions.length - 1 ? partitionSize : lastPartitionSize;
            partitions[i] = new int[]{count, count+sz};
            count += sz;
        }
        for(int i=0; i<partitions.length; ++i){
            ret[i] = buildPartition(instances, partitions[i], dataset);
        }
        return ret;
    }

    private static DatasetPartition buildPartition(Instance[] instances, int[] selectedTestRange, Dataset masterDataset){
        int testingSetSize = selectedTestRange[1] - selectedTestRange[0];
        int trainingSetSize = instances.length - testingSetSize;
        double[][] testingInputs = new double[testingSetSize][], testingTargets = new double[testingSetSize][];
        double[][] trainingInputs = new double[trainingSetSize][], trainingTargets = new double[trainingSetSize][];
        int testingSetIndex = 0, trainingSetIndex = 0;
        for(int i=0; i<instances.length; ++i){
            if(i>=selectedTestRange[0] && i<selectedTestRange[1]){
                testingInputs[testingSetIndex] = instances[i].inputs;
                testingTargets[testingSetIndex] = instances[i].targets;
                testingSetIndex++;
            }else{
                trainingInputs[trainingSetIndex] = instances[i].inputs;
                trainingTargets[trainingSetIndex] = instances[i].targets;
                trainingSetIndex++;
            }
        }
        Dataset trainingDataset = new Dataset.InheritedDataset(trainingInputs, trainingTargets, masterDataset);
        Dataset testingDataset = new Dataset.InheritedDataset(testingInputs, testingTargets, masterDataset);
        return new DatasetPartition(trainingDataset, testingDataset);
    }
}
