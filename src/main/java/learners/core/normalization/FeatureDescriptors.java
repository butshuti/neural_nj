package learners.core.normalization;

import java.io.Serializable;

/**
 * An interface to normalize input vectors.
 */
public abstract class FeatureDescriptors implements Serializable {

    private static final long serialVersionUID = -540358938L;

    private double[] attributesMean, attributesSTD, attributesMin, attributesMax, attributesRange;

    public abstract double[] getAttributesCenters();

    public abstract double[] getAttributesScales();

    /**
     * Invalidate previous state.
     * <p>This method is called when loading a new dataset, thus invalidating previous states</p>
     * <p>Implementations that depend on history such as mean, max,... must reset and recompute their attributes.</p>
     */
    public abstract void invalidate();

    /**
     * Normalize an entire dataset.
     *
     * @param attributes the set of features to normalize.
     * @return a normalized dataset.
     */
    public double[][] normalize(double[][] attributes){
        double ret[][] = new double[attributes.length][];
        invalidate();
        preprocess(attributes);
        double[] centers = getAttributesCenters();
        double[] scales = getAttributesScales();
        for (int i=0; i<attributes.length; i++){
            ret[i] = new double[attributes[i].length];
            for(int j=0; j<attributes[i].length; j++){
                ret[i][j] = (attributes[i][j] - centers[j])/scales[j];
            }
        }
        return ret;
    }

    /**
     * Normalize a raw feature instance based on the current dataset.
     * @param instance the raw instance to normalize
     * @return a normalized feature vector.
     */
    public double[] regulariseInstance(double[] instance){
        double[] ret = new double[instance.length];
        double[] centers = getAttributesCenters();
        double[] scales = getAttributesScales();
        for (int i=0; i<instance.length; i++){
            ret[i] = (instance[i] - centers[i]) / scales[i];
        }
        return ret;
    }

    protected double[] getAttributesMean() {
        return attributesMean;
    }

    protected double[] getAttributesSTD() {
        return attributesSTD;
    }

    protected double[] getAttributesMin() {
        return attributesMin;
    }

    protected double[] getAttributesRange() {
        return attributesRange;
    }

    /**
     * Integrate the dataset.
     * <p>This is done when loading a dataset, so that future instance normalizations are done with regard to the dataset's attribues.</p>
     * @param attributes the dataset.
     */
    private void preprocess(double[][] attributes){
        int numAttributes = attributes[0].length;
        attributesMean = new double[numAttributes];
        attributesSTD = new double[numAttributes];
        attributesMax = new double[numAttributes];
        attributesMin = new double[numAttributes];
        attributesRange = new double[numAttributes];
        for (int attrIdx=0; attrIdx<numAttributes; attrIdx++){
            attributesMean[attrIdx] = 0;
            attributesMin[attrIdx] = Double.MAX_VALUE;
            attributesMax[attrIdx] = Double.MIN_VALUE;
            for(int i=0; i<attributes.length; i++){
                attributesMean[attrIdx] += attributes[i][attrIdx];
                attributesMin[attrIdx] = attributes[i][attrIdx] < attributesMin[attrIdx] ? attributes[i][attrIdx] : attributesMin[attrIdx];
                attributesMax[attrIdx] = attributes[i][attrIdx] > attributesMax[attrIdx] ? attributes[i][attrIdx] : attributesMax[attrIdx];
            }
            attributesMean[attrIdx] /= attributes.length;
        }
        for (int attrIdx=0; attrIdx<numAttributes; attrIdx++){
            attributesSTD[attrIdx] = 0;
            for(int i=0; i<attributes.length; i++){
                attributesSTD[attrIdx] += (attributes[i][attrIdx] - attributesMean[attrIdx]) * (attributes[i][attrIdx] - attributesMean[attrIdx]);
            }
            attributesSTD[attrIdx] = Math.sqrt(attributesSTD[attrIdx]/(attributes.length-1));
            if(attributesSTD[attrIdx] == 0){
                attributesSTD[attrIdx] = 1;
            }
            attributesRange[attrIdx] = Math.max(attributesMax[attrIdx] - attributesMin[attrIdx], 0.0001);
        }
    }
}
