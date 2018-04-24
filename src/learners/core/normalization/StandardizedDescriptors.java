package learners.core.normalization;

public class StandardizedDescriptors extends FeatureDescriptors {
    @Override
    public double[] getAttributesCenters() {
        return getAttributesMean();
    }

    @Override
    public double[] getAttributesScales() {
        return getAttributesSTD();
    }

    @Override
    public void invalidate() {

    }
}
