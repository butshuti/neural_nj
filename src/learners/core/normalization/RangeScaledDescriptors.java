package learners.core.normalization;

/**
 *
 */
public class RangeScaledDescriptors extends FeatureDescriptors {

    @Override
    public double[] getAttributesCenters() {
        return getAttributesMin();
    }

    @Override
    public double[] getAttributesScales() {
        return getAttributesRange();
    }

    @Override
    public void invalidate() {

    }
}
