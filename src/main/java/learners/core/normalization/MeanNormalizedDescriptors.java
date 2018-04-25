package learners.core.normalization;

/**
 *
 */
public class MeanNormalizedDescriptors extends RangeScaledDescriptors{

    @Override
    public double[] getAttributesCenters(){
        return getAttributesMean();
    }
}
