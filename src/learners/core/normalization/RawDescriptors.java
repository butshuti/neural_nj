package learners.core.normalization;

/**
 *
 */
public class RawDescriptors extends FeatureDescriptors {
    private double[] centers, scales;

    @Override
    public double[] getAttributesCenters() {
        if(centers == null){
            centers = new double[getAttributesRange().length];
            for(int i=0; i<centers.length; i++){
                centers[i] = 0;
            }
        }
        return centers;
    }

    @Override
    public double[] getAttributesScales() {
        if(scales == null){
            scales = new double[getAttributesRange().length];
            for(int i=0; i<centers.length; i++){
                scales[i] = 1;
            }
        }
        return scales;
    }

    @Override
    public void invalidate() {
        scales = null;
        centers = null;
    }
}
