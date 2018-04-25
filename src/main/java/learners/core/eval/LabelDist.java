package learners.core.eval;

/**
 * Interface for calculating error/distance between vectors.
 */
public interface LabelDist {

    /**
     * Calculate distance between two vectors.
     * @param p1 first vector
     * @param p2 second vector
     * @return the calculated distance.
     */
    double dist(double[] p1, double[] p2);

    /**
     * Match a vector to the closest reference point.
     * <p>This is only useful when identifying best matches with regards to named references, such as labelled training data.</p>
     * @param target
     * @return
     */
    String getBestMatch(double[] target);
}
