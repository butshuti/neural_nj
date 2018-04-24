package learners.utils;

public class Arithmetic {

    public static double filter(double x){
        if(Double.isNaN(x) || Double.isInfinite(x)){
            return 0;
        }
        return x;
    }

    public static void filter(double[] arr){
        for(int i=0; i<arr.length; i++){
            arr[i] = filter(arr[i]);
        }
    }

    public static void filter(double[][] arr){
        for(double[] r : arr){
            filter(r);
        }
    }

    public static boolean isNan(double[] vals){
        for(int i=0; i<vals.length; i++){
            if(Double.isNaN(vals[i])){
                return true;
            }
        }
        return false;
    }
}
