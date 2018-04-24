package learners.utils;

import java.io.*;

public class Serialization {
    private static final String SUFFIX = ".model";

    public static void saveModel(Object obj, String path) throws IOException {
        FileOutputStream fos = new FileOutputStream(path + SUFFIX);
        ObjectOutputStream oos = new ObjectOutputStream(fos);
        oos.writeObject(obj);
        oos.close();
        fos.close();
    }

    public static Object loadModel(String path) throws IOException, ClassNotFoundException {
        FileInputStream fis = new FileInputStream(path + SUFFIX);
        ObjectInputStream ois = new ObjectInputStream(fis);
        Object obj = ois.readObject();
        ois.close();
        fis.close();
        return obj;
    }
}
