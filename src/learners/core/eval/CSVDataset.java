package learners.core.eval;

import learners.utils.Arithmetic;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/**
 * A dataset constructed from csv files.
 */
public class CSVDataset extends Dataset {
    private String path;

    public CSVDataset(boolean nominal){
        super(nominal);
    }

    /**
     * Filter an attribute index.
     * @param i the candidate attribute index
     * @param selectedAttributes the list of selected attributes.
     * @return true if the index is selected.
     */
    private boolean isSelected(int i, int[] selectedAttributes){
        for(int j=0; j<selectedAttributes.length; j++){
            if(i == selectedAttributes[j]) {
                return true;
            }
        }
        return false;
    }

    /**
     * Parse an input csv file and initialize this dataset from the file's contents.
     * @param path a path to a data file/directory
     * @param selectedAttributes a list of attribute indices to include
     * @throws IOException
     */
    @Override
    public void fromInput(String path, int[] selectedAttributes) throws IOException {
        File file = new File(path);
        if(file.exists() && file.canRead()){
            Scanner scanner = new Scanner(file);
            List<double[]> inputVals = new ArrayList<>();
            List<String> targetVals = new ArrayList<>();
            String header = scanner.nextLine();
            int numCols = header.split(",").length;
            while (scanner.hasNext()){
                String toks[] = scanner.nextLine().split(",");
                if(toks.length != numCols){
                    throw new IOException("Malformatted CSV file: unequal number of columns.");
                }
                double cur[] = new double[toks.length - 1];
                for(int i=0; i<cur.length; i++){
                    if(selectedAttributes == null || isSelected(i, selectedAttributes)){
                        cur[i] = Double.parseDouble(toks[i]);
                    }
                }
                if(!Arithmetic.isNan(cur)){
                    inputVals.add(cur);
                    targetVals.add(toks[toks.length-1]);
                }
            }
            double[][] inputs = new double[inputVals.size()][];
            String[] targets = new String[inputVals.size()];
            for(int i=0; i<inputs.length; i++){
                inputs[i] = inputVals.get(i);
                targets[i] = targetVals.get(i);
            }
            this.reset(inputs, targets);
            this.path = path;
            return;
        }
        throw new IOException("Cannot read input file.");
    }

    @Override
    public String getDataPath(){
        return path;
    }
}
