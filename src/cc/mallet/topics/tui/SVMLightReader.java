package cc.mallet.topics.tui;

import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;

/**
 * Created by changun on 8/15/17.
 */
public class SVMLightReader {
    public static Instance parseLine(String line, Alphabet alphabet){
        String[] tokenStrings = line.split(" ");
        // the first element is the name of the input
        String name = tokenStrings[0];
        // the id of each feature
        int tokenIds[] = new int[tokenStrings.length-1];
        // the occurrence count of each feature
        int tokenCount[] = new int[tokenStrings.length-1];
        int total = 0;
        for (int i=1; i<tokenStrings.length; i++) {
            String[] f_c = tokenStrings[i].split(":", 2);
            int feature = Integer.valueOf(f_c[0]);
            int count = Integer.valueOf(f_c[1]);
            tokenIds[i - 1] = feature;
            tokenCount[i - 1] = count;
            total += count;
        }
        int tokens[] = new int[total];
        for (int i=0, index=0; i<tokenCount.length; i++) {
            for(int j=0; j<tokenCount[i]; j++, index++){
                tokens[index]= tokenIds[i];
            }
        }
        return new Instance(new FeatureSequence(alphabet, tokens), null, name, null);
    }

}
