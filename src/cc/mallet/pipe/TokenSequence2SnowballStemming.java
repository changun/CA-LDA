package cc.mallet.pipe;

import cc.mallet.types.Instance;
import cc.mallet.types.Token;
import cc.mallet.types.TokenSequence;
import org.tartarus.snowball.ext.EnglishStemmer;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

/**
 * A token sequence pipe that stems the english tokens
 * Created by changun on 2016/8/16.
 */
public class TokenSequence2SnowballStemming extends Pipe{

    EnglishStemmer stemmer = new EnglishStemmer();
    public Instance pipe (Instance carrier)
    {
        TokenSequence ts = (TokenSequence) carrier.getData();
        for (int i = 0; i < ts.size(); i++) {
            Token t = ts.get(i);
            stemmer.setCurrent(t.getText());
            stemmer.stem();
            t.setText(stemmer.getCurrent());
        }
        return carrier;
    }

    // Serialization 

    private static final long serialVersionUID = 1;
    private static final int CURRENT_SERIAL_VERSION = 0;

    private void writeObject (ObjectOutputStream out) throws IOException {
        out.writeInt (CURRENT_SERIAL_VERSION);
    }

    private void readObject (ObjectInputStream in) throws IOException, ClassNotFoundException {
        int version = in.readInt ();
    }

}
