package cc.mallet.examples;

import cc.mallet.pipe.*;
import cc.mallet.topics.BackgroundTopicInferencer;
import cc.mallet.topics.MultiBackgroundTopicModel;
import cc.mallet.topics.TopicInferencer;
import cc.mallet.types.*;
import cc.mallet.util.ArrayUtils;
import cc.mallet.util.IoUtils;
import java.io.File;
import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.lang.reflect.Array;
import java.util.*;
import java.util.zip.GZIPInputStream;

/**
 * A demo app for Context-Aware LDA that infers topic distribution and proportion of background words given a
 * trained model.
 * Created by changun on 2016/8/16.
 */
public class RunContextAwareLDA {

    private static Pipe createPipe(Alphabet alphabet){
        SerialPipes pipe = new SerialPipes();

        // the alphabet from a pre-trained model should not grow
        alphabet.stopGrowth();

        // use the same pipes we used to train the model

        pipe.pipes().add(new CharSequenceLowercase());
        pipe.pipes().add(new CharSequence2TokenSequence("\\p{L}\\p{L}\\p{L}+"));
        pipe.pipes().add(new TokenSequence2SnowballStemming());
        pipe.pipes().add(new TokenSequence2FeatureSequence(alphabet));
        return pipe;
    }
    private static double[] normalize(double[] dist, int start, int end){
        double[] ret = new double[end-start];
        double sum = 0.0;
        for(int i=start; i<end; i++){
            sum += dist[i];
        }
        for(int i=0; i<end-start; i++){
            ret[i] = dist[i+start]/sum;
        }
        return ret;

    }
    private static String iterator2String(Iterator<IDSorter> iterator, Alphabet alphabet){
        StringBuilder out = new StringBuilder();
        int word = 0;
        while (iterator.hasNext() && word < 10) {
            IDSorter info = iterator.next();
            out.append(alphabet.lookupObject(info.getID())).append(" ");
            word++;
        }
        return out.toString();
    }
    public static void main(String[] args) throws Exception {
        String modelPath = args[0];
        String documentPath = args[1];
        String context = args[2];

        FileInputStream fileInputStream = new FileInputStream(modelPath);

        System.out.println("Loading model ...");

        // check if model file is gzipped
        ObjectInputStream objectInputStream;
        if(modelPath.endsWith(".gz")){
            objectInputStream = new ObjectInputStream(new GZIPInputStream(fileInputStream));
        }else{
            objectInputStream = new ObjectInputStream(fileInputStream);
        }
        // deserialize the model
        MultiBackgroundTopicModel model = (MultiBackgroundTopicModel) objectInputStream.readObject();

        // get all sources (i.e. contexts) available in this model
        ArrayList<String> sources = new ArrayList<String>();
        for(Object source: model.getSources()){
            sources.add(source.toString());
        }
        // output source names
        System.out.println(String.format("Found %d contexts: %s", sources.size(), String.join(",", sources)));

        try {
            // get the inferencer for the given context
            TopicInferencer inf = model.getInferencer(context);


            // prepare the text pre-processing pipes
            Pipe pipe = createPipe(model.getAlphabet());
            InstanceList instances = new InstanceList(pipe);

            String documents = IoUtils.contentsAsString(new File(documentPath));
            for(String document: documents.split("\n")){
                instances.addThruPipe(new Instance(document, null, "demo", context));
            }
            double[] topicDist = new double[model.getNumTopics()];
            final HashMap<Object, Integer> backgroundCounts = new HashMap<Object, Integer>();
            Iterator<Instance> iter = instances.iterator();


            while(iter.hasNext()) {
                // perform inference
                Instance instance = iter.next();
                FeatureSequence tokens = (FeatureSequence) instance.getData();

                // add topic distribution
                final double[] dist = inf.getSampledDistribution(instance, 1000, 1, 5);
                final double[] normalizedDist = normalize(dist, 0, dist.length-1);
                assert normalizedDist.length == topicDist.length;
                for(int i=0; i< normalizedDist.length; i++){
                    topicDist[i] += normalizedDist[i];
                }


                final int[] assignments = ((BackgroundTopicInferencer) inf).getTopicAssignments(instance, 1000, 1, 5);


                for (int i = 0; i < tokens.size(); i++) {
                    Object token = instances.getAlphabet().lookupObject(tokens.getIndexAtPosition(i));
                    // update background counts
                    if (assignments[i] == -1) {
                        if(!backgroundCounts.containsKey(token)){
                            backgroundCounts.put(token, 1);
                        }else{
                            backgroundCounts.put(token, backgroundCounts.get(token)+1);
                        }
                    }
                }


            }
            final double[] finalTopicDist = normalize(topicDist, 0, topicDist.length);
            // create topic index array
            Integer[] topics = new Integer[finalTopicDist.length - 1];
            for (int i = 0; i < topics.length; i++) {
                topics[i] = i;
            }

            // sort topic index array by topic distribution array
            Arrays.sort(topics, new Comparator<Integer>() {
                @Override
                public int compare(Integer i1, Integer i2) {
                    return -Double.compare(finalTopicDist[i1], finalTopicDist[i2]);
                }
            });

            // output inference results
            System.out.println("Topic distribution:");

            // print all but the last element, which is the proportion of the background words
            ArrayList<TreeSet<IDSorter>> topicSortedWords = model.getSortedWords();
            for (Integer topic : topics) {
                System.out.print(String.format("%.3f of Topic %d: ", finalTopicDist[topic], topic));

                Iterator<IDSorter> iterator = topicSortedWords.get(topic).iterator();
                System.out.println(iterator2String(iterator, model.getAlphabet()));
            }
            Object[] backgroundWords = backgroundCounts.keySet().toArray();
            Arrays.sort(backgroundWords, new Comparator<Object>() {
                @Override
                public int compare(Object i1, Object i2) {
                    return -Double.compare(backgroundCounts.get(i1), backgroundCounts.get(i2));
                }
            });
            for(Object word: backgroundWords){
                System.out.println(word +":" + backgroundCounts.get(word));
            }






        }catch (IllegalArgumentException e){
            e.printStackTrace();
            System.exit(1);
        }


    }
}
