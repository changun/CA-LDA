package cc.mallet.topics;

import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.Instance;

/**
 *
 * Created by Cheng-Kang Hsieh on 9/7/15.
 */
public class BackgroundTopicInferencer extends TopicInferencer {
    protected int[] typeBackgroundCounts; // indexed by <feature index>
    // number of background/topical tokens
    protected int[] backgroundAndTopicalCounts;
    // the index of topical/background tokens counts
    public static int TOPICAL_WORD_INDEX = 0, BACKGROUND_WORD_INDEX =1;
    // the pseudo topic for background words
    public int backgroundTopic = ParallelTopicModel.UNASSIGNED_TOPIC;

    double lambda;
    double alphaSum;
    // the constant part in the background work coefficient which we cache for efficiency
    double backgroundCoeffConst;

    public BackgroundTopicInferencer(int[][] typeTopicCounts, int[] tokensPerTopic,
                                     int[] typeBackgroundCounts, int[] backgroundAndTopicalCounts,
                                     Alphabet alphabet, double[] alpha, double beta, double betaBackground, double betaSum,     double lambda) {
        super(typeTopicCounts, tokensPerTopic, alphabet, alpha, beta, betaSum);
        this.typeBackgroundCounts = typeBackgroundCounts;
        this.backgroundAndTopicalCounts = backgroundAndTopicalCounts;
        this.lambda = lambda;
        this.alphaSum = 0.0;
        this.backgroundCoeffConst = 1.0 / ((betaBackground * numTypes) + backgroundAndTopicalCounts[BACKGROUND_WORD_INDEX] );
        for(double a: alpha){
            alphaSum += a;
        }
    }
    public BackgroundTopicInferencer(int[][] typeTopicCounts, int[] tokensPerTopic,
                                     int[] typeBackgroundCounts, int[] backgroundAndTopicalCounts,
                                     Alphabet alphabet, double[] alpha, double beta, double betaSum, double lambda) {
        this(typeTopicCounts, tokensPerTopic, typeBackgroundCounts, backgroundAndTopicalCounts, alphabet, alpha, beta, beta, betaSum, lambda);
    }
    protected double computeBackgroundCoeff(int token, int[] localBackgroundTopicCount){

        double coff = alphaSum + localBackgroundTopicCount[TOPICAL_WORD_INDEX];
        coff *=  (typeBackgroundCounts[token] + beta);
        coff *=  (localBackgroundTopicCount[BACKGROUND_WORD_INDEX] + lambda );
        coff /=  (localBackgroundTopicCount[TOPICAL_WORD_INDEX] + lambda );
        coff *=  backgroundCoeffConst;

        return coff;
    }
    /**
     *  Use Gibbs sampling to infer a topic distribution.
     *  Topics are initialized to the (or a) most probable topic
     *   for each token. Using zero iterations returns exactly this
     *   initial topic distribution.<p/>
     *  This code does not adjust type-topic counts: P(w|t) is clamped.
     */
    public double[] getSampledDistribution(Instance instance, int numIterations,
                                           int thinning, int burnIn) {

        FeatureSequence tokens = (FeatureSequence) instance.getData();
        int docLength = tokens.size();
        int[] topics = new int[docLength];

        int[] localTopicCounts = new int[numTopics];
        int[] localTopicIndex = new int[numTopics];

        int[] localBackgroundTopicCount = new int[2];

        int type;
        int[] currentTypeTopicCounts;

        // Initialize all positions to the most common topic
        //  for that type.

        for (int position = 0; position < docLength; position++) {
            type = tokens.getIndexAtPosition(position);
            // Ignore out of vocabulary terms
            if (type < numTypes && typeTopicCounts[type].length != 0) {

                currentTypeTopicCounts = typeTopicCounts[type];

                // This value should be a topic such that
                //  no other topic has more tokens of this type
                //  assigned to it. If for some reason there were
                //  no tokens of this type in the training data, it
                //  will default to topic 0, which is no worse than
                //  random initialization.
                topics[position] =
                        currentTypeTopicCounts[0] & topicMask;

                localTopicCounts[topics[position]]++;
                localBackgroundTopicCount[TOPICAL_WORD_INDEX] ++;
            }
        }

        // Build an array that densely lists the topics that
        //  have non-zero counts.
        int denseIndex = 0;
        for (int topic = 0; topic < numTopics; topic++) {
            if (localTopicCounts[topic] != 0) {
                localTopicIndex[denseIndex] = topic;
                denseIndex++;
            }
        }

        // Record the total number of non-zero topics
        int nonZeroTopics = denseIndex;

        //      Initialize the topic count/beta sampling bucket
        double topicBetaMass = 0.0;

        // Initialize cached coefficients and the topic/beta
        //  normalizing constant.

        for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
            int topic = localTopicIndex[denseIndex];
            int n = localTopicCounts[topic];

            //  initialize the normalization constant for the (B * n_{t|d}) term
            topicBetaMass += beta * n / (tokensPerTopic[topic] + betaSum);

            //  update the coefficients for the non-zero topics
            cachedCoefficients[topic] = (alpha[topic] + n) / (tokensPerTopic[topic] + betaSum);
        }

        double topicTermMass = 0.0;
        double[] topicTermScores = new double[numTopics];
        int[] topicTermIndices;
        int[] topicTermValues;
        int i;
        double score;

        int oldTopic, newTopic;

        double[] result = new double[numTopics+1];
        double sum = 0.0;

        for (int iteration = 1; iteration <= numIterations; iteration++) {

            //  Iterate over the positions (words) in the document
            for (int position = 0; position < docLength; position++) {
                type = tokens.getIndexAtPosition(position);

                // ignore out-of-vocabulary terms
                if (type >= numTypes || typeTopicCounts[type].length == 0) { continue; }

                oldTopic = topics[position];
                currentTypeTopicCounts = typeTopicCounts[type];
                if(oldTopic != backgroundTopic) {
                    // remove its contribution to localB/TCount
                    localBackgroundTopicCount[TOPICAL_WORD_INDEX]--;

                    // Prepare to sample by adjusting existing counts.
                    // Note that we do not need to change the smoothing-only
                    //  mass since the denominator is clamped.

                    topicBetaMass -= beta * localTopicCounts[oldTopic] /
                            (tokensPerTopic[oldTopic] + betaSum);

                    // Decrement the local doc/topic counts

                    localTopicCounts[oldTopic]--;
                    //assert(localTopicCounts[oldTopic] >= 0);

                    // Maintain the dense index, if we are deleting
                    //  the old topic
                    if (localTopicCounts[oldTopic] == 0) {

                        // First get to the dense location associated with
                        //  the old topic.

                        denseIndex = 0;

                        // We know it's in there somewhere, so we don't
                        //  need bounds checking.
                        while (localTopicIndex[denseIndex] != oldTopic) {
                            denseIndex++;
                        }

                        // shift all remaining dense indices to the left.
                        while (denseIndex < nonZeroTopics) {
                            if (denseIndex < localTopicIndex.length - 1) {
                                localTopicIndex[denseIndex] =
                                        localTopicIndex[denseIndex + 1];
                            }
                            denseIndex++;
                        }

                        nonZeroTopics--;
                    } // finished maintaining local topic index

                    topicBetaMass += beta * localTopicCounts[oldTopic] /
                            (tokensPerTopic[oldTopic] + betaSum);

                    // Reset the cached coefficient for this topic
                    cachedCoefficients[oldTopic] =
                            (alpha[oldTopic] + localTopicCounts[oldTopic]) /
                                    (tokensPerTopic[oldTopic] + betaSum);
                    if (cachedCoefficients[oldTopic] <= 0) {
                        System.out.println("zero or less coefficient: " + oldTopic + " = (" + alpha[oldTopic] + " + " + localTopicCounts[oldTopic] + ") / ( " + tokensPerTopic[oldTopic] + " + " + betaSum + " );");
                    }



                    boolean alreadyDecremented = false;


                }else{
                    localBackgroundTopicCount[BACKGROUND_WORD_INDEX]--;
                }
                int index = 0;
                int currentTopic, currentValue;

                topicTermMass = 0.0;

                while (index < currentTypeTopicCounts.length &&
                        currentTypeTopicCounts[index] > 0) {
                    currentTopic = currentTypeTopicCounts[index] & topicMask;
                    currentValue = currentTypeTopicCounts[index] >> topicBits;

                    score =
                            cachedCoefficients[currentTopic] * currentValue;
                    topicTermMass += score;
                    topicTermScores[index] = score;

                    index++;
                }
                // compute the mass for background topic
                double backgroundMass = computeBackgroundCoeff(type, localBackgroundTopicCount);
                double sample = random.nextUniform() * (smoothingOnlyMass + topicBetaMass + topicTermMass + backgroundMass);
                //  Make sure it actually gets set
                newTopic = -1;
                // assign it as a background term if sample < backgroundMass
                if(sample < backgroundMass){
                    newTopic = backgroundTopic;
                }else{
                    // otherwise, do what original LDA does.
                    sample -= backgroundMass;
                    if (sample < topicTermMass) {
                        //topicTermCount++;

                        i = -1;
                        while (sample > 0) {
                            i++;
                            sample -= topicTermScores[i];
                        }

                        newTopic = currentTypeTopicCounts[i] & topicMask;
                    }
                    else {
                        sample -= topicTermMass;

                        if (sample < topicBetaMass) {
                            //betaTopicCount++;

                            sample /= beta;

                            for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
                                int topic = localTopicIndex[denseIndex];

                                sample -= localTopicCounts[topic] /
                                        (tokensPerTopic[topic] + betaSum);

                                if (sample <= 0.0) {
                                    newTopic = topic;
                                    break;
                                }
                            }

                        }
                        else {
                            sample -= topicBetaMass;

                            sample /= beta;

                            newTopic = 0;
                            sample -= alpha[newTopic] /
                                    (tokensPerTopic[newTopic] + betaSum);

                            while (sample > 0.0) {
                                newTopic++;

                                if (newTopic >= numTopics) {
                                    index = 0;

                                    while (index < currentTypeTopicCounts.length &&
                                            currentTypeTopicCounts[index] > 0) {
                                        currentTopic = currentTypeTopicCounts[index] & topicMask;
                                        currentValue = currentTypeTopicCounts[index] >> topicBits;

                                        System.out.println(currentTopic + "\t" + currentValue + "\t" + topicTermScores[index] +
                                                "\t" + cachedCoefficients[currentTopic]);
                                        index++;
                                    }
                                }

                                sample -= alpha[newTopic] /
                                        (tokensPerTopic[newTopic] + betaSum);
                            }

                        }

                    }

                }

                // update all the parameters
                topics[position] = newTopic;

                if(newTopic == backgroundTopic){
                    localBackgroundTopicCount[BACKGROUND_WORD_INDEX] ++;
                }else{
                    localBackgroundTopicCount[TOPICAL_WORD_INDEX] ++;

                    topicBetaMass -= beta * localTopicCounts[newTopic] /
                            (tokensPerTopic[newTopic] + betaSum);

                    localTopicCounts[newTopic]++;

                    // If this is a new topic for this document,
                    //  add the topic to the dense index.
                    if (localTopicCounts[newTopic] == 1) {

                        // First find the point where we
                        //  should insert the new topic by going to
                        //  the end (which is the only reason we're keeping
                        //  track of the number of non-zero
                        //  topics) and working backwards

                        denseIndex = nonZeroTopics;

                        while (denseIndex > 0 &&
                                localTopicIndex[denseIndex - 1] > newTopic) {

                            localTopicIndex[denseIndex] =
                                    localTopicIndex[denseIndex - 1];
                            denseIndex--;
                        }

                        localTopicIndex[denseIndex] = newTopic;
                        nonZeroTopics++;
                    }

                    //  update the coefficients for the non-zero topics
                    cachedCoefficients[newTopic] =
                            (alpha[newTopic] + localTopicCounts[newTopic]) /
                                    (tokensPerTopic[newTopic] + betaSum);

                    topicBetaMass += beta * localTopicCounts[newTopic] /
                            (tokensPerTopic[newTopic] + betaSum);
                }

            }

            if (iteration > burnIn &&
                    (iteration - burnIn) % thinning == 0) {

                // Save a sample
                for (int topic=0; topic < numTopics; topic++) {
                    result[topic] += alpha[topic] + localTopicCounts[topic];
                    sum += alpha[topic] + localTopicCounts[topic];
                }
                result[numTopics] += localBackgroundTopicCount[BACKGROUND_WORD_INDEX];
                sum += result[numTopics];

            }
        }

        //  Clean up our mess: reset the coefficients to values with only
        //  smoothing. The next doc will update its own non-zero topics...
        for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
            int topic = localTopicIndex[denseIndex];

            cachedCoefficients[topic] =
                    alpha[topic] / (tokensPerTopic[topic] + betaSum);
        }

        if (sum == 0.0) {
            // Save at least one sample
            for (int topic=0; topic < numTopics; topic++) {
                result[topic] = alpha[topic] + localTopicCounts[topic];
                sum += result[topic];
            }
            result[numTopics] += localBackgroundTopicCount[BACKGROUND_WORD_INDEX];
            sum += result[numTopics];
        }

        // Normalize
        for (int topic=0; topic < result.length; topic++) {
            result[topic] /= sum;
        }

        return result;
    }

}
