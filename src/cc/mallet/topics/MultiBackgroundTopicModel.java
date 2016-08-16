package cc.mallet.topics;

import cc.mallet.types.*;
import cc.mallet.util.Randoms;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Created by Cheng-Kang Hsieh on 9/13/15.
 */
public class MultiBackgroundTopicModel extends BackgroundTopicModel {
    static final long serialVersionUID = -816987635771455958L;

    public Map<Object, Integer> getSourceToSourceId() {
        return sourceToSourceId;
    }

    public List<ArrayList<TopicAssignment>> getDataBySourceId() {
        return dataBySourceId;
    }

    Map<Object, Integer> sourceToSourceId = new HashMap<Object, Integer>();
    List<ArrayList<TopicAssignment>> dataBySourceId = new ArrayList<ArrayList<TopicAssignment>>();

    int[][] typeBackgroundCounts;
    int[][] backgroundAndTopicalCounts;
    int numSources;

    public MultiBackgroundTopicModel(LabelAlphabet topicAlphabet, double alphaSum, double beta, double betaBackground, double lambda) {
        super(topicAlphabet, alphaSum, beta, betaBackground, lambda);
    }
    public MultiBackgroundTopicModel (int numberOfTopics) {
        this (numberOfTopics, 1, DEFAULT_BETA, DEFAULT_BETA, DEFAULT_LAMBDA);
    }


    public MultiBackgroundTopicModel (int numberOfTopics, double alphaSum, double beta, double betaBackground, double lambda) {
        this (newLabelAlphabet (numberOfTopics), alphaSum, beta, betaBackground, lambda);
    }

    public void addInstances (InstanceList training) {
        int id = 0;
        for(Instance instance: training){
            Object source = instance.getSource();
            if(!sourceToSourceId.containsKey(source)){
                sourceToSourceId.put(source, id);
                dataBySourceId.add(new ArrayList<TopicAssignment>());
                id++;
            }
        }
        numSources = sourceToSourceId.size();
        alphabet = training.getDataAlphabet();
        numTypes = alphabet.size();

        betaSum = beta * numTypes;

        typeTopicCounts = new int[numTypes][];
        typeBackgroundCounts = new int[numSources][training.getAlphabet().size()];
        backgroundAndTopicalCounts = new int[numSources][2];

        // Get the total number of occurrences of each word type
        //int[] typeTotals = new int[numTypes];
        typeTotals = new int[numTypes];

        int doc = 0;
        for (Instance instance : training) {
            doc++;
            FeatureSequence tokens = (FeatureSequence) instance.getData();
            for (int position = 0; position < tokens.getLength(); position++) {
                int type = tokens.getIndexAtPosition(position);
                typeTotals[ type ]++;
            }
        }

        maxTypeCount = 0;

        // Allocate enough space so that we never have to worry about
        //  overflows: either the number of topics or the number of times
        //  the type occurs.
        for (int type = 0; type < numTypes; type++) {
            if (typeTotals[type] > maxTypeCount) { maxTypeCount = typeTotals[type]; }
            typeTopicCounts[type] = new int[ Math.min(numTopics, typeTotals[type]) ];
        }

        doc = 0;

        Randoms random = null;
        if (randomSeed == -1) {
            random = new Randoms();
        }
        else {
            random = new Randoms(randomSeed);
        }
        for (Instance instance : training) {
            doc++;
            int sourceId = sourceToSourceId.get(instance.getSource());
            FeatureSequence tokens = (FeatureSequence) instance.getData();
            LabelSequence topicSequence =
                    new LabelSequence(topicAlphabet, new int[tokens.size()]);

            int[] topics = topicSequence.getFeatures();
            for (int position = 0; position < topics.length; position++) {
                if (random.nextUniform() > lambda) {
                    int topic = random.nextInt(numTopics);
                    topics[position] = topic;
                    backgroundAndTopicalCounts[sourceId][TOPICAL_WORD_INDEX]++;

                } else {
                    topics[position] = backgroundTopic;
                    typeBackgroundCounts[sourceId][tokens.getIndexAtPosition(position)]++;
                    backgroundAndTopicalCounts[sourceId][BACKGROUND_WORD_INDEX]++;
                }
            }

            TopicAssignment t = new TopicAssignment(instance, topicSequence);
            dataBySourceId.get(sourceId).add(t);
            data.add(t);
        }

        buildInitialTypeTopicCounts();
        initializeHistograms();

    }
    int runnableToSourceId (BackgroundWorkerRunnable runnable){
        Object source = runnable.data.get(0).instance.getSource();
        return sourceToSourceId.get(source);
    }
    public void sumBackgroundTopicalCounts (BackgroundWorkerRunnable[] runnables) {

        // Clear the background counts
        for(int i=0; i<sourceToSourceId.size(); i++) {
            Arrays.fill(typeBackgroundCounts[i], 0);
            Arrays.fill(backgroundAndTopicalCounts[i], 0);
        }

        for (int thread = 0; thread < numThreads; thread++) {
            int sourceId = runnableToSourceId(runnables[thread]);
            // Handle the background topic distribution
            int[] sourceCounts = runnables[thread].getTypeBackgroundCounts();
            for (int type = 0; type < numTypes; type++) {
                typeBackgroundCounts[sourceId][type] += sourceCounts[type];
            }

            // Now handle the background/topical token counts
            int[] sourceTokenCounts = runnables[thread].getBackgroundAndTopicalCounts();
            for (int i = 0; i < sourceTokenCounts.length; i++){
                backgroundAndTopicalCounts[sourceId][i] += sourceTokenCounts[i];
            }

        }

    }
    public void estimate () throws IOException {

        long startTime = System.currentTimeMillis();
        assert (numThreads > 1);
        int numConcurrentThread = numThreads;

        numThreads = numThreads * sourceToSourceId.size();


        BackgroundWorkerRunnable[] runnables = new BackgroundWorkerRunnable[numThreads];
        for(int sourceId = 0 ; sourceId < sourceToSourceId.size(); sourceId ++){
            ArrayList<TopicAssignment> sourceData = dataBySourceId.get(sourceId);
            int docsPerThread = sourceData.size() / numConcurrentThread;
            int offset = 0;

            for (int sourceThread = 0; sourceThread < numConcurrentThread; sourceThread++) {
                int threadId = (sourceId * numConcurrentThread) + sourceThread;
                int[] runnableTotals = new int[numTopics];
                System.arraycopy(tokensPerTopic, 0, runnableTotals, 0, numTopics);

                int[][] runnableCounts = new int[numTypes][];
                for (int type = 0; type < numTypes; type++) {
                    int[] counts = new int[typeTopicCounts[type].length];
                    System.arraycopy(typeTopicCounts[type], 0, counts, 0, counts.length);
                    runnableCounts[type] = counts;
                }
                int[] runnableTypeBackgroundCounts = new int[numTypes];
                System.arraycopy(typeBackgroundCounts[sourceId], 0, runnableTypeBackgroundCounts, 0, runnableTypeBackgroundCounts.length);

                int[] runnableBackgroundAndTopicalCounts = new int[2];
                System.arraycopy(backgroundAndTopicalCounts[sourceId], 0, runnableBackgroundAndTopicalCounts, 0, runnableBackgroundAndTopicalCounts.length);


                // some docs may be missing at the end due to integer division
                if (sourceThread == numConcurrentThread - 1) {
                    docsPerThread = dataBySourceId.size() - offset;
                }

                Randoms random = null;
                if (randomSeed == -1) {
                    random = new Randoms();
                }
                else {
                    random = new Randoms(randomSeed);
                }

                runnables[threadId] = new BackgroundWorkerRunnable(numTopics,
                        alpha, alphaSum, beta, betaBackground, lambda,
                        random, sourceData,
                        runnableCounts, runnableTotals,
                        runnableTypeBackgroundCounts, runnableBackgroundAndTopicalCounts,
                        offset, docsPerThread);

                runnables[threadId].initializeAlphaStatistics(docLengthCounts.length);

                offset += docsPerThread;

            }

        }


        ExecutorService executor = Executors.newFixedThreadPool(numConcurrentThread);

        for (int iteration = 1; iteration <= numIterations; iteration++) {

            long iterationStart = System.currentTimeMillis();

            if (showTopicsInterval != 0 && iteration != 0 && iteration % showTopicsInterval == 0) {
                logger.info("\n" + displayTopWords (wordsPerTopic, false));
            }

            if (saveStateInterval != 0 && iteration % saveStateInterval == 0) {
                this.printState(new File(stateFilename + '.' + iteration));
            }

            if (saveModelInterval != 0 && iteration % saveModelInterval == 0) {
                this.write(new File(modelFilename + '.' + iteration));
            }


            // Submit runnables to thread pool

            for (BackgroundWorkerRunnable thread: runnables) {
                if (iteration > burninPeriod && optimizeInterval != 0 &&
                        iteration % saveSampleInterval == 0) {
                    thread.collectAlphaStatistics();
                }

                logger.fine("submitting thread " + thread);
                executor.submit(thread);
                //runnables[thread].run();
            }

            // I'm getting some problems that look like
            //  a thread hasn't started yet when it is first
            //  polled, so it appears to be finished.
            // This only occurs in very short corpora.
            try {
                Thread.sleep(20);
            } catch (InterruptedException e) {

            }

            boolean finished = false;
            while (! finished) {

                try {
                    Thread.sleep(10);
                } catch (InterruptedException e) {

                }

                finished = true;

                // Are all the threads done?
                for (int thread = 0; thread < numThreads; thread++) {
                    //logger.info("thread " + thread + " done? " + runnables[thread].isFinished);
                    finished = finished && runnables[thread].isFinished;
                }

            }

            //System.out.print("[" + (System.currentTimeMillis() - iterationStart) + "] ");

            sumTypeTopicCounts(runnables);
            sumBackgroundTopicalCounts(runnables);

            //System.out.print("[" + (System.currentTimeMillis() - iterationStart) + "] ");

            for (int thread = 0; thread < numThreads; thread++) {
                int sourceId = runnableToSourceId(runnables[thread]);
                int[] runnableTotals = runnables[thread].getTokensPerTopic();
                System.arraycopy(tokensPerTopic, 0, runnableTotals, 0, numTopics);

                int[][] runnableCounts = runnables[thread].getTypeTopicCounts();
                for (int type = 0; type < numTypes; type++) {
                    int[] targetCounts = runnableCounts[type];
                    int[] sourceCounts = typeTopicCounts[type];

                    int index = 0;
                    while (index < sourceCounts.length) {

                        if (sourceCounts[index] != 0) {
                            targetCounts[index] = sourceCounts[index];
                        }
                        else if (targetCounts[index] != 0) {
                            targetCounts[index] = 0;
                        }
                        else {
                            break;
                        }

                        index++;
                    }
                }

                System.arraycopy(typeBackgroundCounts[sourceId], 0, runnables[thread].getTypeBackgroundCounts(), 0, typeBackgroundCounts[sourceId].length);
                System.arraycopy(backgroundAndTopicalCounts[sourceId], 0, runnables[thread].getBackgroundAndTopicalCounts(), 0, backgroundAndTopicalCounts[sourceId].length);

            }
            long elapsedMillis = System.currentTimeMillis() - iterationStart;
            if (elapsedMillis < 1000) {
                logger.fine(elapsedMillis + "ms ");
            }
            else {
                logger.fine((elapsedMillis/1000) + "s ");
            }

            if (iteration > burninPeriod && optimizeInterval != 0 &&
                    iteration % optimizeInterval == 0) {

                optimizeAlpha(runnables);
                optimizeBeta(runnables);

                logger.fine("[O " + (System.currentTimeMillis() - iterationStart) + "] ");
            }

            if (iteration % 10 == 0) {
                if (printLogLikelihood) {
                    logger.info ("<" + iteration + "> LL/token: " + formatter.format(modelLogLikelihood() / totalTokens));
                }
                else {
                    logger.info ("<" + iteration + ">");
                }
            }
        }

        executor.shutdownNow();

        long seconds = Math.round((System.currentTimeMillis() - startTime)/1000.0);
        long minutes = seconds / 60;	seconds %= 60;
        long hours = minutes / 60;	minutes %= 60;
        long days = hours / 24;	hours %= 24;

        StringBuilder timeReport = new StringBuilder();
        timeReport.append("\nTotal time: ");
        if (days != 0) { timeReport.append(days); timeReport.append(" days "); }
        if (hours != 0) { timeReport.append(hours); timeReport.append(" hours "); }
        if (minutes != 0) { timeReport.append(minutes); timeReport.append(" minutes "); }
        timeReport.append(seconds); timeReport.append(" seconds");

        logger.info(timeReport.toString());

        // restore the numThreads
        numThreads = numConcurrentThread;
    }

    public String displayTopWords (int numWords, boolean usingNewLines) {

        StringBuilder out = new StringBuilder();

        ArrayList<TreeSet<IDSorter>> topicSortedWords = getSortedWords();
        // Print results for each topic
        for (int topic = 0; topic < topicSortedWords.size(); topic++) {
            TreeSet<IDSorter> sortedWords = topicSortedWords.get(topic);
            int word = 1;
            Iterator<IDSorter> iterator = sortedWords.iterator();
            String alphaStr = formatter.format(alpha[topic]);
            if (usingNewLines) {
                out.append (topic + "\t" + alphaStr + "\n");
                while (iterator.hasNext() && word < numWords) {
                    IDSorter info = iterator.next();
                    out.append(alphabet.lookupObject(info.getID()) + "\t" + formatter.format(info.getWeight()) + "\n");
                    word++;
                }
            }
            else {
                out.append (topic + "\t" + alphaStr + "\t");

                while (iterator.hasNext() && word < numWords) {
                    IDSorter info = iterator.next();
                    out.append(alphabet.lookupObject(info.getID()) + " ");
                    word++;
                }
                out.append ("\n");
            }
        }

        for (Object source : sourceToSourceId.keySet()) {
            int sourceId = sourceToSourceId.get(source);
            TreeSet<IDSorter> sortedWords = getSortedBackgroundWords(typeBackgroundCounts[sourceId]);
            int word = 1;
            Iterator<IDSorter> iterator = sortedWords.iterator();
            double ratio = ((double) backgroundAndTopicalCounts[sourceId][BACKGROUND_WORD_INDEX]) /
                    (backgroundAndTopicalCounts[sourceId][BACKGROUND_WORD_INDEX] + backgroundAndTopicalCounts[sourceId][TOPICAL_WORD_INDEX]);
            String alphaStr = formatter.format(ratio);
            if (usingNewLines) {
                out.append (source + "\t" + alphaStr + "\n");
                while (iterator.hasNext() && word < numWords) {
                    IDSorter info = iterator.next();
                    out.append(alphabet.lookupObject(info.getID()) + "\t" + formatter.format(info.getWeight()) + "\n");
                    word++;
                }
            }
            else {
                out.append (source + "\t" + alphaStr + "\t");

                while (iterator.hasNext() && word < numWords) {
                    IDSorter info = iterator.next();
                    out.append(alphabet.lookupObject(info.getID()) + " ");
                    word++;
                }
                out.append ("\n");
            }
        }


        return out.toString();
    }

    public double modelLogLikelihood() {
        double logLikelihood = 0.0;
        int nonZeroTopics;

        // The likelihood of the model is a combination of a
        // Dirichlet-multinomial for the words in each topic
        // and a Dirichlet-multinomial for the topics in each
        // document.

        // The likelihood function of a dirichlet multinomial is
        //	 Gamma( sum_i alpha_i )	 prod_i Gamma( alpha_i + N_i )
        //	prod_i Gamma( alpha_i )	  Gamma( sum_i (alpha_i + N_i) )

        // So the log likelihood is
        //	logGamma ( sum_i alpha_i ) - logGamma ( sum_i (alpha_i + N_i) ) +
        //	 sum_i [ logGamma( alpha_i + N_i) - logGamma( alpha_i ) ]

        // Do the documents first

        int[] topicCounts = new int[numTopics];
        int[] backgroundTopicalCounts = new int[2];
        double[] topicLogGammas = new double[numTopics];
        double[] backgroundTopicalGammas = new double[2];
        int[] docTopics;

        for (int topic=0; topic < numTopics; topic++) {
            topicLogGammas[ topic ] = Dirichlet.logGammaStirling( alpha[topic] );
        }
        backgroundTopicalGammas[TOPICAL_WORD_INDEX] = Dirichlet.logGammaStirling(1-lambda);
        backgroundTopicalGammas[BACKGROUND_WORD_INDEX] = Dirichlet.logGammaStirling(lambda);
        for (int doc=0; doc < data.size(); doc++) {
            LabelSequence topicSequence =	(LabelSequence) data.get(doc).topicSequence;

            docTopics = topicSequence.getFeatures();

            for (int token=0; token < docTopics.length; token++) {
                if(docTopics[token] == -1){
                    // background word
                    backgroundTopicalCounts[BACKGROUND_WORD_INDEX] ++;
                }else{
                    // topical word
                    topicCounts[ docTopics[token] ]++;
                    backgroundTopicalCounts[TOPICAL_WORD_INDEX] ++;
                }

            }

            for (int topic=0; topic < numTopics; topic++) {
                if (topicCounts[topic] > 0) {
                    logLikelihood += (Dirichlet.logGammaStirling(alpha[topic] + topicCounts[topic]) -
                            topicLogGammas[ topic ]);
                }
            }

            // add log-likelihood for topic-background distribution
            logLikelihood += (Dirichlet.logGammaStirling(1 - lambda + backgroundTopicalCounts[TOPICAL_WORD_INDEX]) -
                    backgroundTopicalGammas[TOPICAL_WORD_INDEX]);

            logLikelihood += (Dirichlet.logGammaStirling(lambda + backgroundTopicalCounts[BACKGROUND_WORD_INDEX]) -
                    backgroundTopicalGammas[BACKGROUND_WORD_INDEX]);

            // subtract the (count + parameter) sum term
            logLikelihood -= Dirichlet.logGammaStirling(alphaSum + docTopics.length);

            // do the same for topic-background
            logLikelihood -= Dirichlet.logGammaStirling(1 + docTopics.length);


            Arrays.fill(topicCounts, 0);
            Arrays.fill(backgroundTopicalCounts, 0);
        }

        // add the parameter sum term
        logLikelihood += data.size() * Dirichlet.logGammaStirling(alphaSum);
        logLikelihood += data.size() * Dirichlet.logGammaStirling(1);



        // And the topics

        // Count the number of type-topic pairs that are not just (logGamma(beta) - logGamma(beta))
        int nonZeroTypeTopics = 0;

        for (int type=0; type < numTypes; type++) {
            // reuse this array as a pointer

            topicCounts = typeTopicCounts[type];

            int index = 0;
            while (index < topicCounts.length &&
                    topicCounts[index] > 0) {
                int topic = topicCounts[index] & topicMask;
                int count = topicCounts[index] >> topicBits;

                nonZeroTypeTopics++;
                logLikelihood += Dirichlet.logGammaStirling(beta + count);

                if (Double.isNaN(logLikelihood)) {
                    logger.warning("NaN in log likelihood calculation");
                    return 0;
                }
                else if (Double.isInfinite(logLikelihood)) {
                    logger.warning("infinite log likelihood");
                    return 0;
                }

                index++;
            }
        }


        for (int topic=0; topic < numTopics; topic++) {
            logLikelihood -=
                    Dirichlet.logGammaStirling( (beta * numTypes) +
                            tokensPerTopic[ topic ] );

            if (Double.isNaN(logLikelihood)) {
                logger.info("NaN after topic " + topic + " " + tokensPerTopic[ topic ]);
                return 0;
            }
            else if (Double.isInfinite(logLikelihood)) {
                logger.info("Infinite value after topic " + topic + " " + tokensPerTopic[ topic ]);
                return 0;
            }

        }

        // background topic for each source
        for (int source=0; source < numSources; source++) {
            for (int type=0; type < numTypes; type++) {
                logLikelihood += Dirichlet.logGammaStirling(betaBackground + typeBackgroundCounts[source][type]);
            }
            logLikelihood -=
                    Dirichlet.logGammaStirling( (betaBackground * numTypes) +
                            backgroundAndTopicalCounts[ source ][BACKGROUND_WORD_INDEX] );

        }

        // logGamma(|V|*beta) for every topic
        logLikelihood +=
                Dirichlet.logGammaStirling(beta * numTypes) * numTopics;

        // logGamma(beta) for all type/topic pairs with non-zero count
        logLikelihood -=
                Dirichlet.logGammaStirling(beta) * nonZeroTypeTopics;

        if (Double.isNaN(logLikelihood)) {
            logger.info("at the end");
        }
        else if (Double.isInfinite(logLikelihood)) {
            logger.info("Infinite value beta " + beta + " * " + numTypes);
            return 0;
        }

        return logLikelihood;
    }

    /** Return a tool for estimating topic distributions for new documents */
    public TopicInferencer getInferencer() {
        throw new UnsupportedOperationException();
    }
    /** Return a tool for estimating topic distributions for new documents of a specific source */
    public TopicInferencer getInferencer(Object source, double[] alpha) {
        int sourceId = sourceToSourceId.get(source);
        return new BackgroundTopicInferencer(
                typeTopicCounts, tokensPerTopic,
                typeBackgroundCounts[sourceId], backgroundAndTopicalCounts[sourceId],
                super.data.get(0).instance.getDataAlphabet(),
                alpha, beta, betaSum, lambda);
    }
    public TopicInferencer getInferencer(Object source) {
        return getInferencer(source, alpha);
    }

    public static void main(String[] args){

        try {
            // deserialize the model
            ObjectInputStream in = new ObjectInputStream(new FileInputStream(args[0]));
            MultiBackgroundTopicModel model = (MultiBackgroundTopicModel) in.readObject();
            // select an inference for a given source, News, Twitter, Meetup, or others specified in
            // the model.sourceToSourceId
            TopicInferencer inferencer = model.getInferencer("News");
            // The inferencer object can be used to infer topic for a particular source.
            // See the last few lines in Mallet tutorial at http://mallet.cs.umass.edu/topics-devel.php
            // For example:
            /*
            double[] topicsDist = inferencer.getSampledDistribution(instance , 100, 1, 5);
            // Note that the last element in the array is the weight of the background topic
            double backgroundWeight = topicsDist[topicsDist.length-1];
            **/


        } catch (IOException e) {
            e.printStackTrace();
            System.err.println("Can't open file " + args[0]);

        } catch (ClassNotFoundException e) {
            e.printStackTrace();
            System.err.println("Error loading class in " + args[0]);
        }

    }


}
