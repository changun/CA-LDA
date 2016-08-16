package cc.mallet.topics;


import cc.mallet.types.*;
import cc.mallet.util.Randoms;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.TreeSet;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Created by Cheng-Kang Hsieh on 9/6/15.
 */
public class BackgroundTopicModel extends ParallelTopicModel {
    static final long serialVersionUID = -7837253890188534391L;
    protected static LabelAlphabet newLabelAlphabet (int numTopics) {
        LabelAlphabet ret = new LabelAlphabet();
        for (int i = 0; i < numTopics; i++)
            ret.lookupIndex("topic"+i);
        return ret;
    }
    public static final double DEFAULT_LAMBDA = 0.5;
    // background topic distribution over types
    protected int[] typeBackgroundCounts; // indexed by <feature index>
    // number of background/topical tokens
    protected int[] backgroundAndTopicalCounts;
    // the index of topical/background tokens counts
    public static int TOPICAL_WORD_INDEX = 0, BACKGROUND_WORD_INDEX =1;
    // the pseudo topic for background words
    public int backgroundTopic = ParallelTopicModel.UNASSIGNED_TOPIC;
    // runnables
    BackgroundWorkerRunnable runnables;
    double lambda;
    double betaBackground;
    public BackgroundTopicModel(LabelAlphabet topicAlphabet, double alphaSum, double beta, double betaBackground, double lambda) {
        super(topicAlphabet, alphaSum, beta);
        this.lambda = lambda;
        this.betaBackground = betaBackground;
    }
    public BackgroundTopicModel (int numberOfTopics) {
        this (numberOfTopics, 1, DEFAULT_BETA, DEFAULT_BETA, DEFAULT_LAMBDA);
    }


    public BackgroundTopicModel (int numberOfTopics, double alphaSum, double beta, double betaBackground, double lambda) {
        this (newLabelAlphabet (numberOfTopics), alphaSum, beta, betaBackground, lambda);
    }

    public void addInstances (InstanceList training) {

        alphabet = training.getDataAlphabet();
        numTypes = alphabet.size();

        betaSum = beta * numTypes;

        typeTopicCounts = new int[numTypes][];

        typeBackgroundCounts = new int[training.getAlphabet().size()];
        backgroundAndTopicalCounts = new int[2];

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

            FeatureSequence tokens = (FeatureSequence) instance.getData();
            LabelSequence topicSequence =
                    new LabelSequence(topicAlphabet, new int[ tokens.size() ]);

            int[] topics = topicSequence.getFeatures();
            for (int position = 0; position < topics.length; position++) {
                if(random.nextUniform() > lambda){
                    int topic = random.nextInt(numTopics);
                    topics[position] = topic;
                    backgroundAndTopicalCounts[TOPICAL_WORD_INDEX]++;

                }else{
                    topics[position] = backgroundTopic;
                    typeBackgroundCounts[tokens.getIndexAtPosition(position)]++;

                    backgroundAndTopicalCounts[BACKGROUND_WORD_INDEX]++;
                }
            }

            TopicAssignment t = new TopicAssignment(instance, topicSequence);
            data.add (t);
        }

        buildInitialTypeTopicCounts();
        initializeHistograms();

    }
    public void sumBackgroundTopicalCounts (BackgroundWorkerRunnable[] runnables) {

        // Clear the topic totals
        Arrays.fill(typeBackgroundCounts, 0);
        Arrays.fill(backgroundAndTopicalCounts, 0);
        for (int thread = 0; thread < numThreads; thread++) {

            // Handle the background topic distribution
            int[] sourceCounts = runnables[thread].getTypeBackgroundCounts();
            for (int type = 0; type < numTypes; type++) {
                typeBackgroundCounts[type] += sourceCounts[type];
            }

            // Now handle the background/topical token counts
            int[] souceTokenCounts = runnables[thread].getBackgroundAndTopicalCounts();
            for (int i = 0; i < souceTokenCounts.length; i++){
                backgroundAndTopicalCounts[i] += souceTokenCounts[i];
            }

        }

    }

    public void estimate () throws IOException {

        long startTime = System.currentTimeMillis();

        BackgroundWorkerRunnable[] runnables = new BackgroundWorkerRunnable[numThreads];

        int docsPerThread = data.size() / numThreads;
        int offset = 0;

        assert (numThreads > 1);
        for (int thread = 0; thread < numThreads; thread++) {
            int[] runnableTotals = new int[numTopics];
            System.arraycopy(tokensPerTopic, 0, runnableTotals, 0, numTopics);

            int[][] runnableCounts = new int[numTypes][];
            for (int type = 0; type < numTypes; type++) {
                int[] counts = new int[typeTopicCounts[type].length];
                System.arraycopy(typeTopicCounts[type], 0, counts, 0, counts.length);
                runnableCounts[type] = counts;
            }
            int[] runnableTypeBackgroundCounts = new int[numTypes];
            System.arraycopy(typeBackgroundCounts, 0, runnableTypeBackgroundCounts, 0, runnableTypeBackgroundCounts.length);

            int[] runnableBackgroundAndTopicalCounts = new int[2];
            System.arraycopy(backgroundAndTopicalCounts, 0, runnableBackgroundAndTopicalCounts, 0, runnableBackgroundAndTopicalCounts.length);


            // some docs may be missing at the end due to integer division
            if (thread == numThreads - 1) {
                docsPerThread = data.size() - offset;
            }

            Randoms random = null;
            if (randomSeed == -1) {
                random = new Randoms();
            }
            else {
                random = new Randoms(randomSeed);
            }

            runnables[thread] = new BackgroundWorkerRunnable(numTopics,
                    alpha, alphaSum, beta, betaBackground, lambda,
                    random, data,
                    runnableCounts, runnableTotals,
                    runnableTypeBackgroundCounts, runnableBackgroundAndTopicalCounts,
                    offset, docsPerThread);

            runnables[thread].initializeAlphaStatistics(docLengthCounts.length);

            offset += docsPerThread;

        }


        ExecutorService executor = Executors.newFixedThreadPool(numThreads);

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

            for (int thread = 0; thread < numThreads; thread++) {
                if (iteration > burninPeriod && optimizeInterval != 0 &&
                        iteration % saveSampleInterval == 0) {
                    runnables[thread].collectAlphaStatistics();
                }

                logger.fine("submitting thread " + thread);
                executor.submit(runnables[thread]);
                //runnables[thread].run();
            }

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

                System.arraycopy(typeBackgroundCounts, 0, runnables[thread].getTypeBackgroundCounts(), 0, typeBackgroundCounts.length);
                System.arraycopy(backgroundAndTopicalCounts, 0, runnables[thread].getBackgroundAndTopicalCounts(), 0, backgroundAndTopicalCounts.length);

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
    }
    public double modelLogLikelihood() {
        throw new UnsupportedOperationException();
    }

    /** Get the smoothed distribution over topics for a topic sequence,
     * which may be from the training set or from a new instance with topics
     * assigned by an inferencer.
     */
    public double[] getTopicProbabilities(LabelSequence topics) {
        double[] topicDistribution = new double[numTopics+1];

        // Loop over the tokens in the document, counting the current topic
        //  assignments.
        for (int position = 0; position < topics.getLength(); position++) {
            int topic = topics.getIndexAtPosition(position);
            if(topic != backgroundTopic){
                topicDistribution[ topics.getIndexAtPosition(position) ]++;
            }else{
                topicDistribution[numTopics]++;
            }
        }

        // Add the smoothing parameters and normalize
        double sum = 0.0;
        for (int topic = 0; topic < numTopics; topic++) {
            topicDistribution[topic] += alpha[topic];
            sum += topicDistribution[topic];
        }

        // And normalize
        for (int topic = 0; topic < numTopics; topic++) {
            topicDistribution[topic] /= sum;
        }

        return topicDistribution;
    }
    /**
     *  Return an array of sorted sets (one set per topic). Each set
     *   contains IDSorter objects with integer keys into the alphabet.
     *   To get direct access to the Strings, use getTopWords().
     */
    protected TreeSet<IDSorter> getSortedBackgroundWords (int[] typeBackgroundCounts) {

        TreeSet<IDSorter> sortedWords = new TreeSet<IDSorter>();


        // Collect counts
        for (int type = 0; type < numTypes; type++) {
            if(typeBackgroundCounts[type] > 0){
                sortedWords.add(new IDSorter(type, typeBackgroundCounts[type]));
            }

        }
        return sortedWords;
    }
    public String displayTopWords (int numWords, boolean usingNewLines) {

        StringBuilder out = new StringBuilder();

        ArrayList<TreeSet<IDSorter>> topicSortedWords = getSortedWords();
        topicSortedWords.add(getSortedBackgroundWords(typeBackgroundCounts));
        double backgroundRatio = (double) backgroundAndTopicalCounts[BACKGROUND_WORD_INDEX] /
                (double)(backgroundAndTopicalCounts[BACKGROUND_WORD_INDEX] + backgroundAndTopicalCounts[TOPICAL_WORD_INDEX]);
        // Print results for each topic
        for (int topic = 0; topic < topicSortedWords.size(); topic++) {
            TreeSet<IDSorter> sortedWords = topicSortedWords.get(topic);
            int word = 1;
            Iterator<IDSorter> iterator = sortedWords.iterator();
            String alphaStr = topic < numTopics ? formatter.format(alpha[topic]) : formatter.format(backgroundRatio);
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

        return out.toString();
    }
    /** Return a tool for estimating topic distributions for new documents */
    public TopicInferencer getInferencer() {
        return new BackgroundTopicInferencer(
                typeTopicCounts, tokensPerTopic,
                typeBackgroundCounts, backgroundAndTopicalCounts,
                data.get(0).instance.getDataAlphabet(),
                alpha, beta, betaSum, lambda);
    }
    public void topicPhraseXMLReport(PrintWriter out, int numWords) {
        int numTopics = this.getNumTopics();
        gnu.trove.TObjectIntHashMap<String>[] phrases = new gnu.trove.TObjectIntHashMap[numTopics];
        Alphabet alphabet = this.getAlphabet();

        // Get counts of phrases
        for (int ti = 0; ti < numTopics; ti++)
            phrases[ti] = new gnu.trove.TObjectIntHashMap<String>();
        for (int di = 0; di < this.getData().size(); di++) {
            TopicAssignment t = this.getData().get(di);
            Instance instance = t.instance;
            FeatureSequence fvs = (FeatureSequence) instance.getData();
            boolean withBigrams = false;
            if (fvs instanceof FeatureSequenceWithBigrams) withBigrams = true;
            int prevtopic = -1;
            int prevfeature = -1;
            int topic = -1;
            StringBuffer sb = null;
            int feature = -1;
            int doclen = fvs.size();
            for (int pi = 0; pi < doclen; pi++) {
                feature = fvs.getIndexAtPosition(pi);
                topic = this.getData().get(di).topicSequence.getIndexAtPosition(pi);
                if (topic != backgroundTopic && topic == prevtopic && (!withBigrams || ((FeatureSequenceWithBigrams)fvs).getBiIndexAtPosition(pi) != -1)) {
                    if (sb == null)
                        sb = new StringBuffer (alphabet.lookupObject(prevfeature).toString() + " " + alphabet.lookupObject(feature));
                    else {
                        sb.append (" ");
                        sb.append (alphabet.lookupObject(feature));
                    }
                } else if (sb != null) {
                    String sbs = sb.toString();
                    //logger.info ("phrase:"+sbs);
                    if (phrases[prevtopic].get(sbs) == 0)
                        phrases[prevtopic].put(sbs,0);
                    phrases[prevtopic].increment(sbs);
                    prevtopic = prevfeature = -1;
                    sb = null;
                } else {
                    prevtopic = topic;
                    prevfeature = feature;
                }
            }
        }
        // phrases[] now filled with counts

        // Now start printing the XML
        out.println("<?xml version='1.0' ?>");
        out.println("<topics>");

        ArrayList<TreeSet<IDSorter>> topicSortedWords = getSortedWords();
        double[] probs = new double[alphabet.size()];
        for (int ti = 0; ti < numTopics; ti++) {
            out.print("  <topic id=\"" + ti + "\" alpha=\"" + alpha[ti] +
                    "\" totalTokens=\"" + tokensPerTopic[ti] + "\" ");

            // For gathering <term> and <phrase> output temporarily
            // so that we can get topic-title information before printing it to "out".
            ByteArrayOutputStream bout = new ByteArrayOutputStream();
            PrintStream pout = new PrintStream (bout);
            // For holding candidate topic titles
            AugmentableFeatureVector titles = new AugmentableFeatureVector(new Alphabet());

            // Print words
            int word = 1;
            Iterator<IDSorter> iterator = topicSortedWords.get(ti).iterator();
            while (iterator.hasNext() && word < numWords) {
                IDSorter info = iterator.next();
                pout.println("	<word weight=\""+(info.getWeight()/tokensPerTopic[ti])+"\" count=\""+Math.round(info.getWeight())+"\">"
                        + alphabet.lookupObject(info.getID()) +
                        "</word>");
                word++;
                if (word < 20) // consider top 20 individual words as candidate titles
                    titles.add(alphabet.lookupObject(info.getID()), info.getWeight());
            }

			/*
			for (int type = 0; type < alphabet.size(); type++)
				probs[type] = this.getCountFeatureTopic(type, ti) / (double)this.getCountTokensPerTopic(ti);
			RankedFeatureVector rfv = new RankedFeatureVector (alphabet, probs);
			for (int ri = 0; ri < numWords; ri++) {
				int fi = rfv.getIndexAtRank(ri);
				pout.println ("	  <term weight=\""+probs[fi]+"\" count=\""+this.getCountFeatureTopic(fi,ti)+"\">"+alphabet.lookupObject(fi)+	"</term>");
				if (ri < 20) // consider top 20 individual words as candidate titles
					titles.add(alphabet.lookupObject(fi), this.getCountFeatureTopic(fi,ti));
			}
			*/

            // Print phrases
            Object[] keys = phrases[ti].keys();
            int[] values = phrases[ti].getValues();
            double counts[] = new double[keys.length];
            for (int i = 0; i < counts.length; i++)	counts[i] = values[i];
            double countssum = MatrixOps.sum (counts);
            Alphabet alph = new Alphabet(keys);
            RankedFeatureVector rfv = new RankedFeatureVector(alph, counts);
            int max = rfv.numLocations() < numWords ? rfv.numLocations() : numWords;
            for (int ri = 0; ri < max; ri++) {
                int fi = rfv.getIndexAtRank(ri);
                pout.println ("	<phrase weight=\""+counts[fi]/countssum+"\" count=\""+values[fi]+"\">"+alph.lookupObject(fi)+	"</phrase>");
                // Any phrase count less than 20 is simply unreliable
                if (ri < 20 && values[fi] > 20)
                    titles.add(alph.lookupObject(fi), 100*values[fi]); // prefer phrases with a factor of 100
            }

            // Select candidate titles
            StringBuffer titlesStringBuffer = new StringBuffer();
            rfv = new RankedFeatureVector(titles.getAlphabet(), titles);
            int numTitles = 10;
            for (int ri = 0; ri < numTitles && ri < rfv.numLocations(); ri++) {
                // Don't add redundant titles
                if (titlesStringBuffer.indexOf(rfv.getObjectAtRank(ri).toString()) == -1) {
                    titlesStringBuffer.append (rfv.getObjectAtRank(ri));
                    if (ri < numTitles-1)
                        titlesStringBuffer.append (", ");
                } else
                    numTitles++;
            }
            out.println("titles=\"" + titlesStringBuffer.toString() + "\">");
            out.print(bout.toString());
            out.println("  </topic>");
        }
        out.println("</topics>");
    }



}
