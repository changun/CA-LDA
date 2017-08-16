package cc.mallet.topics.tui;
import cc.mallet.topics.TopicInferencer;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.Instance;
import cc.mallet.util.CommandOption;

import java.io.File;
import java.util.Scanner;
import java.util.concurrent.*;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * Created by changun on 8/15/17.
 */
public class StreamInferTopics {

    static CommandOption.String inferencerFilename = new CommandOption.String
            (StreamInferTopics.class, "inferencer", "FILENAME", true, null,
                    "A serialized topic inferencer from a trained topic model.\n" +
                            "By default this is null, indicating that no file will be read.", null);


    static CommandOption.Double docTopicsThreshold = new CommandOption.Double
            (StreamInferTopics.class, "doc-topics-threshold", "DECIMAL", true, 0.0,
                    "When writing topic proportions per document with --output-doc-topics, " +
                            "do not print topics with proportions less than this threshold value.", null);

    static CommandOption.Integer docTopicsMax = new CommandOption.Integer
            (StreamInferTopics.class, "doc-topics-max", "INTEGER", true, -1,
                    "When writing topic proportions per document with --output-doc-topics, " +
                            "do not print more than INTEGER number of topics.  "+
                            "A negative value indicates that all topics should be printed.", null);
    static CommandOption.Integer numOfThreads = new CommandOption.Integer
            (StreamInferTopics.class, "num-threads", "INTEGER", true, 1,
                    "number of threads", null);

    static CommandOption.Integer numIterations = new CommandOption.Integer
            (StreamInferTopics.class, "num-iterations", "INTEGER", true, 100,
                    "The number of iterations of Gibbs sampling.", null);

    static CommandOption.Integer sampleInterval = new CommandOption.Integer
            (StreamInferTopics.class, "sample-interval", "INTEGER", true, 10,
                    "The number of iterations between saved samples.", null);

    static CommandOption.Integer burnInIterations = new CommandOption.Integer
            (StreamInferTopics.class, "burn-in", "INTEGER", true, 10,
                    "The number of iterations before the first sample is saved.", null);

    static CommandOption.Integer randomSeed = new CommandOption.Integer
            (StreamInferTopics.class, "random-seed", "INTEGER", true, 0,
                    "The random seed for the Gibbs sampler.  Default is 0, which will use the clock.", null);

    public static void main (String[] args) {

        // Process the command-line options
        CommandOption.setSummary (StreamInferTopics.class,
                "Use an existing topic model to infer topic distributions for new documents in a stream mode");
        CommandOption.process (StreamInferTopics.class, args);



        if (inferencerFilename.value == null) {
            System.err.println("You must specify a serialized topic inferencer. Use --help to list options.");
            System.exit(0);
        }



        try {

            final TopicInferencer[] inferencers = new TopicInferencer[numOfThreads.value];
            inferencers[0] =
                    TopicInferencer.read(new File(inferencerFilename.value));
            if (randomSeed.value != 0) {
                inferencers[0].setRandomSeed(randomSeed.value);
            }
            for(int i=1; i< numOfThreads.value; i++){
                inferencers[i] = inferencers[0].copy();

            }






            ExecutorService outputExecutor = Executors.newSingleThreadExecutor();
            final LinkedBlockingQueue<Future<String>> resultQueue = new LinkedBlockingQueue<Future<String>>(10000);
            outputExecutor.execute(new Runnable() {
                @Override
                public void run() {
                    boolean interrupted = false;
                    while(!resultQueue.isEmpty() || !interrupted){

                        try {
                            Future<String> result = resultQueue.take();
                            System.out.println(result.get());
                        } catch (InterruptedException e) {
                            interrupted = true;
                        } catch (ExecutionException e) {
                            e.printStackTrace();
                            System.out.println("error");
                        }
                        System.out.flush();
                }}
            });

            BlockingQueue<Runnable> blockingQueue = new ArrayBlockingQueue<Runnable>(10000);

            ThreadPoolExecutor executor = new ThreadPoolExecutor(numOfThreads.value + 1,
                    numOfThreads.value + 1, 5000, TimeUnit.MILLISECONDS, blockingQueue);
            executor.setRejectedExecutionHandler(new RejectedExecutionHandler() {
                @Override
                public void rejectedExecution(Runnable r,
                                              ThreadPoolExecutor executor) {

                    System.err.println("Waiting for a second !!");
                    try {
                        Thread.sleep(1000);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                    executor.execute(r);
                }
            });


            // Let start all core threads initially
            executor.prestartAllCoreThreads();

            Scanner scan = new Scanner(System.in);
            while (scan.hasNextLine()) {
                final String line = scan.nextLine();
                Future<String> result = executor.submit(new Callable<String>() {
                    @Override
                    public String call() throws Exception {
                        // assume input is in svmlight format
                        int threadId = (int) (Thread.currentThread().getId() % numOfThreads.value);
                        String[] tokenStrings = line.split(" ");
                        // the first element is the name of the input
                        String name = tokenStrings[0];
                        // the id of each feature
                        int tokenIds[] = new int[tokenStrings.length-1];
                        // the occurrence count of each feature
                        int tokenCount[] = new int[tokenStrings.length-1];
                        int total = 0;
                        for (int i=1; i<tokenStrings.length; i++) {
                            String[] f_c = tokenStrings[i].split(":");
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
                        Instance inst = new Instance(new FeatureSequence(inferencers[0].alphabet, tokens), null, name, null);
                        return inferencers[threadId].printInferredDistributions(inst,
                                numIterations.value, sampleInterval.value,
                                burnInIterations.value,
                                docTopicsThreshold.value, docTopicsMax.value);
                    }
                });
                resultQueue.put(result);


            }
            executor.shutdown();
            outputExecutor.shutdown();

        } catch (Exception e) {
            e.printStackTrace();
            System.err.println(e.getMessage());
        }
    }

    }


