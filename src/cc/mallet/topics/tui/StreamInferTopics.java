package cc.mallet.topics.tui;
import cc.mallet.topics.TopicInferencer;
import cc.mallet.types.Instance;
import cc.mallet.util.CommandOption;

import java.io.File;
import java.util.Scanner;
import java.util.concurrent.*;

/**
 * A multi-threaded streaming inferencer. It reads SVMLight input from the STDIN and output results to STDOUT.
 * The STDOUT is by default buffered, but when encounter an empty line, it will "flush" the STDOUT.
 * Created by changun on 8/15/17.
 */
public class StreamInferTopics {

    public static final String FLUSH_SIGNAL = "";
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
            final LinkedBlockingQueue<Future<String>> resultQueue = new LinkedBlockingQueue<Future<String>>(100000);
            outputExecutor.execute(new Runnable() {
                @Override
                public void run() {
                    long outputLinesCount = 0;
                    boolean interrupted = false;
                    while((!resultQueue.isEmpty()) || (!interrupted)){
                        System.err.println("Run");
                        Future<String> result;
                        // try to get a future result
                        try {
                            result = resultQueue.take();
                        } catch (InterruptedException e) {
                            interrupted = true;
                            continue;
                        }
                        // try to get output result until success or there is an execution exception
                        while (true) {
                            try {
                                String resultStr = result.get();
                                if (resultStr.equals(FLUSH_SIGNAL)){
                                    System.out.flush();
                                    System.err.println("FLUSH STDOUT");
                                }else {
                                    System.out.println(resultStr);
                                }
                                outputLinesCount += 1;
                                break;
                            } catch (InterruptedException e) {
                                interrupted = true;
                            } catch (ExecutionException e) {
                                e.printStackTrace();
                                System.out.println("ERROR");
                                outputLinesCount += 1;
                                break;
                            }

                        }
                        if (Thread.interrupted()) {
                            interrupted = true;
                        }
                        if((outputLinesCount+1) % 10000 == 0){
                            System.out.flush();
                        }


                }
                System.out.flush();
                }
            });

            ExecutorService executor = Executors.newFixedThreadPool(numOfThreads.value);


            Scanner scan = new Scanner(System.in);

            while (scan.hasNextLine()) {
                final String line = scan.nextLine();
                Future<String> result = executor.submit(new Callable<String>() {
                    @Override
                    public String call() throws Exception {
                        // an empty serves as a FLUSH signal
                        if(line.equals(FLUSH_SIGNAL)){
                            return FLUSH_SIGNAL;
                        }else {
                            // assume input is in svmlight format
                            int threadId = (int) (Thread.currentThread().getId() % numOfThreads.value);
                            Instance instance = SVMLightReader.parseLine(line, inferencers[0].alphabet);
                            return inferencers[threadId].printInferredDistributions(instance,
                                    numIterations.value, sampleInterval.value,
                                    burnInIterations.value,
                                    docTopicsThreshold.value, docTopicsMax.value);
                        }
                    }
                });
                while(!resultQueue.offer(result)) {
                    Thread.sleep(1);
                }


            }
            executor.shutdown();
            System.err.println("Inference executor down");
            outputExecutor.shutdownNow();
            System.err.println("Output down");

        } catch (Exception e) {
            e.printStackTrace();
            System.err.println(e.getMessage());
        }
    }

    }


