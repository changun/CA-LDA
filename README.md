# CA-LDA
=======
* The model is implemented on top of and requires Mallet 2.0.7's ParallelTopicModel.
* Trained models are available for download:
  * [200 dimensional](https://s3.amazonaws.com/newsfie/CA_LDA_dim_200.bin.gz)
  * [500 dimensional](https://s3.amazonaws.com/newsfie/CA_LDA_dim_500.bin.gz)
* Sample code

``` java
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
            double[] topicsDist = inferencer.getSampledDistribution(instance , 100, 1, 5);
            // Note that the last element in the array is the weight of the background topic
            double backgroundWeight = topicsDist[topicsDist.length-1];



        } catch (IOException e) {
            e.printStackTrace();
            System.err.println("Can't open file " + args[0]);

        } catch (ClassNotFoundException e) {
            e.printStackTrace();
            System.err.println("Error loading class in " + args[0]);
        }
```

