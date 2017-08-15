# CA-LDA
=======
* The model is implemented on top of and requires Mallet 2.0.7's ParallelTopicModel.
* Trained models are available for download:
  * [200 dimensional](https://s3.amazonaws.com/newsfie/CA_LDA_dim_200.bin.gz)
  * [500 dimensional](https://s3.amazonaws.com/newsfie/CA_LDA_dim_500.bin.gz)

### Input/Output
* Given a document from one of the context it has been trained on (e.g. Mail, Meetup, Twitter, and News), CA-LDA returns the **K-dimensional topic distribution** of the document along with the proportion of the background words in the document.
* The returned **K-dimensional topic distribution** can be used to estimate the *similarity* between two documents from different context **with the influence of the context-depandent background words removed.**
* *Cosine similairty* is recommended similarity metric.

### Usage
#### Preprocessing
* Create a preprocessing ```Pipe```. 
  * The pipe structure needs to be exactly the same as the one we used when training the model (See [example](https://github.com/changun/CA-LDA/blob/master/src/cc/mallet/examples/RunContextAwareLDA.java#L24).)
* Put raw documents into an ```InstanceList``` throuhg the pipe (See [example](https://github.com/changun/CA-LDA/blob/master/src/cc/mallet/examples/RunContextAwareLDA.java#L81).)

#### Inference
* Load the model using ```ObjectStreamInput.readObject()```
* Call ```model.getInferencer(contextName)``` to get a ```TopicInferencer``` for a specific context.
* Infer the topic distribution by calling ```inferencer.getSampledDistribution(instance, 100, 1, 5)```.
* The function returns a ```dobule[]``` of length **K+1** which consists of the distribution of each **K** topic plus the proportion of the background words (at the last element of the array).
  * Usually, we discard the background proportion and only use the K-dimentional topic distribution to estimate the document similarity

#### Demo
* For details, please see [Demo App](https://github.com/changun/CA-LDA/blob/master/src/cc/mallet/examples/RunContextAwareLDA.java)

  ```bash
  java -classpath [CLASSES_LOCATION] cc.mallet.examples.RunContextAwareLDA [MODEL_FILE] [DOCUMENT] [CONTEXT_NAME]
  ```
  * Sample result from inferring topics in one of the EnronSent document (using *Mail* context)
  ``` bash
  > java -classpath CALDA.jar cc.mallet.examples.RunContextAwareLDA CA_LDA_dim_500.bin.gz enronsent04 Mail
  
  Loading model ...
  
  Found 4 contexts: Mail, Meetup, News, Twitter
  
  Topic distribution:
  0.044 of Topic 217: energi power solar electr gas renew fuel wind util effici
  0.029 of Topic 159: messag send account list inform receiv contact pleas updat issu
  0.025 of Topic 424: invest stock market investor compani fund financi firm report ceo
  0.023 of Topic 279: law legal court rule lawyer protect licens patent copyright contract
  0.023 of Topic 395: program fund propos feder provid public budget administr nation bill
  ....
  
  ```

