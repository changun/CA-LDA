# CA-LDA
=======
* The model is implemented on top of and requires Mallet 2.0.7's ParallelTopicModel.
* Trained models are available for download:
  * [200 dimensional](https://s3.amazonaws.com/newsfie/CA_LDA_dim_200.bin.gz)
  * [500 dimensional](https://s3.amazonaws.com/newsfie/CA_LDA_dim_500.bin.gz)
* [Demo app](https://github.com/changun/CA-LDA/blob/master/src/cc/mallet/examples/RunContextAwareLDA.java)

  ```bash
  java -classpath [CLASSES_LOCATION] cc.mallet.examples.RunContextAwareLDA [MODEL_FILE] [DOCUMENT] [CONTEXT_NAME]
  ```
  * Sample result from inferring topics in one of the EnronSent document (using Mail context)
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

