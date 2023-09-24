# BERTopic Streamlit Demo

This is a demo implementation of a web app for searching for thematic trends over time in a corpus of 3,747 abstracts drawn from a PubMed search for the word "oxycontin."  The demo is intended to give a sense of the usability and power of topic models created with the [BERTopic](https://maartengr.github.io/BERTopic/index.html) library, which was developed by [Maarten Grootendorst](https://arxiv.org/abs/2203.05794).

## Key Takeaways:

* This code could run on a secure, air-gap-ed instance.  All dependencies can be downloaded and run locally or on a VPC
* This approach is modular.  The language model used for sentence transformation can be replaced as better models become available.
* This approach is flexible and tunable with very low effort and relatively little time (though it does require access to GPUs).
* LLMs can be used in fine tuning if desired to make topic names more readable and to generate brief summaries of each topic cluster

The abstracts were embedded using the "paraphrase-MiniLM-L6-v2" language model, which was downloaded through the [HuggingFace API](https://huggingface.co/models).

Dimensionality reduction was done with the UMAP library.

Topic clustering was completed with the HDBSCAN Library.

Topic extraction from the clustered documents was executed with the PartOfSpeech representation model.


## License

[MIT](https://choosealicense.com/licenses/mit/)