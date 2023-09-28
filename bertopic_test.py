import streamlit as st
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd
import plotly.express as px
import html
import streamlit.components.v1 as components
from wordcloud import WordCloud
import matplotlib.pyplot as plt


st.set_page_config(page_title="Oxycontin Research Exploration with BERTopic")

# Load a pre-trained Sentence Transformer model
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Load a pre-trained BERTopic model (you can replace this with your own trained model)
topic_model = BERTopic.load("data/bertopic_model_oxycontin_pos",model)

# Load the original csv from which the model was transformed
df = pd.read_csv('data/oxycontin_with_topics_pos.csv')
df['date'] = pd.to_datetime(df['date'])

tab1, tab2 = st.tabs(["Demo", "About"])

with tab1:

    # Streamlit App
    st.title("BERTopic Med Journal Article Theme Extraction")

    st.image("data/forest_trees.jpg")

    st.write("This is a demo of a topic modeling library using a pre-trained language model to surface articles with topics similar to user input text among a corpus of 3,747 abstracts drawn from a PubMed search for the word 'Oxycontin.'")
    st.markdown("You can read more about how this works in the [BERTopic Documentation](https://maartengr.github.io/BERTopic/index.html).")
    st.write("This approach often works better than more traditional NLP algorithms and requires a lot less pre-processing. Check out the results for yourself!")

    ## Demo example
    ##
    st.header("Try an experiment:") 
    st.markdown('* Type "marketing and advertising" into the search box to explore the Oxycontin article corpus')
    st.markdown("* Compare thoes results to what you'd get from a [PubMed search](https://pubmed.ncbi.nlm.nih.gov/?term=%28%28oxycontin%29+AND+%28marketing%29%29+AND+%28advertising%29&sort=&filter=simsearch1.fha) for the Oxycontin marketing and advertising.")
    st.header('**Which do you find more useful?** :i_love_you_hand_sign:')
    st.write('(Hint: PubMed will give you 3 articles...this does better)')

    # Text input from the user
    user_text = st.text_area("Enter text you'd like to find Oxycontin articles about here:")

    if user_text:
        # Extract topics from the user-provided text
        # topics, probs = topic_model.transform([user_text])

        topic, prob = topic_model.find_topics(user_text)
        findings_frame = pd.DataFrame({'topic':topic,'prob':prob})
        findings_frame = findings_frame[findings_frame.prob>0.1]

        try:
            st.header(f"BERTopic Search Found {len(df[df.topic.isin(findings_frame.topic.tolist())])} Articles with Similar Themes:")

            ## Create wordcloud based on TF-IDF of each of the matching topic clusters
            ##

            st.set_option('deprecation.showPyplotGlobalUse', False)     
            topic_wc, prob_wc = topic_model.find_topics(user_text)
            findings_frame = pd.DataFrame({'topic_wc':topic_wc,'prob_wc':prob_wc})
            findings_frame = findings_frame[findings_frame.prob_wc>0.1]

            def create_wordcloud(model, topic_wc):
                text = {}
                for t in  topic_wc:
                    new_text =  {word: value for word, value in topic_model.get_topic(t)}
                    text = text | new_text
                # text = {word: value for word, value in model.get_topic(5)}
                wc = WordCloud(background_color="white", max_words=1000)
                wc.generate_from_frequencies(text)
                plt.imshow(wc, interpolation="bilinear")
                plt.axis("off")
                # plt.show()
            wc_fig = create_wordcloud(topic_model, topic_wc=findings_frame.topic_wc.tolist())

            ## Display word cloud
            ##

            st.write("Word cloud based on TF-IDF of the clusters of articles with themes that matched the entered text:")
            st.pyplot(wc_fig)

            ## Create line plot of articles from matching themes per year
            ##

            # Convert the date column to a pandas datetime object
            df['date'] = pd.to_datetime(df['date'])

            # Filter dataframe down to only entries in the topics that match the user_text
            topic, prob = topic_model.find_topics(user_text)
            findings_frame = pd.DataFrame({'topic':topic,'prob':prob})
            findings_frame = findings_frame[findings_frame.prob>0.1]
            df = df[df.topic.isin(findings_frame.topic.tolist())]

            # Extract the month and year from the date column
            df['Year'] = df['date'].dt.year

            # Group by Year and count the rows
            yearly_row_counts = df.groupby(['Year']).size().reset_index(name='Article Count')

            # Create custom hover text with titles separated by line breaks for each year
            def create_hover_text(year):
                titles = df[df['Year'] == year]['title'].tolist()
                titles_html = "<br>".join(html.escape(title) for title in titles)
                return f"Year: {year}<br>Article Count: {yearly_row_counts[yearly_row_counts['Year'] == year]['Article Count'].values[0]}<br>Titles:<br>{titles_html}"

            yearly_row_counts['Hover Text'] = yearly_row_counts.apply(lambda row: create_hover_text(row['Year']), axis=1)

            # Create a line plot using Plotly Express with custom hover text and x-axis date range
            fig = px.line(
                yearly_row_counts,
                x='Year',
                y='Article Count',
                title='In-Theme Article Counts by Year via BERTopic Search',
                hover_data=['Hover Text']
            )

            # Set the x-axis date range to 1996 through 2023
            fig.update_xaxes(title_text='Year', range=[1996, 2023])
            fig.update_yaxes(range=[0,yearly_row_counts['Article Count'].max()+5])

            fig.update_yaxes(title_text='Article Count')

            st.plotly_chart(fig, use_container_width=True)

            ## Display article titles and pubmed links
            ##
            st.title("BERTopic App Search Results:")
            st.markdown('([Click Here](https://pubmed.ncbi.nlm.nih.gov/?term=%28%28oxycontin%29+AND+%28marketing%29%29+AND+%28advertising%29&sort=&filter=simsearch1.fha) to see PubMed results for the demo search for oxycontin marketing and advertising.)')
            st.write('')
            st.write('')
            for row in df[df.topic.isin(findings_frame.topic.tolist())].itertuples():
                st.write(f"{row.title}")

                st.write(f"https://pubmed.ncbi.nlm.nih.gov/{row.pubmed_id}")

        except:
            st.write("We're sorry.  There were no topics in the corpus similar enough to your input to display.")

with tab2:

    st.markdown("""# About this Demo

### BERTopic Overview
This demo uses a pre-trained topic model created using the [BERTopic](https://maartengr.github.io/BERTopic/index.html) Python library.  The library uses a set of modular tools to facilitate the use of language models from the [HuggingFace API](https://huggingface.co/models) in surfacing themes in free text corpora.

Bertopic was developed by [Maarten Grootendorst](https://www.maartengrootendorst.com/) and introduced in [this](https://arxiv.org/abs/2203.05794) 2022 paper.  The most important thing that you need to know about BERTopic is that it's based on an NLP approach called Bidirectional Encoder Representations from Transformers (BERT) which was [introduced](https://arxiv.org/abs/1810.04805) by scientists from Google in 2018.  BERT differs from many other topic modeling approaches in that it retains the context before and after a given word.  Traditional NLP methodologies like Latent Derichlet Allocation (LDA) use a "bag-of-words" approach in which the document "Dog bites man" encodes the same way as the document "Man bites dog."  BERT avoids this problem.

At a high level, BERTopic breaks down into a few steps:
1. Transform the corpus of documents beings studied into vectors ("embeddings") using a language model the user selects
2. Reduce the dimensionality of the resulting vectors
3. Cluster the reduced vectors 
4. Apply TF-IDF to the resulting topical clusters to weight the words that appear in each
5. Use one or more representation models to select the words that best represent the themes in each cluster.

Each of these steps involves a variety of algorithmic options and hyperparemeters and it's worthwhile to go through the documentation in detail.  

### To me, there are two things that are especially exciting:

First, in transforming the documents to embeddings (step 1), you can select any language model from the HuggingFace API.  That means that you can readily upgrade your code as new and improved language models become available.

Second, the representation model phase (step 5) you can choose from a variety of different approaches to selecting the words that best define a thematic cluster of documents.  You have the option to provide a set of candidate topics and have the algorithm attempt to assign clusters to them.  If you're using an open or non-proprietary corpus you can even use the OpenAI API to provide summarizations of the documents in each topic. 

BERTopic can be used to create modular, reusable pipelines that are readily tunable and applicable to a broad variety of tasks.  **Applications like this one could be hosted in secure VPCs or air gapped environments without access to the web.**

## Overview of the approach behind this demo

**Objective:**

Imagine you're a senior FDA official overseeing panels of expert physicians who advise the administration on policy matters around specific drugs or devices.  You're not able to read every paper on every product, but you want to come into meetings with a sense of where the research around the given drug is in terms of themes and trends.  When faced with potentially dangerous drugs like Purdue Pharma's Oxycontin, you want to be sure that you have an independent sense of whether the broader medical research community is raising any alarms.

**You don't have time to read three thousand abstracts.  You want to rapidly get a sense of emerging themes over time.**

BERTopic can help with that!

**Data:**

A collection of 3,737 abstracts was drawn from [PubMed](https://pubmed.ncbi.nlm.nih.gov/) using the search term "Oxycontin."  These were downloaded as an .nbib file and reformatted as a Pandas dataframe using the nbib Python library.

**Methods:**

This app was developed using [Google Colab](https://colab.research.google.com) for free access to the GPUs needed to conduct embeddings relatively quickly.  The notebook used can be viewed and forked [here](https://github.com/alexF3/BERTopic_overview/blob/main/bertopic_research/BERTopic_demo_sourcecode_21sep23.ipynb).

I used the sentence transformer "all-MiniLM-L6-v2" to produce embeddings of the abstracts in the corpus.  The BERT approach requires little to no pre-processing, which makes for fast development cycles.  I used the UMAP library to reduce the dimensions of the corpus embedding and the HDBSCAN library to cluster the resulting vectors with a minimum cluster size of 20 documents.  I then used the PartOfSpeech representation model to select the most important words in each topical cluster.

I was able to save the resulting topic model without the embeddings, which came in at under 8 MB.""")
    st.image("data/cluster_view.png")

    st.markdown("""
For the demo, I set up a Streamlit app to transform user input text in the same way that the embeddings the model is based on were transformed.  BERTopic's find_topics method is then used to find topics in the model with a probability of matching of at least 0.1 (this is tunable depending on whether you want to cast a wider or narrower net).  

With matching topics captured, I then use the wordcloud library to generate a word cloud of the highest TF-IDF words in all of the documents in each topic.

Next, I use metadata from each of the topic matched abstracts to created a plotly line chart of annual occurances of matched topics since Oxycontin's introduction in 1996.

Finally, the titles and hyperlinks to PubMed pages for each matched article are presented.





""")
