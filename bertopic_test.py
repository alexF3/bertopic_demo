import streamlit as st
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd
import plotly.express as px
import html
import streamlit.components.v1 as components
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load a pre-trained Sentence Transformer model
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Load a pre-trained BERTopic model (you can replace this with your own trained model)
topic_model = BERTopic.load("data/bertopic_model_oxycontin_pos",model)

# Load the original csv from which the model was transformed
df = pd.read_csv('data/oxycontin_with_topics_pos.csv')
df['date'] = pd.to_datetime(df['date'])


# Streamlit App
st.title("BERTopic Med Journal Article Theme Extraction")

st.image("data/forest_trees.jpg")

st.write("This is a demo of a sentence transformer based search engine for discovering articles with topics similar to user input text among a corpus of 3,747 abstracts drawn from a PubMed search for the word 'Oxycontin.'")
st.write("The demo uses the Python library BERTopic to embed the user input text and compare it to a pre-trained model.  All embeddings are conducted with the paraphrase-MiniLM-L6-v2 language model.")
st.write("It is important to note that the code generating these results makes no use of MeSH terms or any other categorization within PubMed metadata.  These results are solely the result of sentence transformations, dimension reduction, clustering, and topic extraction using the BERTopic library and selected language model.")
 
# components.iframe('/content/drive/MyDrive/a_datasets/hierarch_cluster.html')

# Text input from the user
user_text = st.text_area("Enter text you'd like to find articles about here:")

if user_text:
    # Extract topics from the user-provided text
    # topics, probs = topic_model.transform([user_text])

    topic, prob = topic_model.find_topics(user_text)
    findings_frame = pd.DataFrame({'topic':topic,'prob':prob})
    findings_frame = findings_frame[findings_frame.prob>0.1]

    try:
        st.header(f"{len(df[df.topic.isin(findings_frame.topic.tolist())])} Articles with Similar Themes:")

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

        st.write("Word cloud based on TF-IDF of the clusters of articles with themes that matched the entered text:")
        st.pyplot(wc_fig)


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
            title='In-Theme Article Counts by Year',
            hover_data=['Hover Text']
        )

        # Set the x-axis date range to 1996 through 2023
        fig.update_xaxes(title_text='Year', range=[1996, 2023])
        fig.update_yaxes(range=[0,yearly_row_counts['Article Count'].max()+5])

        fig.update_yaxes(title_text='Article Count')






        st.plotly_chart(fig, use_container_width=True)

        st.title("Articles and PubMed links:")


        for row in df[df.topic.isin(findings_frame.topic.tolist())].itertuples():
            st.write(f"{row.title}")

            st.write(f"https://pubmed.ncbi.nlm.nih.gov/{row.pubmed_id}")

    except:
        st.write("We're sorry.  There were no topics in the corpus similar enough to your input to display.")

