"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.probability import FreqDist
from wordcloud import WordCloud, ImageColorGenerator

# for data cleaning and stopwords removal
import re
import nltk
import string
import unidecode
from nltk import pos_tag
from nlppreprocess import NLP
nlp = NLP()
from nltk.stem.wordnet import WordNetLemmatizer

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

def clean_tweet(text):
    
    # lower-case all c
    text=text.lower()
    
    # remove twitter handles
    text= re.sub(r'@\S+', '',text) 
    
    # remove urls and rt
    text = re.sub(r'^rt ','', re.sub(r'https://t.co/\w+', '', text).strip()) 
    
      
    # replace unidecode characters
    text=unidecode.unidecode(text)
      
    # regex only keeps characters
    text= re.sub(r"[^a-zA-Z+']", ' ',text)
    
    # keep words with length>1 only
    text=re.sub(r'/^[a-zA-Z]{2,}$/', ' ', text+' ') 
    
    #removing pnctuaions
    def remove_punctuation(message):
        return ''.join([l for l in message if l not in string.punctuation])
    
    text = remove_punctuation(text)

    
    # regex removes repeated spaces, strip removes leading and trailing spaces
    text= re.sub("\s[\s]+", " ",text).strip()  
    
    # removing uncessessary stopwords
    nlp_stopwords = NLP(remove_stopwords=True, 
                            remove_numbers=True, remove_punctuations=False) 
    text = nlp_stopwords.process(text)
    
    # tokenisation
    # We used the split method instead of the word_tokenise library because our tweet is already clean at this point
    # and the twitter data is not complicated
    text = text.split() 

    # POS 
    # Part of Speech tagging is essential to ensure Lemmatization perfoms well.
    pos = pos_tag(text)

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word, po[0].lower()) 
                      if (po[0].lower() in ['n', 'r', 'v', 'a'] and word[0] != '@') else word for word, po in pos])

    
    return text
      


# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App"""

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")

	from PIL import Image

	image = Image.open('resources/imgs/bird.jpg')
	st.image(image, caption='What does your tweet say about climate change?', use_column_width=True)
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Information","Visuals","Prediction"]
	selection = st.sidebar.radio("Navigation", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("About")
		# You can read a markdown file from supporting resources folder
		st.markdown("""This App allows the user to input text(ideally a tweet relating to climate change), and will
                       classify it according to whether or not they believe in climate change.
                       You can have a look at the Exploratory Data Analysis on **Visuals Analysis** page,
					   and make your predictions on the **Predictions** page by navigating on the sidebar.""")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show Table'): # data is hidden if box is unchecked
			st.table(raw.head()) # raw data into table format

	if selection == "Visuals":
		st.info('The below graphs offers a visual analysis of data')

		# output variable distribution
		train_eda = raw.copy()
		train_eda['sent_labels']  = train_eda['sentiment'].map({-1: 'Anti',0:'Neutral', 1:'Pro', 2:'News'})
		fig, axes = plt.subplots(ncols=2, 
                         nrows=1, 
                         figsize=(10,5), 
                         dpi=100)

		sns.countplot(train_eda['sent_labels'], ax=axes[0])

		code_labels=['Pro', 'News', 'Neutral', 'Anti']
		axes[1].pie(train_eda['sent_labels'].value_counts(),
            labels= code_labels,
            autopct='%1.0f%%',
            startangle=90,
            explode= (0.04, 0, 0, 0))

		fig.suptitle('Target variable distribution', fontsize=20)
		st.pyplot(fig)

		st.markdown("""The above chart shows that our raw corpus has data imbalance amongst classes and the **Pro** 
		                class hugely dominates other classes""")

	   # wordcloud
	    
		st.subheader("Most Common Words per Class")
		wordcimg = Image.open('resources/imgs/wordclouds.png')
		st.image(wordcimg, caption='Most used words per sentiment')

		st.subheader("Pro Class most popular tags")
		tagsimg = Image.open('resources/imgs/positivetags.png')
		st.image(tagsimg, caption='Pro class most popular tags',use_column_width=True)

		st.subheader("Anti Class most popular tags")
		tagsmg = Image.open('resources/imgs/negativetags.png')
		st.image(tagsmg, caption='Anti class most popular tags',use_column_width=True)

		st.subheader("News Class most popular tags")
		tagsimgs = Image.open('resources/imgs/newstags.png')
		st.image(tagsimgs, caption='News class most popular tags',use_column_width=True)
	   

	# Building out the predication page

	# sentiment definations
	sent = {-1: {'Anti': 'The tweet does not believe in climate change'},
             0: {'Neutral': 'The tweet neither support nor no refutes the belief of man-made climate change'},
             1: {'Pro': 'The tweet supports the belief of man-made climate change'},
             2: {'News': 'The tweet links to factual news about climate change'}}

	if selection == "Prediction":
		model = st.selectbox('Select Model to use for Prediction',["SVC","LR","MNB","KNN"])

		st.markdown("""This dropdown offers multiple classification models. You are required to choose a model
		            	to use, the recommended model ``SVC`` is selected by default.
						 Also, the models are ranked on best classification performance.""")

		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text/Tweet to Classify","Type Here")
		
		if model == 'LR':

			if st.button("Classify"):
				# applying clean_tweet function to clean the user input
				tweet_text = clean_tweet(tweet_text)
				# user string input to iterable
				tweet_text = [tweet_text]
				# Load your .pkl file with the model of your choice + make predictions
				predictor = joblib.load(open(os.path.join("resources/LR_model.pkl"),"rb"))
				prediction = predictor.predict(tweet_text)
				# display prediction results
				st.success("Text Categorized as: {}".format(sent[prediction[0]]))
		
		elif model == 'SVC':

			if st.button("Classify"):
				# applying clean_tweet function to clean the user input
				tweet_text = clean_tweet(tweet_text)
				# user string input to iterable
				tweet_text = [tweet_text]
				# loading a .pkl file and predict with loaded model
				predictor = joblib.load(open(os.path.join("resources/SVC_model.pkl"),"rb"))
				prediction = predictor.predict(tweet_text)
				# display prediction results
				st.success("Text Categorized as: {}".format(sent[prediction[0]]))
		
		elif model == 'MNB':

			if st.button("Classify"):
				# applying clean_tweet function to clean the user input
				tweet_text = clean_tweet(tweet_text)
				# user string input to iterable/list
				tweet_text = [tweet_text]
				# loading a .pkl file + predict with loaded model
				predictor = joblib.load(open(os.path.join("resources/MNB_model.pkl"),"rb"))
				prediction = predictor.predict(tweet_text)
				# display prediction results
				st.success("Text Categorized as: {}".format(sent[prediction[0]]))

		elif model == 'KNN':

			if st.button("Classify"):
				# applying clean_tweet function to clean the user input
				tweet_text = clean_tweet(tweet_text)
				# user input string to iterable/list
				tweet_text = [tweet_text]
				# loading a .pkl file + predict with loaded model
				predictor = joblib.load(open(os.path.join("resources/KNN_mode.pkl"),"rb"))
				prediction = predictor.predict(tweet_text)
				# display prediction results
				st.success("Text Categorized as: {}".format(sent[prediction[0]]))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
