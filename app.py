import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

#
def main():
    st.title("Spam Prediction")
    selected_box = st.sidebar.selectbox('Select your choice', ('Spam Prediction', 'About the App'))
    if selected_box == 'About the App':
        about()
    elif selected_box == 'Spam Prediction':


        input_sms = st.text_area("Enter the message")
        def transform_text(text):
            text = text.lower()
            text = nltk.word_tokenize(text)

            y = []
            for i in text:
                if i.isalnum():
                    y.append(i)

            text = y[:]
            y.clear()

            for i in text:
                if i not in stopwords.words('english') and i not in string.punctuation:
                    y.append(i)

            text = y[:]
            y.clear()

            for i in text:
                y.append(ps.stem(i))

            return " ".join(y)

        if st.button('Predict'):

            # 1. preprocess
            transformed_sms = transform_text(input_sms)
            # 2. vectorize
            vector_input = tfidf.transform([transformed_sms])
            # 3. predict
            result = model.predict(vector_input)[0]
            # 4. Display
            if result == 1:
                st.header("Spam")
            else:
                st.header("Not Spam")


def about():
    st.title("Welcome!")
    st.caption("Spam Prediction Web App")
    with st.expander("Abstract"):
        st.write("""By using this app:\n1.Increased Productivity by reducing distractions\n
2.Enhanced Security Reducing the risk of Malware\n
3.Cleaning Inbox: By minimizing spam, the app ensures that users' inboxes are cleaner and more               organized.\n
4.Easy to Use: The app is designed with          user-friendliness in mind, ensuring that even those with limited technical skills can use it effectively.\n
""")
    #with st.expander("Block Diagram"):
        #st.image(r'E:\7SemesterLab\7thLab\DIPLab\underwater-image-enhancement-main\SpamPrediction\images\sample2.png',
                 #use_column_width=True)

    with st.expander("Results On Sample text"):
        st.image(r'E:\7SemesterLab\7thLab\DIPLab\underwater-image-enhancement-main\SpamPrediction\images\sampleSpam1.png',
                 use_column_width=True)
        st.image(
            r'E:\7SemesterLab\7thLab\DIPLab\underwater-image-enhancement-main\SpamPrediction\images\sample2.png',
            use_column_width=True)

    with st.expander("Developer: "):
        st.write("""Farhana Akter Suci
                    \nID:B190305001""")




#def transform_text(text):
    #text = text.lower()
    #text = nltk.word_tokenize(text)

    #y = []
    #for i in text:
        #if i.isalnum():
            #y.append(i)

    #text = y[:]
    #y.clear()

    #for i in text:
        #if i not in stopwords.words('english') and i not in string.punctuation:
            #y.append(i)

    #text = y[:]
    #y.clear()

    #for i in text:
        #y.append(ps.stem(i))

    #return " ".join(y)'''




tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

#st.title("Spam Prediction")

#input_sms = st.text_area("Enter the message")


if __name__ == "__main__":
    main()