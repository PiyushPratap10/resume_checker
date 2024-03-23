import streamlit as st
import pickle
import re
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# Loading models
clf = pickle.load(open('clf.pkl','rb'))
tfidf=pickle.load(open('tfidf.pkl','rb'))

def cleanResume(txt):
    cleanTxt=re.sub('http\S+\s'," ",txt)
    cleanTxt=re.sub('RT|cc'," ",cleanTxt)
    cleanTxt=re.sub('#\S+\s'," ",cleanTxt)
    cleanTxt = re.sub('@\S+', " ", cleanTxt)
    cleanTxt=re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")," ",cleanTxt)
    cleanTxt=re.sub(r'[^\x00-\x7f]'," ",cleanTxt)
    cleanTxt=re.sub('\s+'," ",cleanTxt)
    return cleanTxt
# Web Application
def main():
    st.set_page_config(
        page_title="Resume Checker",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown(
        """
        <style>
        body {
            background-color: #f0f2f6;
            font-family: Arial, sans-serif;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("Resume Checker - Resume Screening App")
    upload_file=st.file_uploader("Submit Your Resume",type=['pdf','txt','docx'])

    if upload_file is not None:
        try:
            resume_bytes = upload_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume=cleanResume(resume_text)
        input_features = tfidf.transform([cleaned_resume])
        prediction_id=clf.predict(input_features)[0]

        # Map category ID to category name
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        category_name = category_mapping.get(prediction_id, "Unknown")
        st.write("Predicted Category:",category_name)

    st.subheader("About Resume Checker")
    st.write("The Resume Checker is a powerful tool designed to streamline the screening process for resumes. Built using Python programming language and leveraging Natural Language Processing (NLP) techniques, this application enables recruiters and hiring managers to efficiently analyze and categorize resumes based on their content. By simply uploading a resume file, users can instantly receive predictions regarding the most relevant job category for the candidate. The application employs machine learning models trained on a diverse set of resumes to ensure accurate and reliable results. With its intuitive interface and advanced capabilities, the Resume Checker empowers organizations to make informed decisions and expedite the candidate selection process.")

    st.subheader("Packages Used")
    st.write("**Streamlit:** Streamlit is an open-source Python library that makes it easy to create web apps for machine learning, data analysis, and more.")

    st.write("**Pickle:** Pickle is used for serializing and deserializing Python object structures.")

    st.write("**Re:** The re module provides support for regular expressions in Python.")

    st.write("**NLTK:** NLTK (Natural Language Toolkit) is a leading platform for building Python programs to work with human language data.")

# Python Main
if __name__=="__main__":
    main()