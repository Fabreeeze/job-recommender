
import os
import nltk
os.environ["NLTK_DATA"]="D:/Anaconda3/lib/site-packages/nltk"

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def setup_nltk():
    # Download NLTK resources
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')


def GetJobRecommendation( skills ):
  file_path = 'modified_job_descriptions.csv'

  import pandas as pd
  setup_nltk()
  # the dataset
  print("reading dataset.....")
  data = pd.read_csv(file_path)
  if(len(data) > 0):
    print("file read")
  # filtered_data = data[data['Qualifications'].str.contains('M.Tech|B.Tech', case=False)]

  # data = filtered_data

  import string

 

  # Define stopwords and lemmatizer
  stop_words = set(stopwords.words('english'))
  lemmatizer = WordNetLemmatizer()

  # # Function to preprocess text
  # def preprocess_text(text):
  #     # Tokenize text
  #     tokens = word_tokenize(text)

  #     # Removing punctuation and stopwords, and lemmatize tokens
  #     tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token not in string.punctuation and token.lower() not in stop_words]

  #     # Join tokens back into a single string
  #     preprocessed_text = ' '.join(tokens)

  #     return preprocessed_text

  # data['Job Description'] = data['Job Description'].apply(preprocess_text)


  # num_rows, num_columns = data.shape


  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.metrics.pairwise import cosine_similarity

  # Initialize TF-IDF vectorizer
  tfidf_vectorizer = TfidfVectorizer(stop_words='english')

  skills_tfidf = tfidf_vectorizer.fit_transform(data['skills'])

  print("past skills vectorisation ")
  # Example user input skills
  user_skills = skills

  # Transform user input to TF-IDF vector
  user_skills_tfidf = tfidf_vectorizer.transform([' '.join(user_skills)])

  
  from sklearn.neighbors import NearestNeighbors

  # KNN model
  knn = NearestNeighbors(n_neighbors=5, metric='cosine')

  # Fit KNN model on TF-IDF vectorized skills
  knn.fit(skills_tfidf)

  # nearest neighbors for example 1 user input skills
  distances, indices = knn.kneighbors(user_skills_tfidf)

  print("Most Similar Job Listings:")

  SimilarJobs = [ ]
  for i, index in enumerate(indices):
    job = data.iloc[index]
    SimilarJobs.append({
            'job_id': job['Job Id'].tolist(),
            'title': job['Job Title'].tolist(),
            'role': job['Role'].tolist(),
            'description': job['Job Description'].tolist(),
            'skills': job['skills'].tolist()
        })

  return SimilarJobs
  
  
      # print(f"\nNeighbor {i+1}:")
      # print("Job ID:", job['Job Id'])
      # print("Job Title:", )
      # print("Role:", job['Role'])
      # print("Job Description:", job['Job Description'])
      # print(" job skills required", job['skills'])
      # Add more details as needed

  # using euclidean distances as metric

  # knn_eucl = NearestNeighbors(n_neighbors=5, metric='euclidean')

  # knn_eucl.fit(skills_tfidf)

  # distances_eucl, indices_eucl = knn_eucl.kneighbors(user_skills_tfidf)

  # print("Most Similar Job Listings:")

  # #example 3 user skilss
  # user_skills_3 = ['html','css','javascript','react','nodejs','mysql','mongodb']

  # user_skills_3_tfidf = tfidf_vectorizer.transform([' '.join(user_skills_3)])


  # distances_eucl, indices_eucl = knn_eucl.kneighbors(user_skills_3_tfidf)

  # print("Most Similar Job Listings:")
  # for i, index in enumerate(indices_eucl[0]):
  #     job = data.iloc[index]
  #     print(f"\nNeighbor {i+1}:")
  #     print("Job ID:", job['Job Id'])
  #     print("Job Title:", job['Job Title'])
  #     print("Role:", job['Role'])
  #     print("Job Description:", job['Job Description'])
  #     print(" job skills required", job['skills'])
  #     # Add more details as needed



  # """# Content Based filter, matrix factoration etc to be approached next,  also need to add in other aspects like work exp, location , salary range etc
  # #
  # """

  # # # Define the user's experience level
  # # user_experience_level = '5 to 10 Years'  # Example user's experience level

  # # # Filter job listings based on the user's experience level
  # # filtered_jobs = data[data['Experience_' + user_experience_level.replace(' ', '_')] == 1]

  # # # Display the filtered job listings
  # # print("Filtered Jobs based on User's Experience Level:")
  # # print(filtered_jobs)

  # # Define the user's work type preference
  # user_work_type_preference = 'Full-Time'  # Example user's work type preference

  # # Construct the column name for the user's work type preference
  # work_type_column_name = 'Work Type_' + user_work_type_preference

  # # Filter job listings based on the user's work type preference
  # work_type_filtered_jobs = data[data[work_type_column_name] == 1]

  # # Display the filtered job listings based on work type preference
  # print("Filtered Jobs based on User's Work Type Preference:")
  # print(work_type_filtered_jobs)
  # filtered_jobs = work_type_filtered_jobs

  # # Sort filtered jobs by salary in descending order
  # sorted_jobs = filtered_jobs.sort_values(by=['Max Salary'], ascending=False)

  # #example emplopyer pref.
  # emp_pref = "None"

  # # Consider male-female preferences
  # if emp_pref == 'Male':
  #   sorted_jobs = sorted_jobs[sorted_jobs['Preference_Male'] == 1]
  # elif emp_pref== 'Female':
  #   sorted_jobs = sorted_jobs[sorted_jobs['Preference_Female'] == 1]
  # else:
  #   sorted_jobs = sorted_jobs[sorted_jobs['Preference_Both'] == 1]

  # sorted_jobs_skills_tfidf = tfidf_vectorizer.fit_transform(sorted_jobs['skills'])

  # knn_eucl2 = NearestNeighbors(n_neighbors=5, metric='euclidean')

  # knn_eucl2.fit(sorted_jobs_skills_tfidf)

  # # #example 3 user skilss
  # # user_skills_3 = ['team']

  # # user_skills_3_tfidf = tfidf_vectorizer.transform([' '.join(user_skills_3)])


  # # distances_eucl, indices_eucl = knn_eucl2.kneighbors(user_skills_3_tfidf)

  # # print("Most Similar Job Listings:")
  # # for i, index in enumerate(indices_eucl[0]):
  # #     job = data.iloc[index]
  # #     print(f"\nNeighbor {i+1}:")
  # #     print("Job ID:", job['Job Id'])
  # #     print("Job Title:", job['Job Title'])
  # #     print("Role:", job['Role'])
  # #     print("Job Description:", job['Job Description'])
  # #     print(" job skills required", job['skills'])
  # #     # Add more details as needed

