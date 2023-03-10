import streamlit as st
import pandas as pd
from util import get_embeds, mean_pooling, get_cos_sim, read_file
import os
import numpy as np
import seaborn as sns


# model = "jjzha/jobbert-base-cased"
# thresh = 0.70

# # model = "jjzha/jobspanbert-base-cased"
# # thresh = 0.55
# top_n = 10

st.set_page_config(layout="wide")
st.title("Skills and Activities Extraction")

st.markdown("### 🎲 Matching Application")
st.markdown("""The Job description in text format is read sentence by sentence and converted into vector (Job embeding).
               Then the job embeding is matched with the skill embedings(vectors) from the database to give the best matched skills for each sentence. """)
menu = ["Select examole job text file from the below list", "Upload From Computer"]

model = st.sidebar.selectbox("Select the model", ('JobBert', 'JobspanBert'))

if model == "JobBert":
   model = "jjzha/jobbert-base-cased"
   thresh = 0.550
else:
   model = "jjzha/jobspanbert-base-cased"
   thresh = 0.55

choice = st.sidebar.radio(label="Menu", options=["Select .txt file from the below list", "choose your own .txt file"])

if choice == "Select .txt file from the below list":
    file = st.sidebar.selectbox("Upload your .txt file", os.listdir("clean/example"))
    uploaded_file = os.path.join(os.getcwd(), "clean/example", file)
else:
    uploaded_file = st.sidebar.file_uploader("Please upload a .txt file:", type=['txt'])


skills_df = pd.read_csv('clean/skillgroups_df.csv').fillna('')
skills_df = skills_df[skills_df['Level 0 preferred term'] == 'skills']
sk = skills_df.loc[:,~skills_df.columns.str.contains('Uri',case=False, regex=True)]

sk['Level 1 code'] = np.where(sk['Level 1 code'] == '',sk['Level 0 code'],sk['Level 1 code'])
sk['Level 2 code'] = np.where(sk['Level 2 code'] == '',sk['Level 1 code'],sk['Level 2 code'])
sk['Level 3 code'] = np.where(sk['Level 3 code'] == '',sk['Level 2 code'],sk['Level 3 code'])

# skills = sk['Level 3 preferred term'].tolist()
skills = sk['Description'].tolist()
# print(len(skills))


skill_embeds = get_embeds(skills, model=model)
st.cache(skill_embeds)

job = read_file(uploaded_file)

job_embeds = get_embeds(job, model=model)

sent_sim = dict()
skill_tree = dict()
score = dict()
for i, sent in enumerate(job_embeds):
    sim_sent = []
    for j, skill in enumerate(skill_embeds):
        sim = get_cos_sim(sent,skill)
        sim_sent.append(sim)
        # print(sim)
    max_sim = np.where(np.array([0 if x <= thresh else 1 for x in sim_sent])==1)[0]
    # print(max_sim)
    sent_sim[i] = [skills[i] for i in max_sim]
    skill_tree[i] =  [sk['Level 3 code'].tolist()[k] for k in max_sim]
    score[i] = [sim_sent[k] for k in max_sim]

codes = [i for j in list(skill_tree.values()) for i in j]
scores = [i for j in list(score.values()) for i in j]

res = dict()
for i in range(len(skill_tree)):
  res[i] = dict(zip(skill_tree[i], score[i]))

sorted_match = dict()
for key, value in res.items():
  sorted_match[key] = dict(sorted(value.items(), key=lambda x:x[1], reverse=True))

sk_indexed = sk.set_index(['Level 3 code']).sort_index(ascending=True)

mapped = []
match_score = []
for key, value in sorted_match.items():
  code_ls = [pd.concat([sk_indexed.loc[[i for i in list(value.keys())]][['Level 1 preferred term', 
                                                                         'Level 2 preferred term',
                                                                         'Level 3 preferred term']]]).head(5)]
  scores = list(value.values())
  if len(scores) != 0:
    match_score.append(scores)
  else:
    match_score.append("")
  code_ls[0]['Match_Score'] = scores[:5]
  # print(match_score)
  mapped.append(code_ls[0])
  # print(code_ls)

result = pd.concat(mapped)

# Styling the outpud DF 
# Set colormap equal to seaborns light green color palette
cm = sns.light_palette("green", as_cmap=True)
result = result.reset_index(drop=False).style.background_gradient(cmap=cm, subset=['Match_Score']).format({'Match_Score': "{:.3%}"})



# print(pd.concat(mapped))
#   print(code_ls)


if uploaded_file is not None:
    # pred_dic = predict(uploaded_file)
    st.write("**Job Description:**", job)
    st.markdown('# Skills Required')
    # st.dataframe(result)
    for sent, _ in enumerate(job):
      # print(mapped[sent])
      if len(mapped[sent]) != 0:
        st.write(f'<p style="font-size:26px; color:red;">{job[sent]}</p>',
                 mapped[sent].reset_index(drop=False).style.background_gradient(cmap=cm, subset=['Match_Score']).format({'Match_Score': "{:.3%}"}),
                 unsafe_allow_html=True)
      else:
         st.write(f'<p style="font-size:26px; color:red;">{job[sent]}</p>',
                   " \n No skills Detected \n",
                   unsafe_allow_html=True)
  
