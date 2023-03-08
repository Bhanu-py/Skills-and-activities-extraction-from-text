import pandas as pd
import numpy as np
from util import get_embeds, mean_pooling

skills_df = pd.read_csv('clean/skillgroups_df.csv').fillna('')
skills_df = skills_df[skills_df['Level 0 preferred term'] == 'skills']
sk = skills_df.loc[:,~skills_df.columns.str.contains('Uri',case=False, regex=True)]

sk['Level 1 code'] = np.where(sk['Level 1 code'] == '',sk['Level 0 code'],sk['Level 1 code'])
sk['Level 2 code'] = np.where(sk['Level 2 code'] == '',sk['Level 1 code'],sk['Level 2 code'])
sk['Level 3 code'] = np.where(sk['Level 3 code'] == '',sk['Level 2 code'],sk['Level 3 code'])

# skills = sk['Level 3 preferred term'].tolist()
skills = sk['Description'].tolist()
print(len(skills))

skill_embeds = get_embeds(skills)
st.cache(skill_embeds)
