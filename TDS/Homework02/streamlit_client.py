import pandas as pd
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

import weaviate
client = weaviate.Client("http://localhost:8080")

def query_phrases(near_words=None, avoid_words=None, min_word_count=0, max_words=None, certainty=0.22, limit=20):
    near_text_filter = {}
    if near_words:
        near_text_filter["concepts"] = near_words
        near_text_filter["certainty"] = certainty
    if avoid_words:
        near_text_filter["moveAwayFrom"] = {"concepts": avoid_words, "force": certainty}

    where_filter = {}
    operands = []
    if min_word_count > 0:
        operands.append({
            "path": ["word_count"],
            "operator": "GreaterThan",
            "valueInt": min_word_count
        })
    if max_words:
        operands.append({
            "path": ["word_count"],
            "operator": "LessThan",
            "valueInt": max_words
        })
    if operands:
        where_filter = {"operator": "And", "operands": operands}

    query = client.query.get("Phrase", ["text"]).with_near_text(near_text_filter)
    if operands:
        query = query.with_where(where_filter)
    query = query.with_limit(limit).with_additional(["certainty"]).do()
    
    phrases = query.get("data", {}).get("Get", {}).get("Phrase", [])
    df = pd.DataFrame([{
        "text": p["text"],
        "certainty": p["_additional"]["certainty"]
    } for p in phrases if "_additional" in p])
    return df

st.title("WordNet Synonym Explorer")

st.header("Query 1: similar to 'vocation', 'values', not similar to 'vices', 'rot', 'debauchery'")
df1 = query_phrases(
    near_words=["vocation", "values"],
    avoid_words=["vices", "rot", "debauchery"],
    max_words=3,
    certainty=0.22,
    limit=22
)
st.subheader("22 Results")
st.dataframe(df1)

st.header("Query 2: similar to 'vices', 'debauchery', not similar to 'values', 'joy'")
df2 = query_phrases(
    near_words=["vices", "debauchery"],
    avoid_words=["values", "joy"],
    max_words=2,
    certainty=0.33,
    limit=33
)
st.subheader("33 Results")
st.dataframe(df2)

st.header("Synonyms and Antonyms Interactive")

col1, col2 = st.columns(2)

with col1:
    attract_input = st.text_input("Attractive words (comma separated)", "vocation, joy")
    min_len, max_len = st.slider("Word length (min and max)", 1, 5, (1,3))

with col2:
    repel_input = st.text_input("Repelling words (comma separated)", "vices, rot")

attract_words = [w.strip() for w in attract_input.replace(";",",").split(",") if w.strip()]
repel_words = [w.strip() for w in repel_input.replace(";",",").split(",") if w.strip()]

synonyms_df = query_phrases(
    near_words=attract_words,
    avoid_words=repel_words,
    min_word_count=min_len,
    max_words=max_len,
    certainty=0.22,
    limit=30
)

antonyms_df = query_phrases(
    near_words=repel_words,
    avoid_words=attract_words,
    min_word_count=min_len,
    max_words=max_len,
    certainty=0.22,
    limit=30
)

with col1:
    st.subheader("Synonyms")
    st.dataframe(synonyms_df)

with col2:
    st.subheader("Antonyms")
    st.dataframe(antonyms_df)

st.write("Awesome!")
st.balloons()
