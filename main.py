import streamlit as st
import spacy
from annotated_text import annotated_text
from spacy.cli import download
import pandas as pd

model_name = "en_core_web_sm"
if 'mainDf' not in st.session_state:
    st.session_state.breakdownDf = pd.DataFrame(columns=['query', 'token', 'pos', 'dep'])
    st.session_state.mainDf = pd.DataFrame(columns=['query', 'is_valid', 'spacy_response'])


@st.cache_resource
def load_nlp():
    try:
        print(f"\n\nLoading spacy model - {model_name}....\n\n")
        return spacy.load(model_name)
    except OSError:
        download(model_name)
        return spacy.load(model_name)


settings = {
    "lenient": {
        "root": ("VERB", "NOUN", "ADJ", "PRON", "PROPN"),
        # "root_required": True,
        # "sub_required": True,
        # "obj_required": False
    },
    "balanced": {
        "root": ("VERB", "NOUN", "ADJ", "PRON", "PROPN"),
        # "root_required": True,
        # "sub_required": True,
        # "obj_required": False
    },
    "strict": {
        "root": ("VERB",),
        # "root_required": True,
        # "sub_required": True,
        # "obj_required": True
    }
}


def check_grammar_lenient(doc):
    ann_list = []
    has_root = False
    has_subject = False
    has_object = False
    root_condition = settings["lenient"]['root']

    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ in root_condition:
            has_root = True
            ann_list.append((token.text + " ", token.pos_, "#8ef"))

        elif token.pos_ in ("NOUN", "PROPN", "ADJ") or 'subj' in token.dep_:
            has_subject = True
            ann_list.append((token.text + " ", token.dep_, "#faa"))

        elif 'obj' in token.dep_:
            has_object = True
            ann_list.append((token.text + " ", token.dep_, "#afa"))

        else:
            ann_list.append(token.text + " ")

    return ann_list, has_subject, has_root, has_object


def check_grammar_balanced(doc):
    ann_list = []
    has_root = False
    has_subject = False
    has_object = False
    root_condition = settings["balanced"]['root']

    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ in root_condition:
            has_root = True
            ann_list.append((token.text + " ", token.pos_, "#8ef"))

        elif 'subj' in token.dep_:
            has_subject = True
            ann_list.append((token.text + " ", token.dep_, "#faa"))

        elif 'obj' in token.dep_:
            has_object = True
            ann_list.append((token.text + " ", token.dep_, "#afa"))

        else:
            ann_list.append(token.text + " ")

    return ann_list, has_subject, has_root, has_object


def check_grammar_strict(doc):
    ann_list = []
    has_root = False
    has_subject = False
    has_object = False
    root_condition = settings["strict"]['root']

    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ in root_condition:
            has_root = True
            ann_list.append((token.text + " ", token.pos_, "#8ef"))

        elif 'subj' in token.dep_:
            has_subject = True
            ann_list.append((token.text + " ", token.dep_, "#faa"))

        elif 'obj' in token.dep_:
            has_object = True
            ann_list.append((token.text + " ", token.dep_, "#afa"))

        else:
            ann_list.append(token.text + " ")

    return ann_list, has_subject, has_root, has_object


# modified code
def check_grammar_and_syntax(doc, curr_sett="lenient"):
    if curr_sett == "lenient":
        ann_list, has_subject, has_root, has_object = check_grammar_lenient(doc)
    elif curr_sett == "balanced":
        ann_list, has_subject, has_root, has_object = check_grammar_balanced(doc)
    else:
        ann_list, has_subject, has_root, has_object = check_grammar_strict(doc)

    print(ann_list)
    return ann_list, has_subject, has_root, has_object


# def check_grammar_and_syntax(doc, curr_sett="lenient"):
#     root_condition = settings[curr_sett]['root']
#     has_root = False
#     has_subject = False
#     has_object = False
#
#     ann_list = []
#     for token in doc:
#         print(token.text, token.dep_, token.pos_)
#         if token.dep_ == "ROOT" and token.pos_ in root_condition:
#             has_root = True
#             ann_list.append((token.text + " ", token.pos_, "#8ef"))
#
#         elif token.dep_ in ("nsubj", "nsubjpass"):
#             has_subject = True
#             ann_list.append((token.text + " ", token.dep_, "#faa"))
#
#         elif token.dep_ in ("dobj", "pobj"):
#             has_object = True
#             ann_list.append((token.text + " ", token.dep_, "#afa"))
#
#         else:
#             ann_list.append(token.text + " ")
#
#     print(ann_list)
#     return ann_list, has_subject, has_root, has_object


def checkSentenceValidity(current_settings, sub, verb, obj):
    if current_settings == "strict":
        if sub and verb and obj:
            return True
        else:
            return False
    elif current_settings == "lenient":
        if (sub or obj) and verb:
            return True
        else:
            return False
    else:
        if sub and verb:
            return True
        else:
            return False


nlp = load_nlp()
st.title("Sentence Syntax Checker")

tab1, tab2 = st.tabs(["Data Builder", "Data Visualiser"])


def add_to_dataframes(document, text, expected_label, spacy_label):
    curr_len = len(st.session_state.mainDf.index)
    new_data = [text, expected_label, spacy_label]

    st.session_state.mainDf.loc[curr_len] = new_data
    st.toast("Row added successfully to Main Dataframe !!", icon='🎉')

    curr_len = len(st.session_state.breakdownDf.index)
    new_data = [[text, token.text, token.pos_, token.dep_] for token in document]

    for val in new_data:
        st.session_state.breakdownDf.loc[curr_len] = val
        curr_len += 1
    st.toast("Row added successfully to Breakdown Dataframe !!", icon='🎉')


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode("utf-8")


def calculate_overlap(df, col1, col2):
    overlap = df[((df[col1] == True) & (df[col2] == True)) | ((df[col1] == False) & (df[col2] == False))].shape[0]
    total = df.shape[0]
    overlap_percentage = (overlap / total) * 100
    return overlap_percentage


with tab1:
    current_settings = st.select_slider(
        "Select the level of strictness",
        options=["lenient", "balanced", "strict"], help="Pick between 3 modes -> Lenient, Balanced, Strict."
                                                        " Determines to include sub-verb-obj")

    st.write("\n\n")
    with st.form("Add query examples...", clear_on_submit=True, border=False):
        col1, col2 = st.columns([2.5, 1])
        with col1:
            text = st.text_input("Query", placeholder="Enter the rewritten query")
        with col2:
            st.write("Select if question is valid")
            opt = st.checkbox("Is valid", label_visibility='hidden')

        submitted = st.form_submit_button("Add query")
        if submitted:
            doc = nlp(text)
            ann_text, sub, verb, obj = check_grammar_and_syntax(doc, current_settings)
            annotated_text(ann_text)
            st.write("\n")
            validity = checkSentenceValidity(current_settings, sub, verb, obj)
            add_to_dataframes(doc, text, opt, validity)

with tab2:
    st.subheader("Main Dataframe")

    st.dataframe(st.session_state.mainDf, use_container_width=True)
    st.download_button(
        label="Download MainDataframe as CSV",
        data=convert_df(st.session_state.mainDf),
        file_name="large_df.csv",
        mime="text/csv",
    )

    overlap_percent = calculate_overlap(st.session_state.mainDf, 'is_valid', 'spacy_response')
    st.write(f"Overlap percent is: {overlap_percent}%")

    st.subheader("Breakdown Dataframe")
    st.dataframe(st.session_state.breakdownDf, use_container_width=True)
    st.download_button(
        label="Download Breakdown Dataframe as CSV",
        data=convert_df(st.session_state.breakdownDf),
        file_name="large_df.csv",
        mime="text/csv",
    )
