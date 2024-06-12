import streamlit as st
import spacy
from annotated_text import annotated_text
from spacy.cli import download

model_name = "en_core_web_sm"


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
        "root": ("VERB", "AUX"),
        # "root_required": True,
        # "sub_required": True,
        # "obj_required": False
    },
    "balanced": {
        "root": ("VERB",),
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


def check_grammar_and_syntax(doc, curr_sett="lenient"):
    root_condition = settings[curr_sett]['root']
    has_root = False
    has_subject = False
    has_object = False

    ann_list = []
    print(curr_sett)
    for token in doc:
        # Check for root verb
        print(f"token.dep")
        if token.dep_ == "ROOT" and token.pos_ in root_condition:
            has_root = True
            ann_list.append((token.text + " ", token.pos_, "#8ef"))

        # Check for subject (nsubj or equivalent)
        elif token.dep_ in ("nsubj", "nsubjpass"):
            has_subject = True
            ann_list.append((token.text + " ", token.dep_, "#faa"))

        # Check for object (dobj or equivalent)
        elif token.dep_ == "dobj":
            has_object = True
            ann_list.append((token.text + " ", token.dep_, "#afa"))

        else:
            ann_list.append(token.text + " ")

    print(ann_list)
    return ann_list, has_subject, has_root, has_object


nlp = load_nlp()
st.title("Sentence Syntax Checker")
current_settings = st.select_slider(
    "Select the level of strictness",
    options=["lenient", "balanced", "strict"], help="Pick between 3 modes -> Lenient, Balanced, Strict."
                                                    " Determines to include sub-verb-obj")
text = st.text_input("Query", placeholder="Enter the rewritten query")

if text:
    doc = nlp(text)
    ann_text, sub, verb, obj = check_grammar_and_syntax(doc, current_settings)
    annotated_text(ann_text)
    st.write("\n")
    print(sub, verb, obj)
    if current_settings == "strict":
        if sub and verb and obj:
            st.write("The Entered Text seems syntactically correct")
        else:
            st.write("Please use the original query")
    else:
        if sub and verb:
            st.write("The Entered Text seems syntactically correct")
        else:
            st.write("Please use the original query")
