
from signver import cleaner
from signver.detector import Detector
from signver.cleaner import Cleaner
from signver.extractor import MetricExtractor
from signver.matcher import Matcher
from signver.utils import data_utils
from signver.utils.data_utils import resnet_preprocess, invert_img
from signver.utils.visualization_utils import plot_np_array, plot_prediction_score, visualize_boxes, get_image_crops, make_square

import sys, os
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

DIR = os.path.dirname(__file__)
DETECTOR_MODEL_PATH = "models/detector/small"
EXTRACTOR_MODEL_PATH = "models/extractor/metric"
CLEANER_MODEL_PATH = "models/cleaner/small"

@st.cache(allow_output_mutation=True)
def load_detector_model():
    detector = Detector()
    detector.load(os.path.join(DIR, DETECTOR_MODEL_PATH))
    return detector

@st.cache(allow_output_mutation=True)
def load_cleaner_model():
    cleaner = Cleaner() 
    cleaner.load(os.path.join(DIR, CLEANER_MODEL_PATH))
    return cleaner

@st.cache(allow_output_mutation=True)
def load_extractor_model():
    extractor = MetricExtractor() 
    extractor.load(os.path.join(DIR, EXTRACTOR_MODEL_PATH))
    return extractor

def detect(input_tensor):
    detection_model = load_detector_model()
    return detection_model.detect(input_tensor)

def clean(image_np):
    cleaner_model = load_cleaner_model()
    return cleaner_model.clean(image_np)

def extract(image_np):
    extractor_model = load_extractor_model()
    return extractor_model.extract(image_np)

def match_verify(feat_1, feat_2):
    matcher = Matcher()
    return matcher.verify(feat_1, feat_2)

def match_cosine_distance(feat_1, feat_2):
    matcher = Matcher()
    return matcher.cosine_distance(feat_1, feat_2)

def pil_to_np(image_pil):
    return np.array(image_pil)

def invert_image(image_np):
    return data_utils.invert_img(image_np)

def np_to_tensor(image_np):
    inverted_image_np = invert_img(image_np)
    img_tensor = tf.convert_to_tensor(inverted_image_np)
    img_tensor = img_tensor[tf.newaxis, ...]
    return img_tensor

def signature_preprocessor(sig_np):
    return resnet_preprocess(sig_np, resnet=False , invert_input=False)

def signature_cleaner(signatures):
    if isinstance(signatures, list):
        preprocessed_sigs = np.array([signature_preprocessor(sign) * (1./255) for sign in signatures])
    else:
        preprocessed_sigs = np.expand_dims(signature_preprocessor(signatures), axis=0)

    return clean(preprocessed_sigs)

def signature_feature_extractor(signatures):
    if isinstance(signatures, list):
        sign_features = extract(np.array(signatures) / 255)
    else:
        sign_features = extract(signatures)
    
    return sign_features

def verify_signatures(sig1_np, sig2_np):    
    sig1_clean = signature_cleaner(sig1_np)
    sig2_clean = signature_cleaner(sig2_np)

    sig1_feats = extract(sig1_clean)
    sig2_feats = extract(sig2_clean)

    return {
        "similarity_index": match_cosine_distance(sig1_feats[0, :], sig2_feats[0, :]),
        "is_match": match_verify(sig1_feats[0, :], sig2_feats[0, :])
    }


def st_ui_sign_verification():
    st.header("Signature Verification")
    orig_sign = st.sidebar.file_uploader("Upload Original Signature")
    check_sign = st.sidebar.file_uploader("Upload Signature to be Checked")

    if orig_sign is not None:
        orig_sign_pil = Image.open(orig_sign).convert(mode='RGB')
    else:
        orig_sign_pil = Image.open(os.path.join(DIR, r'./data/test/extractor/forgeries_2_12.png')).convert(mode='RGB')
    orig_sign_np = pil_to_np(orig_sign_pil)

    if check_sign is not None:
        check_sign_pil = Image.open(check_sign).convert(mode='RGB')
    else:
        check_sign_pil = Image.open(os.path.join(DIR, r'./data/test/extractor/original_2_11.png')).convert(mode='RGB')
    check_sign_np = pil_to_np(check_sign_pil)
    
    col1, col2 = st.columns(2)
    col1.image(orig_sign_np, "Original Signature")
    col2.image(check_sign_np, "Sign to be verified")

    if st.button("Verify"):
        cleaned_orig_sign = signature_cleaner(orig_sign_np)
        cleaned_check_sign = signature_cleaner(check_sign_np)

        with st.expander("Cleaned Signatures"):
            col1, col2 = st.columns(2)
            col1.image(cleaned_orig_sign, "Original Signature")
            col2.image(cleaned_check_sign, "Sign to be verified")

        
        st.json(verify_signatures(orig_sign_np, check_sign_np))

def st_ui_sign_extraction():
    img_buf = st.sidebar.file_uploader("Upload_file")
    if img_buf is not None:
        img_pil = Image.open(img_buf).convert(mode='RGB')
    else:
        img_pil = Image.open(os.path.join(DIR, r'./data/test/localizer/signdoc.jpg')).convert(mode='RGB')

    image_np = pil_to_np(img_pil)
    inverted_image_np = invert_image(image_np)
    img_tensor = np_to_tensor(image_np)

    st.header("Document vs Inverted Document")
    st.image(np.concatenate((image_np, inverted_image_np ), axis = 1))

    detect_button = st.button("Detect Signature")
    if detect_button:
        # get a list of bounding box predictions for image
        with st.spinner("Getting detections...\n Might be slow for the first time"):
            boxes, scores, classes, detections = detect(img_tensor)

        st.header("Signature Detection & Extraction")
        # plot confidence scores for each detections
        threshold = 0.22
        fig_box_prediction = plot_prediction_score(scores, threshold)
        with st.expander("Prediction Scores"):
            st.pyplot(fig_box_prediction)
            
        # annotate image with bounding boxes above a given threshold and plot 
        annotated_image = visualize_boxes(image_np, boxes, scores, threshold=threshold, color="green")
        with st.expander("Annoted Document", expanded=True):
            st.image(annotated_image)

        # st.header("Extracted Signatures")
        signatures = get_image_crops(image_np, boxes, scores,  threshold = 0.22)
        fig = plot_np_array(signatures, fig_size=(12,14), ncols=4, title="Extracted Signatures")
        with st.expander("Signature Extraction", expanded=True):
            st.pyplot(fig)

        sigs= [ resnet_preprocess( x, resnet=False, invert_input=False ) for x in signatures ]
        fig = plot_np_array(sigs,"Preprocessed Signatures") 
        with st.expander("Signature Preprocessing", expanded=True):
            st.pyplot(fig)

        norm_sigs = [ x * (1./255) for x in sigs]
        plot_np_array(norm_sigs, "Preprocessed Signatures")

        cleaned_sigs = signature_cleaner(signatures)

        if len(cleaned_sigs) > 0:
            with st.expander("Original vs Cleaned Signatures"):
                for i in range(len(cleaned_sigs)):
                    fig_comparison = plot_np_array(np.concatenate((norm_sigs[i], cleaned_sigs[i]) , axis=1))
                    st.pyplot(fig_comparison)

def st_ui():
    st.sidebar.header('Signature Extraction & Verification Toolkit')
    activities = ["Signature Extraction", "Signature Verification"]
    choose = st.sidebar.selectbox("Select Application", activities)
    if choose == "Signature Extraction":
        st_ui_sign_extraction()
    elif choose == "Signature Verification":
        st_ui_sign_verification()

if __name__ == "__main__":
    st_ui()
