from signver.cleaner import Cleaner
from signver.matcher import Matcher
from signver.utils import data_utils
from signver.utils.data_utils import resnet_preprocess, invert_img
from signver.utils.visualization_utils import plot_np_array, plot_prediction_score, visualize_boxes, get_image_crops, make_square

import os
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
import pydaisi as pyd

svt_detector_model = pyd.Daisi("soul0101/SVT Detector Model")
svt_extractor_model = pyd.Daisi("soul0101/SVT Extractor Model")

# try:
#     if svt_detector_model.workers.number == 0:
#         svt_detector_model.workers.set(1)
# except:
#     pass
# try:
#     if svt_extractor_model.workers.number == 0:
#         svt_extractor_model.workers.set(1)
# except:
#     pass

DIR = os.path.dirname(__file__)
CLEANER_MODEL_PATH = "models/cleaner/small"

@st.cache(allow_output_mutation=True)
def load_cleaner_model():
    """
    Loads a trained model for cleaning signatures
    Returns:
        cleaner: trained model for cleaning signatures
    """
    cleaner = Cleaner() 
    cleaner.load(os.path.join(DIR, CLEANER_MODEL_PATH))
    return cleaner

def clean(image_np):
    """
    Clean the given image. 

    Returns a numpy array of shape (height, width, depth) with a range of 0 to 255.

    Parameters
    ----------
    image_np: numpy array
        shape (height, width, depth) with a range of 0 to 255.
    Returns
    -------
    numpy array
        shape (height, width, depth) with a range of 0 to 255.
    """
    cleaner_model = load_cleaner_model()
    print("Cleaner model load time", cleaner_model.model_load_time)
    return cleaner_model.clean(image_np)

def match_verify(feat_1, feat_2):
    """
    Compares the two given feature vectors, by cosine distance.
    
    Parameters
    ----------
    feat_1: A 1-D array.
    feat_2: A 1-D array.
    
    Returns
    -------
    float:
        A float value between 0 and 2, with 0 meaning the two vectors are identical, 
        and 2 meaning they are perfectly opposite.
    """
    matcher = Matcher()
    return matcher.verify(feat_1, feat_2)

def match_cosine_distance(feat_1, feat_2):
    matcher = Matcher()
    return matcher.cosine_distance(feat_1, feat_2)

def pil_to_np(image_pil):
    """
    Converts an image from PIL format to NumPy format.

    Parameters
    ----------
    image_pil : PIL.image
        Image in PIL format.
        
    Returns
    -------
    image_np : numpy.array
        Array of image in NumPy format.
    """
    return np.array(image_pil)

def invert_image(image_np):
    """
    Inverts a given image, represented as a numpy array.

    Parameters
    ----------
        image_np: np.ndarray
            The image to be inverted.

    Returns
    -------
        np.ndarray
            A numpy array representing the inverted image.
    """
    return data_utils.invert_img(image_np)

def sanitize_np(image_np):
    """
    This function is used to convert the given image_np numpy.array to a 3 channel numpy.array.

    Parameters
    ----------
    image_np : numpy.ndarray
        A numpy array containing an image.

    Returns
    -------
    image_np : numpy.ndarray
        A 3 channel RGB numpy array containing an image.
        
    Raises
    ------
    TypeError
        If the image_np array is not a numpy array.
        
    TypeError
        If the image_np array is not a valid shape.
    """
    if not isinstance(image_np, (np.ndarray, np.generic)):
        raise TypeError("Input must be a numpy array")

    if(len(image_np.shape) == 2):
        # return cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        return np.stack((image_np,)*3, axis=-1)

    elif(len(image_np.shape) == 3):
        if(image_np.shape[2] == 3):
            return image_np
        elif(image_np.shape[2] > 3):
            return image_np[:, :, :3]
        else:
            return np.concatenate((image_np, np.stack((image_np[:, :, -1],) * (3 - image_np.shape[2]), axis=-1)), axis=2)
    else:
        raise TypeError("Invalid shape of image_np array")

def np_to_tensor(image_np):
    """
    Converts a numpy array to a tensor.
    
    Parameters
    ----------
    image_np : numpy.ndarray
        A numpy array representing an image.
    
    Returns
    -------
    img_tensor : tensor
        A tensor representing an image.
    """
    inverted_image_np = invert_img(image_np)
    img_tensor = tf.convert_to_tensor(inverted_image_np)
    img_tensor = img_tensor[tf.newaxis, ...]
    return img_tensor

def signature_preprocessor(sig_np):
    """
    This function preprocesses the signature image using the ResNet model.
    
    Parameters
    ----------
    sig_np : numpy.ndarray
        The signature image
        
    Returns
    -------
    numpy.ndarray
        The preprocessed signature image.
    """
    return resnet_preprocess(sig_np, resnet=False , invert_input=False)

def signature_detector(img_tensor):
    """
    This function takes in an image tensor and returns the bounding boxes, scores, classes, and detections of the image.

    Parameters
    ----------
    img_tensor : Tensor
        Image tensor of the form (1, img_height, img_width, 3)
        
    Returns
    -------
    boxes : Tensor
        A list of 4 element tuples of the form (y1, x1, y2, x2)
    scores : Tensor
        A list of confidence scores for each of the detected objects
    classes : Tensor
        A list of class labels for each detected object
    detections : Tensor
        The detections of the image.
    """
    boxes, scores, classes, detections = svt_detector_model.detect(img_tensor).value
    return boxes, scores, classes, detections

def signature_cleaner(signatures):
    """
    This function takes in a list of signatures, or a single signature and
    returns the cleaned signature images (removal of background lines and text).

    Parameters
    ----------
    signatures: np.ndarray or list<np.ndarray>
        If a list is received, it is first sanitized and normalized, and output is
        returned for each signature. If a single signature is received, there is no
        sanitization step, and the signature is normalized before output.
    
    Returns
    -------
    list<np.ndarray>
        The output is a list of cleaned signature images.
    """
    if isinstance(signatures, list):
        preprocessed_sigs = np.array([signature_preprocessor(sanitize_np(sign)) * (1./255) for sign in signatures])
    else:
        preprocessed_sigs = np.expand_dims(signature_preprocessor(sanitize_np(signatures)), axis=0)

    return clean(preprocessed_sigs)

def verify_signatures(sig1_np, sig2_np):    
    """
    Verify two signatures for match
    
    Parameters
    ----------
    sig1_np : numpy.ndarray
        The first signature.
    sig2_np : numpy.ndarray
        The second signature.
    
    Returns
    -------
    dict 
        A dictionary with two keys:
        - cosine_distance: the cosine distance between the two signatures
        - is_match: a boolean indicating whether the two signatures are a match or not
    """
    sig1_clean = signature_cleaner(sig1_np)
    sig2_clean = signature_cleaner(sig2_np)

    sig1_feats = svt_extractor_model.extract(sig1_clean).value
    sig2_feats = svt_extractor_model.extract(sig2_clean).value

    return {
        "cosine_distance": match_cosine_distance(sig1_feats[0, :], sig2_feats[0, :]),
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

        with st.expander("Cleaned Signatures", expanded=True):
            col1, col2 = st.columns(2)
            col1.image(cleaned_orig_sign, "Original Signature")
            col2.image(cleaned_check_sign, "Sign to be verified")

        with st.spinner("Extracting Features...\n Might be slow for the first time"):
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
            boxes, scores, classes, detections = signature_detector(img_tensor)

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
    else:
        st_ui_sign_verification()

if __name__ == "__main__":
    st_ui()
