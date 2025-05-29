import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import convnext_base
from PIL import Image
import os
import time
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#GPU-enabled
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dir = "models"
model_options = [
    "convnext_base_all_data_best.pt",
    "efficientnet_b3_all_data.pt"
]

class_names = ["Proper", "Defect", "Double", "Defect_Minor", "Defect_Major", "Broken"]
defect_labels = {label for label in class_names if label != "Proper"}

#map true label from filename
def extract_true_label(file_name: str) -> str:
    stem = os.path.splitext(os.path.basename(file_name))[0].lower()
    for label in class_names:
        if label.lower() in stem:
            return label
    return "Unknown"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

#load selected model from file
@st.cache_resource(show_spinner=False)
def load_model(model_path):
    loaded = torch.load(model_path, map_location=device) 

    if isinstance(loaded, dict) and all(k.startswith("_orig_mod.") for k in loaded):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in loaded.items()}
    elif isinstance(loaded, dict):
        state_dict = loaded
    else:
        try:
            state_dict = loaded.state_dict()
        except Exception as e:
            raise RuntimeError(f"Unknown model format: {e}")

    classifier_weight = [v for k, v in state_dict.items() if "classifier.2.weight" in k]
    num_classes = classifier_weight[0].shape[0] if classifier_weight else len(class_names)

    model = convnext_base(weights=None)
    model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, num_classes)
    model.load_state_dict(state_dict)
    model.to(device) 
    model.eval()
    return model, num_classes

#predict image class and return probabilities
@torch.inference_mode()
def predict(image, model):
    image_tensor = transform(image).unsqueeze(0).to(device)
    output = model(image_tensor)
    probs = F.softmax(output, dim=1).squeeze()
    pred_idx = int(torch.argmax(probs))
    return pred_idx, probs

#initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

st.title("💊 Pill Type Classifier")
st.write("Upload one or more pill images, select a model, and view predictions.")
st.caption(f"Inference device: `{device}`")  

col1, col2 = st.columns(2)
with col1:
    uploaded_files = st.file_uploader(
        "Upload image(s)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="uploader",
    )
with col2:
    selected_model = st.selectbox("Choose a model", model_options, key="model_select")

if uploaded_files and selected_model:
    st.write(f"{len(uploaded_files)} file(s) selected.")

    if st.button("Predict" if len(uploaded_files) == 1 else f"Predict {len(uploaded_files)} images", use_container_width=True):
        model_path = os.path.join(model_dir, selected_model)
        model, _ = load_model(model_path)

        progress_bar = st.progress(0.0)
        results_container = st.container()

        total_imgs = len(uploaded_files)
        defect_preds = 0

        for idx, file in enumerate(uploaded_files, start=1):
            img = Image.open(file).convert("RGB")

            start_t = time.perf_counter()
            pred_idx, probs_tensor = predict(img, model)
            run_time = time.perf_counter() - start_t

            probs = probs_tensor.tolist()
            pred_label = class_names[pred_idx] if pred_idx < len(class_names) else f"Class {pred_idx}"
            confidence = probs[pred_idx] * 100

            true_label = extract_true_label(file.name)
            caption = f"{file.name}\n→ Pred: {pred_label} ({confidence:.1f}%) | True: {true_label}"

            with results_container:
                st.image(img, caption=caption, width=220)

            if pred_label in defect_labels:
                defect_preds += 1

            st.session_state.history.insert(0, {
                "File name": file.name,
                "True label": true_label,
                "Prediction": pred_label,
                "Probability": f"{confidence:.1f}%",
                "Processing time (s)": f"{run_time:.3f}",
            })

            progress_bar.progress(idx / total_imgs)

        progress_bar.empty()

        defect_pct = (defect_preds / total_imgs) * 100 if total_imgs else 0.0
        st.subheader("Batch summary")
        col_a, col_b = st.columns(2)
        col_a.metric("Total images", total_imgs)

        #alert about defects
        if defect_preds > 0:
            st.warning(f"⚠️ {defect_preds} out of {total_imgs} images were predicted as defective.")
        else:
            st.success("No defects predicted in this batch.")
        
        #binary label logic
        def is_defect(label):
            return label in defect_labels
        
        #extract binary ground truths and predictions
        true_labels_bin = [is_defect(row["True label"]) for row in st.session_state.history]
        pred_labels_bin = [is_defect(row["Prediction"]) for row in st.session_state.history]
        
        #сalculate performance metrics for defect detection
        recall = recall_score(true_labels_bin, pred_labels_bin, zero_division=0)
        precision = precision_score(true_labels_bin, pred_labels_bin, zero_division=0)
        f1 = f1_score(true_labels_bin, pred_labels_bin, zero_division=0)
        accuracy = accuracy_score(true_labels_bin, pred_labels_bin)

        #display metrics
        st.subheader("Binary classification metrics")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Accuracy", f"{accuracy:.2%}")
        col_m2.metric("Precision", f"{precision:.2%}")
        col_m3.metric("Recall", f"{recall:.2%}")
        col_m4.metric("F1-score", f"{f1:.2%}")


if st.session_state.history:
    st.subheader("Recent predictions")
    hist_df = pd.DataFrame(st.session_state.history)
 
    sort_col1, sort_col2 = st.columns(2)
    with sort_col1:
        sort_choice = st.selectbox("Sort by", ("Latest first", "Defects first", "Proper first"))
    with sort_col2:
        filter_defects_only = st.checkbox("Show only predicted defects")
 
    df_view = hist_df.copy()
 
    if filter_defects_only:
        df_view = df_view[df_view["Prediction"].isin(defect_labels)]
 
    if sort_choice == "Defects first":
        df_view["_is_defect"] = df_view["Prediction"].isin(defect_labels)
        df_view = df_view.sort_values(by=["_is_defect", "Probability"], ascending=[False, False])
        df_view = df_view.drop(columns="_is_defect")
    elif sort_choice == "Proper first":
        df_view["_is_proper"] = df_view["Prediction"] == "Proper"
        df_view = df_view.sort_values(by=["_is_proper", "Probability"], ascending=[False, False])
        df_view = df_view.drop(columns="_is_proper")
 
    #visual highlight for defects
    def highlight_defects(row):
        if row["Prediction"] in defect_labels:
            return ['background-color: #ffcccc'] * len(row)  #light red background for defect
        else:
            return [''] * len(row)
 
    styled_df = df_view.style.apply(highlight_defects, axis=1)
    st.dataframe(styled_df, use_container_width=True)
 
else:
    st.info("No predictions yet. Upload some images to begin.")
