"""
AgriScan AI - Crop Disease Detection & Diagnosis Platform
Clean, modern dashboard with Grad-CAM Explainable AI, live pathology alerts & agronomic precautions.
"""

import os
import sys
import json
from io import BytesIO
from textwrap import dedent
from PIL import Image
import streamlit as st
import pandas as pd

# Add project root to sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from models.predictor import CropDiseasePredictor
from src.report_generator import generate_pdf_report


# Page configuration
st.set_page_config(
    page_title="AgriScan AI - Crop Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
css_path = os.path.join(ROOT_DIR, "web", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_html(html_str: str, target=st):
    """Safely renders HTML without markdown indentation interpretation."""
    target.markdown(dedent(html_str).strip(), unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    return CropDiseasePredictor()


@st.cache_data
def load_disease_info():
    info_path = os.path.join(ROOT_DIR, "data", "disease_info.json")
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_metrics():
    metrics_path = os.path.join(ROOT_DIR, "models", "saved_models", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


ORDERED_CROPS = [
    {"id": 1, "name": "Apple", "icon": "🍎", "conditions": "Scab, Black Rot, Cedar Rust, Healthy", "count": 4},
    {"id": 2, "name": "Corn (Maize)", "icon": "🌽", "conditions": "Common Rust, Northern Blight, Healthy", "count": 3},
    {"id": 3, "name": "Potato", "icon": "🥔", "conditions": "Early Blight, Late Blight, Healthy", "count": 3},
    {"id": 4, "name": "Rice", "icon": "🌾", "conditions": "Brown Spot, Leaf Blast, Healthy", "count": 3},
    {"id": 5, "name": "Tomato", "icon": "🍅", "conditions": "Bacterial Spot, Early Blight, Late Blight, TYLCV, Healthy", "count": 5}
]


def render_sidebar():
    st.sidebar.title("🌿 AgriScan AI")
    st.sidebar.markdown("**Deep Learning Crop Pathology & Treatment Platform**")
    
    app_mode = st.sidebar.radio(
        "Navigation",
        ["🔍 Disease Diagnosis", "📊 Model Architecture & Evaluation", "📖 Crop Pathology Encyclopedia", "🚀 REST API & Deployment"],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Crops Available")
    st.sidebar.caption("Supported agricultural cash crops and pathology profiles:")

    # Clean rendering without any Markdown indentation parsing
    items_html = "".join([
        f'<div class="crop-list-item"><span>{c["id"]}. {c["icon"]} <b>{c["name"]}</b></span><span class="crop-tag">{c["count"]} conditions</span></div>'
        for c in ORDERED_CROPS
    ])
    render_html(f'<div class="sidebar-crop-card">{items_html}</div>', target=st.sidebar)

    st.sidebar.markdown("---")
    status_html = """
    <div style="font-size: 0.8rem; color: #64748b; line-height: 1.5;">
        <div><b>Model:</b> Custom 4-Block CNN</div>
        <div><b>Inference:</b> &lt; 35ms (CPU)</div>
        <div><b>Status:</b> <span style="color: #16a34a; font-weight: 700;">🟢 Online & Ready</span></div>
    </div>
    """
    render_html(status_html, target=st.sidebar)

    return app_mode


def main():
    app_mode = render_sidebar()
    predictor = load_predictor()
    disease_info = load_disease_info()
    metrics = load_metrics()

    # Clean, uncluttered header
    header_html = """
    <div class="main-header-clean">
        <h1>🌾 Crop Disease Detection Using Convolutional Neural Networks</h1>
        <p>Automated visual crop pathology diagnosis, Grad-CAM foliar lesion attention mapping, and instant agronomic treatment advisories.</p>
        <div class="header-feature-pills">
            <span class="header-pill">⚡ &lt; 35ms Inference Latency</span>
            <span class="header-pill">🔬 Grad-CAM Saliency Maps</span>
            <span class="header-pill">🌱 18 Pathology & Healthy Classes</span>
            <span class="header-pill">📄 PDF Prescription Export</span>
        </div>
    </div>
    """
    render_html(header_html)

    # -------------------------------------------------------------
    # TAB 1: DISEASE DIAGNOSIS
    # -------------------------------------------------------------
    if app_mode == "🔍 Disease Diagnosis":
        col_input, col_results = st.columns([1, 1], gap="large")

        input_image = None
        selected_sample_name = None

        with col_input:
            st.subheader("1. Upload or Select Leaf Image")
            
            input_source = st.radio(
                "Choose Input Source:",
                ["📁 Upload File", "📷 Webcam / Camera", "🖼️ Curated Sample"],
                horizontal=True
            )

            if input_source == "📁 Upload File":
                uploaded_file = st.file_uploader(
                    "Drop a clear leaf image here (JPG, PNG, JPEG, WEBP):",
                    type=["jpg", "jpeg", "png", "webp"],
                    help="Upload clear photos of top or bottom leaf surface."
                )
                if uploaded_file is not None:
                    input_image = Image.open(uploaded_file)
                    selected_sample_name = uploaded_file.name

            elif input_source == "📷 Webcam / Camera":
                camera_file = st.camera_input("Take a photo of the crop leaf")
                if camera_file is not None:
                    input_image = Image.open(camera_file)
                    selected_sample_name = "Camera Capture Specimen"

            elif input_source == "🖼️ Curated Sample":
                samples_dir = os.path.join(ROOT_DIR, "data", "sample_test_images")
                if os.path.exists(samples_dir):
                    sample_files = sorted([f for f in os.listdir(samples_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
                    if sample_files:
                        sample_labels = {f: f.replace("___", " - ").replace("_sample.jpg", "").replace("_", " ") for f in sample_files}
                        chosen_file = st.selectbox(
                            "Select sample leaf for instant test:",
                            sample_files,
                            format_func=lambda x: f"🌿 {sample_labels.get(x, x)}"
                        )
                        if chosen_file:
                            sample_path = os.path.join(samples_dir, chosen_file)
                            input_image = Image.open(sample_path)
                            selected_sample_name = sample_labels.get(chosen_file, chosen_file)

            if input_image is not None:
                st.markdown("#### Input Leaf Preview")
                st.image(input_image, caption=selected_sample_name or "Uploaded Leaf Sample", use_container_width=True)

        with col_results:
            st.subheader("2. AI Diagnosis & Pathology Analysis")

            if input_image is not None:
                with st.spinner("Analyzing foliar morphology and activating CNN feature maps..."):
                    result = predictor.predict(input_image, top_k=4, generate_heatmap=True)

                best = result["primary_prediction"]
                details = result["details"]
                severity = result["severity"]

                # Clean Status Banner
                if best["is_healthy"]:
                    badge_class = "badge-healthy"
                    status_text = f"✅ HEALTHY CROP: {best['crop'].upper()} (No Pathological Lesions)"
                elif severity in ["Critical", "Critical / Devastating", "High"]:
                    badge_class = "badge-alert"
                    status_text = f"⚠️ HIGH ALERT: {best['crop'].upper()} - {best['disease'].upper()}"
                else:
                    badge_class = "badge-warning"
                    status_text = f"⚠️ PATHOLOGY DETECTED: {best['crop'].upper()} - {best['disease'].upper()}"

                badge_html = f"""
                <div style="margin-bottom: 14px;">
                    <span class="{badge_class}">
                        {status_text}
                    </span>
                </div>
                """
                render_html(badge_html)

                # Core Metric Cards
                m1, m2, m3 = st.columns(3)
                m1.metric("Target Crop", best["crop"])
                m2.metric("Diagnosis Confidence", f"{best['confidence']}%")
                m3.metric("Severity Level", severity)

                # Explainable AI: Grad-CAM
                if "cam_overlay" in result:
                    st.markdown("---")
                    st.markdown("#### 🔬 Explainable AI: Grad-CAM Attention Heatmap")
                    st.caption("Visual heatmap generated from the final convolutional layer highlighting pathological regions influencing the diagnosis.")
                    
                    cam_c1, cam_c2 = st.columns(2)
                    cam_c1.image(input_image, caption="Original Input", use_container_width=True)
                    cam_c2.image(result["cam_overlay"], caption="Grad-CAM Saliency Map", use_container_width=True)

                # Top Candidate Probability Distribution
                st.markdown("---")
                st.markdown("#### Classification Confidence Distribution")
                top_df = pd.DataFrame([
                    {"Class Candidate": f"{p['crop']} - {p['disease']}", "Confidence (%)": p["confidence"]}
                    for p in result["top_predictions"]
                ])
                st.bar_chart(top_df.set_index("Class Candidate"), height=170, color="#16a34a")

                # Actionable Treatment Advisory Card
                st.markdown("---")
                st.markdown("### 📋 Agronomic Treatment & Advisory Card")

                with st.expander("🔎 Biological Pathogen & Environmental Triggers", expanded=True):
                    st.markdown(f"**Identified Pathogen:** `{details.get('pathogen', 'N/A')}`")
                    st.markdown(f"**Environmental Trigger Conditions:** {details.get('cause', 'N/A')}")
                    st.markdown(f"**Observed Symptoms:** {details.get('symptoms', 'N/A')}")

                with st.expander("🌱 Organic & Bio-Control Measures", expanded=True):
                    for org in details.get("organic_treatment", []):
                        st.markdown(f"- 🌿 {org}")

                if not best["is_healthy"]:
                    with st.expander("🧪 Chemical & Fungicidal Interventions", expanded=True):
                        for chem in details.get("chemical_treatment", []):
                            st.markdown(f"- 💊 {chem}")

                with st.expander("🛡️ Long-Term Preventive Cultural Practices", expanded=False):
                    for prev in details.get("prevention", []):
                        st.markdown(f"- 🛡️ {prev}")

                # Download PDF Report Button
                st.markdown("---")
                try:
                    pdf_bytes = generate_pdf_report(result)
                    st.download_button(
                        label="📄 Download Official Agronomic PDF Report",
                        data=pdf_bytes,
                        file_name=f"AgriScan_Report_{best['crop']}_{best['disease'].replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as e:
                    st.info(f"Report generator ready: {e}")

            else:
                st.info("👈 Please upload an image or choose a curated sample on the left panel to begin diagnosis.")

        # -------------------------------------------------------------
        # BELOW THE MAIN DASHBOARD: NEWS, ALERTS & PRECAUTIONS
        # -------------------------------------------------------------
        st.markdown("<hr style='margin-top: 35px; margin-bottom: 25px; border: 0; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)
        
        col_news, col_precautions = st.columns([1, 1], gap="large")

        with col_news:
            render_html("""
            <div class="info-section-title">
                <span>📰</span> Current Disease Alerts & Seasonal Crop News
            </div>
            """)

            news_col1, news_col2 = st.columns(2)
            with news_col1:
                render_html("""
                <div class="news-card">
                    <span class="news-tag tag-red">Active Alert</span>
                    <h4>Potato Late Blight Humidity Warning</h4>
                    <p>Prolonged fog and cool temperatures (12-18°C) are elevating <i>Phytophthora infestans</i> spore pressure. Maintain preventative copper spray intervals.</p>
                </div>
                """)

                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

                render_html("""
                <div class="news-card">
                    <span class="news-tag tag-amber">Seasonal Advisory</span>
                    <h4>Tomato Whitefly & TYLCV Migration</h4>
                    <p>Dry and warm spells trigger <i>Bemisia tabaci</i> whitefly flights transmitting Yellow Leaf Curl Virus. Deploy yellow sticky traps every 10 meters.</p>
                </div>
                """)

            with news_col2:
                render_html("""
                <div class="news-card">
                    <span class="news-tag tag-blue">Field Monitoring</span>
                    <h4>Rice Leaf Blast Nitrogen Caution</h4>
                    <p>Heavy vegetative nitrogen fertilization combined with overcast humidity elevates Blast lesion formation. Split nitrogen top-dressing into 3 doses.</p>
                </div>
                """)

                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

                render_html("""
                <div class="news-card">
                    <span class="news-tag tag-green">Orchard Health</span>
                    <h4>Apple Scab Post-Rain Protocol</h4>
                    <p>Ascospore release peaks after consecutive rain showers. Apply systemic fungicides within 24-48 hours of primary infection events.</p>
                </div>
                """)

        with col_precautions:
            render_html("""
            <div class="info-section-title">
                <span>🛡️</span> Smart Farming Precautions & Prevention Tips
            </div>
            """)

            render_html("""
            <div class="precaution-card">
                <div class="precaution-icon">💧</div>
                <div class="precaution-text">
                    <h5>Drip Irrigation vs Overhead Sprinklers</h5>
                    <p>Overhead sprinklers create prolonged leaf wetness and soil splash, which is the primary vector for Early Blight and Bacterial Spot. Drip systems keep foliar canopy dry.</p>
                </div>
            </div>

            <div class="precaution-card">
                <div class="precaution-icon">🔄</div>
                <div class="precaution-text">
                    <h5>3-Year Crop Rotation Protocol</h5>
                    <p>Never plant solanaceous crops (Potato, Tomato, Pepper) consecutively in the same plot. Rotate with legumes, maize, or brassicas to disrupt soil-borne pathogens.</p>
                </div>
            </div>

            <div class="precaution-card">
                <div class="precaution-icon">✂️</div>
                <div class="precaution-text">
                    <h5>Lower Canopy Pruning & Trellising</h5>
                    <p>Prune the lowest 15-20 cm of tomato foliage once plants establish. Improving bottom airflow reduces humidity pockets and prevents ground spore splash.</p>
                </div>
            </div>

            <div class="precaution-card">
                <div class="precaution-icon">🌱</div>
                <div class="precaution-text">
                    <h5>Bio-Fungicide Pre-Treatment</h5>
                    <p>Incorporate <i>Trichoderma harzianum</i> and <i>Bacillus subtilis</i> during transplanting to colonize root zones and activate systemic plant defenses.</p>
                </div>
            </div>
            """)

    # -------------------------------------------------------------
    # TAB 2: MODEL ARCHITECTURE & EVALUATION
    # -------------------------------------------------------------
    elif app_mode == "📊 Model Architecture & Evaluation":
        st.subheader("Deep Learning CNN Model Architecture & Evaluation")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pathology Classes", "18")
        c2.metric("Input Resolution", "224 x 224")
        c3.metric("Core Topology", "Custom 4-Block CNN")
        c4.metric("Inference Latency", "< 35 ms")

        st.markdown("---")
        st.markdown("### CNN Topology Diagram")
        st.code("""
Input Leaf Image (3 x 224 x 224)
   │
   ▼
[ConvBlock 1]: Conv2d(3->32, 3x3) -> BatchNorm -> ReLU -> Conv2d(32->32) -> MaxPool(2x2) -> Dropout(0.15)
   │
   ▼
[ConvBlock 2]: Conv2d(32->64, 3x3) -> BatchNorm -> ReLU -> Conv2d(64->64) -> MaxPool(2x2) -> Dropout(0.20)
   │
   ▼
[ConvBlock 3]: Conv2d(64->128, 3x3) -> BatchNorm -> ReLU -> Conv2d(128->128) -> MaxPool(2x2) -> Dropout(0.25)
   │
   ▼
[ConvBlock 4]: Conv2d(128->256, 3x3) -> BatchNorm -> ReLU -> Conv2d(256->256) -> MaxPool(2x2) -> Dropout(0.30)
   │
   ▼
[Global Average Pooling (GAP)]: AdaptiveAvgPool2d(1, 1) -> (Batch, 256)
   │
   ▼
[Dense Classifier Head]: Linear(256 -> 128) -> BatchNorm1d -> ReLU -> Dropout(0.35) -> Linear(128 -> 18)
   │
   ▼
Softmax Probability Distribution (18 Classes)
        """, language="text")

        st.markdown("---")
        st.markdown("### Model Evaluation & Training Visualizations")
        
        col_c1, col_c2 = st.columns(2)
        saved_dir = os.path.join(ROOT_DIR, "models", "saved_models")
        curves_file = os.path.join(saved_dir, "training_curves.png")
        cm_file = os.path.join(saved_dir, "confusion_matrix.png")

        with col_c1:
            st.markdown("#### Loss & Accuracy Convergence Curves")
            if os.path.exists(curves_file):
                st.image(curves_file, use_container_width=True)

        with col_c2:
            st.markdown("#### 18-Class Multi-Category Confusion Matrix")
            if os.path.exists(cm_file):
                st.image(cm_file, use_container_width=True)

        if metrics:
            st.markdown("#### Summary Training Metadata")
            st.json(metrics)

    # -------------------------------------------------------------
    # TAB 3: CROP PATHOLOGY ENCYCLOPEDIA
    # -------------------------------------------------------------
    elif app_mode == "📖 Crop Pathology Encyclopedia":
        st.subheader("Agricultural Crop Pathology Encyclopedia")
        st.markdown("Browse complete symptom guides, causal agents, and verified treatment protocols across all 5 staple crops in ordered hierarchy.")

        selected_crop = st.selectbox(
            "Select Crop:",
            ["All Crops"] + [c["name"] for c in ORDERED_CROPS],
            index=0
        )
        search_query = st.text_input("🔍 Search symptoms, treatments, or pathogens:", "")

        filtered_items = []
        for class_key, d in disease_info.items():
            crop = d.get("crop", "")
            disease = d.get("disease", "")
            pathogen = d.get("pathogen", "")
            symptoms = d.get("symptoms", "")

            if selected_crop != "All Crops" and crop.lower() != selected_crop.lower():
                continue

            query_lower = search_query.lower()
            if query_lower and (query_lower not in crop.lower() and query_lower not in disease.lower() and query_lower not in pathogen.lower() and query_lower not in symptoms.lower()):
                continue

            filtered_items.append((class_key, d))

        st.markdown(f"**Displaying {len(filtered_items)} pathology profiles:**")

        for class_key, d in filtered_items:
            with st.expander(f"🌿 {d.get('crop')} - {d.get('disease')} ({'Healthy' if d.get('is_healthy') else 'Pathogenic'})"):
                st.markdown(f"**Pathogen:** `{d.get('pathogen')}` | **Severity:** `{d.get('severity')}`")
                st.markdown(f"**Symptoms:** {d.get('symptoms')}")
                st.markdown(f"**Environmental Triggers:** {d.get('cause')}")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**🌱 Organic Remedies:**")
                    for org in d.get("organic_treatment", []):
                        st.markdown(f"- {org}")
                with col2:
                    st.markdown("**🧪 Chemical Interventions:**")
                    for chem in d.get("chemical_treatment", []):
                        st.markdown(f"- {chem}")

    # -------------------------------------------------------------
    # TAB 4: REST API & DEPLOYMENT
    # -------------------------------------------------------------
    elif app_mode == "🚀 REST API & Deployment":
        st.subheader("Production Integration & Cloud Deployment")

        st.markdown("""
        AgriScan AI is built with decoupled microservice architecture:
        - **Interactive Dashboard:** Streamlit (`run_app.bat` or port 8501)
        - **FastAPI REST Service:** High-throughput JSON endpoints (`run_api.bat` or port 8000)
        - **Containerized Stack:** Complete `Dockerfile` & `docker-compose.yml`
        """)

        st.markdown("---")
        st.markdown("### REST API Endpoint Examples")

        st.markdown("#### Python Client Example")
        st.code("""
import requests

url = "http://localhost:8000/predict"
with open("leaf_specimen.jpg", "rb") as f:
    response = requests.post(url, files={"file": ("leaf.jpg", f, "image/jpeg")}, data={"top_k": 3})

print(response.json()["primary_prediction"])
        """, language="python")

        st.markdown("#### cURL Request Example")
        st.code("""
curl -X POST "http://localhost:8000/predict" \\
     -H "accept: application/json" \\
     -F "file=@specimen.jpg"
        """, language="bash")


if __name__ == "__main__":
    main()
