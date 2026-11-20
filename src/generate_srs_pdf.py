"""
SRS Document Generator for Crop Disease Detection Using CNN
Generates a complete academic Software Requirements Specification (SRS) PDF matching
the official university submission format, with embedded screenshots and structured tables.
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image as RLImage, PageBreak, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and print 'Page X' with running header and footer.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Oblique", 8)
        self.setFillColor(colors.HexColor("#555555"))

        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawRightString(572, 760, "Crop Disease Detection Using Convolutional Neural Networks")
            self.setStrokeColor(colors.HexColor("#dddddd"))
            self.setLineWidth(0.5)
            self.line(40, 752, 572, 752)

        # Footer (all pages)
        self.setFont("Helvetica", 8.5)
        self.setFillColor(colors.HexColor("#333333"))
        footer_text = f"Department of Computer Science & Engineering  |  Page {self._pageNumber}"
        self.drawCentredString(306, 28, footer_text)

        self.restoreState()


def build_srs_pdf(
    output_pdf_path: str,
    students: list = None,
    branch: str = "CSE",
    semester_year: str = "5th Semester / 3rd Year",
    college_name: str = "ABES Engineering College, Ghaziabad",
    academic_year: str = "2024 - 2028",
    project_guide: str = "Mr. Natesan Rajan",
    github_url: str = "https://github.com/devuphub/hcl-crop-disease",
    live_app_url: str = "https://agriscan-ai.streamlit.app"
):
    if students is None:
        students = [
            {"name": "DEV PRAKASH UPADHYAY", "roll": "2400320100401"},
            {"name": "DEVANSH RASTOGI", "roll": "2400320100408"}
        ]

    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=35,
        bottomMargin=35
    )

    styles = getSampleStyleSheet()

    # Custom typography matching university standards
    title_cover = ParagraphStyle(
        'CoverTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        alignment=1,
        textColor=colors.HexColor("#1b365d"),
        spaceAfter=6
    )

    subtitle_cover = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9.5,
        leading=13.5,
        alignment=1,
        textColor=colors.HexColor("#4a5568"),
        spaceAfter=10
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1b365d"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#2b6cb0"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13.5,
        textColor=colors.HexColor("#222222"),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        leftIndent=15,
        firstLineIndent=-10,
        textColor=colors.HexColor("#222222"),
        spaceAfter=3
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.8,
        leading=10.5,
        textColor=colors.HexColor("#1a202c")
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.8,
        leading=10.5,
        textColor=colors.HexColor("#1a202c")
    )

    table_cell_header = ParagraphStyle(
        'TableCellHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=11,
        textColor=colors.white
    )

    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        alignment=1,
        textColor=colors.HexColor("#4a5568"),
        spaceBefore=4,
        spaceAfter=8
    )

    story = []

    # =========================================================================
    # PAGE 1: COVER PAGE
    # =========================================================================
    story.append(Spacer(1, 8))
    
    # Top Banner Box
    header_box = [
        [Paragraph("<font color='white'><b>SOFTWARE REQUIREMENTS SPECIFICATION</b><br/><b>B.Tech (CSE)</b> &nbsp;|&nbsp; <b>3rd Year Project</b></font>", ParagraphStyle('HBox', alignment=1, fontSize=11, leading=15))]
    ]
    t_header = Table(header_box, colWidths=[532])
    t_header.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#1b365d")),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))
    story.append(t_header)
    story.append(Spacer(1, 14))

    story.append(Paragraph("<font color='#4a5568' size=9><b>PROJECT</b></font>", ParagraphStyle('PLabel', alignment=1, spaceAfter=4)))
    story.append(Paragraph("Crop Disease Detection<br/>Using Convolutional Neural Networks (CNN)", title_cover))
    story.append(Paragraph("An Explainable AI (Grad-CAM) & Deep Learning Platform for Automated Crop Foliar Pathology & Agronomic Treatment Recommendation", subtitle_cover))

    story.append(Paragraph(f"<b>GitHub Repository:</b> <font color='#2b6cb0'>{github_url}</font>", ParagraphStyle('LinkStyle', alignment=1, fontSize=8, leading=11)))
    story.append(Paragraph(f"<b>Live Application (Streamlit):</b> <font color='#2b6cb0'>{live_app_url}</font>", ParagraphStyle('LinkStyle2', alignment=1, fontSize=8, leading=11, spaceAfter=12)))

    story.append(Paragraph("<i>Submitted in partial fulfillment of the requirements for the degree of</i>", ParagraphStyle('SubDesc', alignment=1, fontSize=8.5, leading=12, spaceAfter=2)))
    story.append(Paragraph("<b>Bachelor of Technology in Computer Science & Engineering</b>", ParagraphStyle('Degree', alignment=1, fontSize=10, leading=13, textColor=colors.HexColor("#1b365d"), spaceAfter=14)))

    # Submission Table with Both Students
    sub_data = [
        [Paragraph("<font color='white'><b>SUBMITTED BY</b></font>", table_cell_header), Paragraph("", table_cell_header)],
    ]
    
    for i, s in enumerate(students, 1):
        sub_data.append([Paragraph(f"<b>STUDENT {i} NAME</b>", table_cell), Paragraph(f"<b>{s['name']}</b>", table_cell)])
        sub_data.append([Paragraph(f"<b>ROLL NUMBER ({s['name'].split()[0]})</b>", table_cell), Paragraph(s['roll'], table_cell)])

    sub_data.extend([
        [Paragraph("<b>BRANCH</b>", table_cell), Paragraph(branch, table_cell)],
        [Paragraph("<b>SEMESTER / YEAR</b>", table_cell), Paragraph(semester_year, table_cell)],
        [Paragraph("<b>COLLEGE NAME</b>", table_cell), Paragraph(college_name, table_cell)],
        [Paragraph("<b>ACADEMIC YEAR</b>", table_cell), Paragraph(academic_year, table_cell)],
        [Paragraph("<b>PROJECT GUIDE</b>", table_cell), Paragraph(f"<b>{project_guide}</b>", table_cell)],
    ])

    t_sub = Table(sub_data, colWidths=[170, 362])
    t_sub.setStyle(TableStyle([
        ('SPAN', (0,0), (1,0)),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1b365d")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e0")),
        ('TOPPADDING', (0,0), (-1,-1), 2.8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.8),
        ('BACKGROUND', (0,1), (0,-1), colors.HexColor("#f7fafc")),
    ]))
    story.append(t_sub)
    story.append(Spacer(1, 14))

    story.append(Paragraph("<b>Department of Computer Science & Engineering</b>", ParagraphStyle('Dept', alignment=1, fontSize=10, leading=13, textColor=colors.HexColor("#1b365d"))))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: TABLE OF CONTENTS
    # =========================================================================
    story.append(Paragraph("Table of Contents", ParagraphStyle('TOCTitle', fontName='Helvetica-Bold', fontSize=15, leading=19, textColor=colors.HexColor("#1b365d"), spaceAfter=12)))
    
    toc_items = [
        "1. Abstract",
        "2. Problem Statement",
        "3. Objectives",
        "4. Proposed Solution",
        "5. Scope of the Project",
        "   5.1 In-Scope Functionalities",
        "   5.2 Out-of-Scope (Current Prototype)",
        "6. Functional Requirements",
        "7. Non-Functional Requirements",
        "8. Technology Stack",
        "   8.1 Live Deployment Resources",
        "9. System Architecture",
        "   9.1 Layered Architecture",
        "   9.2 Deployment Architecture",
        "   9.3 Main External Services",
        "10. System Design",
        "   10.1 Use Case Diagram & Actor Interaction",
        "   10.2 Data Flow Diagram (DFD Level 1)",
        "   10.3 Entity Relationship / Data Flow Schema",
        "   10.4 Project Directory Structure",
        "   10.5 Core Inference & Diagnosis Workflow",
        "11. Module Description",
        "12. Database / Knowledge Base Design",
        "   12.1 Agronomic Disease Knowledge Base",
        "   12.2 Model Evaluation & Checkpoint Structure",
        "13. Testing",
        "   13.1 Test Approach & Cases",
        "   13.2 Requirements Traceability Matrix (RTM)",
        "14. System Implementation & Working Verification (Screenshots)",
        "15. Limitations",
        "16. Future Scope",
        "17. References",
        "Appendix A: Definitions & Abbreviations",
        "Appendix B: Submission Notes"
    ]

    for item in toc_items:
        story.append(Paragraph(f"<b>{item}</b>" if not item.startswith("   ") else f"&nbsp;&nbsp;&nbsp;&nbsp;{item.strip()}", body_style))

    story.append(PageBreak())

    # =========================================================================
    # SECTION 1: ABSTRACT
    # =========================================================================
    story.append(Paragraph("1. Abstract", h1_style))
    story.append(Paragraph(
        "Crop diseases represent a catastrophic vulnerability to global agricultural productivity, causing estimated yield losses "
        "between 20% and 40% annually across major cash and food security crops. Traditional crop disease identification depends upon "
        "expert human visual scouting, which is labor-intensive, slow, geographically inaccessible in rural agrarian zones, and prone to "
        "misdiagnosis during early, asymptomatic phases of pathogen onset.",
        body_style
    ))
    story.append(Paragraph(
        "This project presents <b>AgriScan AI</b>, an end-to-end, deep learning-powered crop disease detection, explainability, and "
        "treatment recommendation platform utilizing <b>Convolutional Neural Networks (CNN)</b>. Built upon PyTorch, Streamlit, and FastAPI, "
        "the system diagnoses foliar plant diseases across five high-priority crops (<b>Apple, Corn/Maize, Potato, Rice, and Tomato</b>) spanning "
        "<b>18 distinct pathological and healthy classes</b>. Beyond traditional 'black-box' classification, the application integrates "
        "<b>Explainable AI (XAI) using Gradient-weighted Class Activation Mapping (Grad-CAM)</b> to generate visual saliency heatmaps highlighting "
        "the exact cellular lesion areas triggering classification decisions.",
        body_style
    ))
    story.append(Paragraph(
        "Upon diagnosis, the platform queries an integrated agronomic knowledge base to provide actionable prescriptions, detailing causal pathogens, "
        "organic bio-control agents (e.g., <i>Bacillus subtilis</i>, <i>Trichoderma</i>), chemical fungicides with exact dosages, and long-term "
        "cultural prevention guidelines. A built-in PDF generator allows field scouts and farmers to export downloadable, formatted Diagnostic Advisory Reports. "
        "The system has been containerized with Docker, published to GitHub, and deployed globally on Streamlit Community Cloud with sub-35 millisecond inference latency.",
        body_style
    ))

    # =========================================================================
    # SECTION 2: PROBLEM STATEMENT
    # =========================================================================
    story.append(Paragraph("2. Problem Statement", h1_style))
    story.append(Paragraph(
        "Smallholder farmers and agricultural extension officers face acute operational difficulties in identifying crop foliar diseases timely and accurately. "
        "Delayed or inaccurate diagnoses lead to either under-treatment resulting in crop destruction, or excessive, haphazard chemical pesticide spraying that destroys "
        "soil ecosystems and induces chemical pathogen resistance. Key problem dimensions include:",
        body_style
    ))
    story.append(Paragraph("• <b>Scarcity of Field Agronomists:</b> The ratio of certified plant pathologists to farming acreage is critically low in developing agricultural belts.", bullet_style))
    story.append(Paragraph("• <b>Symptom Ambiguity:</b> Many fungal, bacterial, and viral pathologies exhibit deceptively similar early foliar spotting (e.g., Early Blight vs. Bacterial Spot).", bullet_style))
    story.append(Paragraph("• <b>Lack of Explainability:</b> Conventional AI classifiers produce cold probability scores without justifying why a diagnosis was rendered, hindering farmer adoption.", bullet_style))
    story.append(Paragraph("• <b>Absence of Actionable Advice:</b> Raw class labels (e.g., 'Potato Late Blight') do not tell the farmer what organic or chemical steps to apply immediately.", bullet_style))
    story.append(Paragraph("• <b>Deployment Barriers:</b> Complex ML models often demand costly GPU cloud servers, making lightweight edge/browser accessibility mandatory.", bullet_style))

    # =========================================================================
    # SECTION 3: OBJECTIVES
    # =========================================================================
    story.append(Paragraph("3. Objectives", h1_style))
    story.append(Paragraph("• Architect, train, and validate a lightweight, 4-block deep <b>Convolutional Neural Network (CNN)</b> capable of multi-class foliar pathology classification across 18 distinct classes.", bullet_style))
    story.append(Paragraph("• Integrate <b>Grad-CAM Explainable AI</b> to extract gradient activation maps from the final convolutional layer, producing visual saliency heatmaps over input leaf specimens.", bullet_style))
    story.append(Paragraph("• Curate an exhaustive <b>Agronomic Knowledge Base</b> containing causal pathogens, environmental triggers, bio-organic solutions, chemical fungicide schedules, and prevention tips.", bullet_style))
    story.append(Paragraph("• Develop a responsive, modern <b>Streamlit Web Application</b> supporting file upload, live webcam photography, and curated test specimen selection.", bullet_style))
    story.append(Paragraph("• Implement automated <b>PDF Advisory Report Generation</b> allowing farmers to export official diagnostic certificates directly from the browser.", bullet_style))
    story.append(Paragraph("• Build an asynchronous, production-grade <b>FastAPI REST API</b> (`/predict`, `/health`, `/classes`) for cross-platform integration.", bullet_style))
    story.append(Paragraph("• Deploy the system live to <b>Streamlit Community Cloud</b> (`agriscan-ai.streamlit.app`) with sub-35ms CPU inference.", bullet_style))

    # =========================================================================
    # SECTION 4: PROPOSED SOLUTION
    # =========================================================================
    story.append(Paragraph("4. Proposed Solution", h1_style))
    story.append(Paragraph(
        "The proposed AgriScan AI solution bifurcates the pipeline into an Offline Training Phase and an Online Diagnostic & Prescription Phase:",
        body_style
    ))
    story.append(Paragraph(
        "1. <b>Data Pipeline & Preprocessing:</b> Leaf images are resized to 224x224 pixels, augmented through random rotations (±20°), horizontal/vertical flips, and color jittering, followed by ImageNet channel normalization.<br/>"
        "2. <b>CNN Model Architecture:</b> A 4-stage convolutional hierarchy (Conv2D -> BatchNorm -> ReLU -> MaxPool -> Dropout) extracts low-level edges to high-level lesion patterns, condensed via Global Average Pooling (GAP) into a dense classification head.<br/>"
        "3. <b>Dual Output Engine:</b> The forward pass yields class confidence probabilities via Softmax, while the backward pass computes Grad-CAM feature gradients to highlight disease spots visually.<br/>"
        "4. <b>Prescription & Report Delivery:</b> Predicted pathology is cross-referenced with the agronomic knowledge base to display chemical/organic remedies and generate a downloadable PDF certificate.",
        body_style
    ))

    # =========================================================================
    # SECTION 5: SCOPE OF THE PROJECT
    # =========================================================================
    story.append(Paragraph("5. Scope of the Project", h1_style))
    story.append(Paragraph("5.1 In-Scope Functionalities", h2_style))
    story.append(Paragraph("• Multi-class pathology identification across 5 crops (Apple, Corn, Potato, Rice, Tomato) and 18 conditions.", bullet_style))
    story.append(Paragraph("• High-resolution Grad-CAM visual attention overlays on foliar inputs.", bullet_style))
    story.append(Paragraph("• Three input modes: file drag-and-drop, live webcam capture, and curated test gallery.", bullet_style))
    story.append(Paragraph("• Comprehensive agronomic advisory (Pathogen, Causes, Organic Treatments, Chemical Sprays, Prevention).", bullet_style))
    story.append(Paragraph("• Downloadable PDF diagnostic advisory certificates generated in-memory.", bullet_style))
    story.append(Paragraph("• Model performance evaluation analytics (Accuracy curves, Loss curves, 18-class Confusion Matrix).", bullet_style))
    story.append(Paragraph("• Full REST API backend with OpenAPI/Swagger documentation.", bullet_style))

    story.append(Paragraph("5.2 Out-of-Scope (Current Prototype)", h2_style))
    story.append(Paragraph("• Automated drone autopilot telemetry routing for aerial field scanning.", bullet_style))
    story.append(Paragraph("• Multi-lingual speech synthesis for illiterate farming communities (planned for Phase 2).", bullet_style))
    story.append(Paragraph("• Soil chemical NPK sensor IoT telemetry fusion (planned for future hardware iteration).", bullet_style))

    story.append(PageBreak())

    # =========================================================================
    # SECTION 6: FUNCTIONAL REQUIREMENTS
    # =========================================================================
    story.append(Paragraph("6. Functional Requirements", h1_style))
    
    fr_data = [
        [Paragraph("<font color='white'><b>ID</b></font>", table_cell_header), Paragraph("<font color='white'><b>Requirement</b></font>", table_cell_header), Paragraph("<font color='white'><b>Technical Description</b></font>", table_cell_header)],
        [Paragraph("FR-01", table_cell_bold), Paragraph("Image Ingestion", table_cell), Paragraph("The system shall accept leaf images via Drag-and-Drop file upload, direct webcam capture, or curated specimen library.", table_cell)],
        [Paragraph("FR-02", table_cell_bold), Paragraph("Image Normalization", table_cell), Paragraph("The pipeline shall resize specimens to 224x224 and normalize tensor channels with ImageNet mean and standard deviation.", table_cell)],
        [Paragraph("FR-03", table_cell_bold), Paragraph("CNN Feature Extraction", table_cell), Paragraph("The custom 4-block CNN shall extract multi-scale convolutional spatial features and perform Global Average Pooling.", table_cell)],
        [Paragraph("FR-04", table_cell_bold), Paragraph("Pathology Classification", table_cell), Paragraph("The model shall compute Softmax probability scores across all 18 crop disease and healthy classes.", table_cell)],
        [Paragraph("FR-05", table_cell_bold), Paragraph("Grad-CAM Saliency", table_cell), Paragraph("The system shall calculate gradients of the predicted class score against the last Conv layer to generate an attention heatmap.", table_cell)],
        [Paragraph("FR-06", table_cell_bold), Paragraph("Agronomic Advisory", table_cell), Paragraph("The system shall query `disease_info.json` to fetch pathogen taxonomy, causes, organic treatments, and chemical dosages.", table_cell)],
        [Paragraph("FR-07", table_cell_bold), Paragraph("PDF Certificate Export", table_cell), Paragraph("The system shall dynamically compile and stream a downloadable ReportLab PDF diagnostic certificate.", table_cell)],
        [Paragraph("FR-08", table_cell_bold), Paragraph("REST API Endpoints", table_cell), Paragraph("FastAPI shall provide asynchronous `/predict`, `/health`, and `/classes` endpoints accepting multipart image payloads.", table_cell)],
        [Paragraph("FR-09", table_cell_bold), Paragraph("Analytics Dashboard", table_cell), Paragraph("The web UI shall render multi-class confusion matrices, training convergence curves, and confidence bar charts.", table_cell)],
        [Paragraph("FR-10", table_cell_bold), Paragraph("Disease Encyclopedia", table_cell), Paragraph("The platform shall supply a searchable index of all 18 foliar pathologies filterable by crop species.", table_cell)],
    ]
    t_fr = Table(fr_data, colWidths=[45, 125, 362])
    t_fr.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1b365d")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#ffffff")),
    ]))
    story.append(t_fr)

    # =========================================================================
    # SECTION 7: NON-FUNCTIONAL REQUIREMENTS
    # =========================================================================
    story.append(Paragraph("7. Non-Functional Requirements", h1_style))
    
    nfr_data = [
        [Paragraph("<font color='white'><b>ID</b></font>", table_cell_header), Paragraph("<font color='white'><b>Quality Attribute</b></font>", table_cell_header), Paragraph("<font color='white'><b>Requirement Specification</b></font>", table_cell_header)],
        [Paragraph("NFR-01", table_cell_bold), Paragraph("Inference Latency", table_cell), Paragraph("The CNN forward pass and Grad-CAM generation shall execute within 35 milliseconds on standard commodity CPU.", table_cell)],
        [Paragraph("NFR-02", table_cell_bold), Paragraph("Memory Footprint", table_cell), Paragraph("The serialized model checkpoint (`crop_disease_model.pth`) shall not exceed 5.0 MB for easy edge/mobile distribution.", table_cell)],
        [Paragraph("NFR-03", table_cell_bold), Paragraph("Reliability & Fallback", table_cell), Paragraph("If hook gradients fail to capture during Grad-CAM, the system shall safely return the class diagnosis without crashing.", table_cell)],
        [Paragraph("NFR-04", table_cell_bold), Paragraph("Cross-Platform Portability", table_cell), Paragraph("The codebase shall run seamlessly on Windows, Linux, Docker containers, and Streamlit Community Cloud.", table_cell)],
        [Paragraph("NFR-05", table_cell_bold), Paragraph("Usability & UX", table_cell), Paragraph("The UI shall provide a clean, high-contrast, responsive interface requiring zero machine learning knowledge from users.", table_cell)],
        [Paragraph("NFR-06", table_cell_bold), Paragraph("Modularity", table_cell), Paragraph("Model architecture, inference pipeline, report generator, web frontend, and REST API shall be strictly decoupled.", table_cell)],
        [Paragraph("NFR-07", table_cell_bold), Paragraph("Security", table_cell), Paragraph("File upload routes shall enforce strict MIME type validation and buffer limits to prevent malicious arbitrary payloads.", table_cell)],
    ]
    t_nfr = Table(nfr_data, colWidths=[48, 115, 369])
    t_nfr.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1b365d")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_nfr)

    # =========================================================================
    # SECTION 8: TECHNOLOGY STACK
    # =========================================================================
    story.append(Paragraph("8. Technology Stack", h1_style))
    
    tech_data = [
        [Paragraph("<font color='white'><b>Component Layer</b></font>", table_cell_header), Paragraph("<font color='white'><b>Technology</b></font>", table_cell_header), Paragraph("<font color='white'><b>Role in Project</b></font>", table_cell_header)],
        [Paragraph("Core Runtime", table_cell_bold), Paragraph("Python 3.11", table_cell), Paragraph("Primary programming language for deep learning and application logic.", table_cell)],
        [Paragraph("Deep Learning Framework", table_cell_bold), Paragraph("PyTorch 2.0+ & Torchvision", table_cell), Paragraph("CNN layer definitions, backprop, GPU/CPU acceleration, and tensor operations.", table_cell)],
        [Paragraph("Web Application", table_cell_bold), Paragraph("Streamlit 1.30+", table_cell), Paragraph("Interactive browser user interface with reactive state management and tabs.", table_cell)],
        [Paragraph("REST API", table_cell_bold), Paragraph("FastAPI & Uvicorn", table_cell), Paragraph("High-throughput asynchronous JSON inference endpoints with Swagger docs.", table_cell)],
        [Paragraph("Explainable AI (XAI)", table_cell_bold), Paragraph("Custom Grad-CAM Module", table_cell), Paragraph("Extracts backward activation gradients to produce attention heatmaps.", table_cell)],
        [Paragraph("Computer Vision & Math", table_cell_bold), Paragraph("PIL, NumPy, Matplotlib", table_cell), Paragraph("Image decoding, array transformations, and colormap overlay blending.", table_cell)],
        [Paragraph("Evaluation & Metrics", table_cell_bold), Paragraph("Scikit-Learn, Seaborn", table_cell), Paragraph("Confusion matrix calculation, precision, recall, and F1-score evaluation.", table_cell)],
        [Paragraph("Document Generation", table_cell_bold), Paragraph("ReportLab 4.0+", table_cell), Paragraph("Automated in-memory generation of professional PDF advisory certificates.", table_cell)],
        [Paragraph("Containerization", table_cell_bold), Paragraph("Docker & Docker Compose", table_cell), Paragraph("Multi-service container orchestration for one-command cloud deployment.", table_cell)],
    ]
    t_tech = Table(tech_data, colWidths=[110, 130, 292])
    t_tech.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1b365d")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_tech)

    story.append(Paragraph("8.1 Live Deployment Resources", h2_style))
    deploy_data = [
        [Paragraph("<font color='white'><b>Resource</b></font>", table_cell_header), Paragraph("<font color='white'><b>Platform / Identifier</b></font>", table_cell_header), Paragraph("<font color='white'><b>Access URL / Details</b></font>", table_cell_header)],
        [Paragraph("Live Application", table_cell_bold), Paragraph("Streamlit Community Cloud", table_cell), Paragraph(f"<font color='#2b6cb0'>{live_app_url}</font>", table_cell)],
        [Paragraph("Source Repository", table_cell_bold), Paragraph("GitHub (Public)", table_cell), Paragraph(f"<font color='#2b6cb0'>{github_url}</font>", table_cell)],
        [Paragraph("Model Weights Checkpoint", table_cell_bold), Paragraph("Local / Git LFS", table_cell), Paragraph("`models/saved_models/crop_disease_model.pth` (4.86 MB)", table_cell)],
        [Paragraph("Knowledge Base", table_cell_bold), Paragraph("JSON Database", table_cell), Paragraph("`data/disease_info.json` (18 detailed pathology records)", table_cell)],
    ]
    t_deploy = Table(deploy_data, colWidths=[120, 130, 282])
    t_deploy.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1b365d")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_deploy)

    story.append(PageBreak())

    # =========================================================================
    # SECTION 9: SYSTEM ARCHITECTURE
    # =========================================================================
    story.append(Paragraph("9. System Architecture", h1_style))
    story.append(Paragraph("9.1 Layered Architecture", h2_style))
    story.append(Paragraph(
        "AgriScan AI is engineered upon a cohesive 4-tier layered architecture ensuring strict separation of concerns, "
        "high testability, and edge-deployability:",
        body_style
    ))
    
    arch_box = [
        [Paragraph("<b>PRESENTATION LAYER:</b> Streamlit Web UI &nbsp;|&nbsp; FastAPI Swagger `/docs` &nbsp;|&nbsp; Camera & File Drag-and-Drop", ParagraphStyle('A1', fontSize=8.5, textColor=colors.HexColor("#1b365d"), alignment=1))],
        [Paragraph("↓", ParagraphStyle('Arrow', alignment=1, fontSize=10, textColor=colors.HexColor("#4a5568")))],
        [Paragraph("<b>APPLICATION & INFERENCE LAYER:</b> `predictor.py` &nbsp;|&nbsp; Image Normalizer &nbsp;|&nbsp; Top-K Class Ranking &nbsp;|&nbsp; ReportLab PDF Engine", ParagraphStyle('A2', fontSize=8.5, textColor=colors.HexColor("#1b365d"), alignment=1))],
        [Paragraph("↓", ParagraphStyle('Arrow', alignment=1, fontSize=10, textColor=colors.HexColor("#4a5568")))],
        [Paragraph("<b>DEEP LEARNING & XAI LAYER:</b> Custom 4-Block CNN (`cnn_model.py`) &nbsp;|&nbsp; Grad-CAM Saliency Engine (`gradcam.py`)", ParagraphStyle('A3', fontSize=8.5, textColor=colors.HexColor("#1b365d"), alignment=1))],
        [Paragraph("↓", ParagraphStyle('Arrow', alignment=1, fontSize=10, textColor=colors.HexColor("#4a5568")))],
        [Paragraph("<b>DATA & KNOWLEDGE LAYER:</b> `disease_info.json` (Agronomic KB) &nbsp;|&nbsp; `crop_disease_model.pth` &nbsp;|&nbsp; `class_names.json`", ParagraphStyle('A4', fontSize=8.5, textColor=colors.HexColor("#1b365d"), alignment=1))],
    ]
    t_arch = Table(arch_box, colWidths=[532])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), colors.HexColor("#edf2f7")),
        ('BACKGROUND', (0,2), (0,2), colors.HexColor("#e2e8f0")),
        ('BACKGROUND', (0,4), (0,4), colors.HexColor("#edf2f7")),
        ('BACKGROUND', (0,6), (0,6), colors.HexColor("#e2e8f0")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_arch)
    story.append(Paragraph("Figure 9.1: Multi-tier system architecture of AgriScan AI platform.", caption_style))

    story.append(Paragraph("9.2 Deployment Architecture", h2_style))
    story.append(Paragraph("• <b>Client Side:</b> End-user accesses the platform via any standard web browser (Chrome, Firefox, Safari) on desktop, tablet, or smartphone.", bullet_style))
    story.append(Paragraph("• <b>Cloud Host:</b> Streamlit Community Cloud runs the containerized Python 3.11 environment on Debian Linux with CPU-optimized PyTorch wheels.", bullet_style))
    story.append(Paragraph("• <b>API Service:</b> FastAPI runs under Uvicorn ASGI server with non-blocking worker threads handling multipart image prediction requests.", bullet_style))

    # =========================================================================
    # SECTION 10: SYSTEM DESIGN
    # =========================================================================
    story.append(Paragraph("10. System Design", h1_style))
    story.append(Paragraph("10.1 Use Case & Actor Interaction", h2_style))
    story.append(Paragraph(
        "The system serves two primary actors: the <b>Farmer / Agronomist</b> (who uploads leaf specimens, reviews Grad-CAM heatmaps, reads prescriptions, and downloads PDF reports), "
        "and the <b>System Maintainer / ML Engineer</b> (who re-trains models on new pathology classes, evaluates confusion matrices, and updates treatment protocols).",
        body_style
    ))

    story.append(Paragraph("10.2 Data Flow Diagram (DFD Level 1)", h2_style))
    dfd_data = [
        [Paragraph("<font color='white'><b>Process ID</b></font>", table_cell_header), Paragraph("<font color='white'><b>Input</b></font>", table_cell_header), Paragraph("<font color='white'><b>Transformation / Algorithm</b></font>", table_cell_header), Paragraph("<font color='white'><b>Output / Data Store</b></font>", table_cell_header)],
        [Paragraph("P1 — Acquisition", table_cell_bold), Paragraph("Foliar Image (JPG/PNG)", table_cell), Paragraph("MIME check, byte decoding, RGB conversion", table_cell), Paragraph("224x224 Normalized Tensor", table_cell)],
        [Paragraph("P2 — Inference", table_cell_bold), Paragraph("Input Tensor (3,224,224)", table_cell), Paragraph("4-block CNN forward pass + Softmax", table_cell), Paragraph("18-class probability distribution", table_cell)],
        [Paragraph("P3 — Saliency", table_cell_bold), Paragraph("Predicted Class Logit", table_cell), Paragraph("Grad-CAM backprop on ConvBlock 4", table_cell), Paragraph("Colormap heat overlay image", table_cell)],
        [Paragraph("P4 — Advisory", table_cell_bold), Paragraph("Top Class Label", table_cell), Paragraph("JSON Knowledge Base lookup", table_cell), Paragraph("Symptoms, chemical & organic remedies", table_cell)],
        [Paragraph("P5 — Export", table_cell_bold), Paragraph("Diagnosis Result Dict", table_cell), Paragraph("ReportLab Flowable document compilation", table_cell), Paragraph("Downloadable PDF certificate bytes", table_cell)],
    ]
    t_dfd = Table(dfd_data, colWidths=[70, 110, 180, 172])
    t_dfd.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1b365d")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_dfd)

    story.append(Paragraph("10.4 Project Directory Structure", h2_style))
    story.append(Paragraph("The physical implementation repository structure is organized cleanly as follows:", body_style))
    dir_code = (
        "hcl-crop-disease/\n"
        "├── data/\n"
        "│   ├── dataset/                      # 18 organized crop pathology folders (720 images)\n"
        "│   ├── sample_test_images/           # Curated test samples for 1-click evaluation\n"
        "│   └── disease_info.json             # Exhaustive agronomic treatment knowledge base\n"
        "├── models/\n"
        "│   ├── cnn_model.py                  # Custom 4-block deep CNN & MobileNetV2\n"
        "│   ├── gradcam.py                    # Explainable AI Grad-CAM heatmap engine\n"
        "│   ├── predictor.py                  # Production inference pipeline & treatment lookup\n"
        "│   └── saved_models/                 # Model checkpoint, confusion matrix, curves\n"
        "├── src/\n"
        "│   ├── dataset_builder.py            # Image synthesis & balancing pipeline\n"
        "│   ├── train.py                      # Training script with AdamW & LR scheduling\n"
        "│   ├── evaluate.py                   # Confusion matrix & classification report\n"
        "│   └── report_generator.py           # ReportLab PDF diagnostic certificate generator\n"
        "├── web/ (app.py, style.css)          # Modern Streamlit UI with tabs & contrast\n"
        "├── api/ (main.py)                    # FastAPI production asynchronous REST service\n"
        "└── tests/ (test_pipeline.py)         # Unit & integration test suite (7/7 passing)"
    )
    story.append(Paragraph(f"<font face='Courier' size=7.5 color='#2d3748'>{dir_code.replace(chr(10), '<br/>').replace(' ', '&nbsp;')}</font>", ParagraphStyle('CodeBox', leftIndent=10, spaceAfter=8)))

    story.append(PageBreak())

    # =========================================================================
    # SECTION 11: MODULE DESCRIPTION
    # =========================================================================
    story.append(Paragraph("11. Module Description", h1_style))
    story.append(Paragraph("<b>Module 1 — Image Ingestion & Preprocessing (`dataset_builder.py`, `predictor.py`)</b><br/>"
                           "Validates file dimensions and formats. Implements affine rotations, horizontal/vertical flipping, and color jittering during training. Normalizes input RGB channels using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].", body_style))
    story.append(Paragraph("<b>Module 2 — Deep Convolutional Neural Network (`cnn_model.py`)</b><br/>"
                           "Hierarchical feature extraction engine comprising 4 sequential ConvBlocks (32 -> 64 -> 128 -> 256 filters) with Batch Normalization, ReLU activations, MaxPool2d subsampling, and spatial dropout. Global Average Pooling replaces dense flatten layers, yielding a compact 4.86 MB model.", body_style))
    story.append(Paragraph("<b>Module 3 — Explainable AI Grad-CAM Engine (`gradcam.py`)</b><br/>"
                           "Registers forward and backward hooks on the last Conv2d layer of ConvBlock 4. Calculates the spatial gradient of target class logit with respect to feature maps, applies ReLU, and interpolates a jet-colormap heatmap blended at alpha=0.45 over the raw input leaf.", body_style))
    story.append(Paragraph("<b>Module 4 — Agronomic Knowledge & Diagnostic Engine (`predictor.py`)</b><br/>"
                           "Translates class indices into crop species and disease names. Computes severity ratings (Mild / Moderate / High Alert / Healthy) and cross-references `disease_info.json` to assemble an integrated diagnostic payload.", body_style))
    story.append(Paragraph("<b>Module 5 — Automated PDF Report Generator (`report_generator.py`)</b><br/>"
                           "Constructs an official, downloadable Agronomic Health Certificate utilizing ReportLab. Formats diagnostic findings, observed symptoms, bio-control remedies, and chemical spray schedules into an in-memory byte buffer.", body_style))
    story.append(Paragraph("<b>Module 6 — Web Presentation Layer (`web/app.py`, `web/style.css`)</b><br/>"
                           "Streamlit-based dashboard featuring cool-slate navigation, vibrant emerald headers, image previewers, Grad-CAM side-by-side visualizers, confidence bar charts, and live seasonal disease news cards.", body_style))
    story.append(Paragraph("<b>Module 7 — Asynchronous REST API (`api/main.py`)</b><br/>"
                           "FastAPI microservice exposing `/predict`, `/health`, and `/classes`. Provides CORS headers, Swagger OpenAPI documentation, and asynchronous multipart image payload handling.", body_style))
    story.append(Paragraph("<b>Module 8 — Evaluation & Verification Suite (`evaluate.py`, `test_pipeline.py`)</b><br/>"
                           "Executes full model evaluation on 720 test images, generating high-resolution 18-class confusion matrices, precision/recall reports, and automated unit tests across all project modules.", body_style))

    # =========================================================================
    # SECTION 12: DATABASE / KNOWLEDGE BASE DESIGN
    # =========================================================================
    story.append(Paragraph("12. Database / Knowledge Base Design", h1_style))
    story.append(Paragraph("12.1 Agronomic Disease Knowledge Base Structure (`disease_info.json`)", h2_style))
    
    kb_data = [
        [Paragraph("<font color='white'><b>JSON Key Field</b></font>", table_cell_header), Paragraph("<font color='white'><b>Data Type</b></font>", table_cell_header), Paragraph("<font color='white'><b>Agronomic Description & Purpose</b></font>", table_cell_header)],
        [Paragraph("crop", table_cell_bold), Paragraph("String", table_cell), Paragraph("Agricultural crop common name (Apple, Corn, Potato, Rice, Tomato).", table_cell)],
        [Paragraph("disease", table_cell_bold), Paragraph("String", table_cell), Paragraph("Specific pathology name (e.g., 'Early Blight', 'Leaf Blast', 'Healthy').", table_cell)],
        [Paragraph("is_healthy", table_cell_bold), Paragraph("Boolean", table_cell), Paragraph("Flag indicating whether specimen is disease-free or pathological.", table_cell)],
        [Paragraph("pathogen", table_cell_bold), Paragraph("String", table_cell), Paragraph("Scientific pathogen identity (e.g., 'Venturia inaequalis (Fungus)').", table_cell)],
        [Paragraph("severity", table_cell_bold), Paragraph("String", table_cell), Paragraph("Pathology severity index ('None', 'Moderate', 'High', 'Critical').", table_cell)],
        [Paragraph("symptoms", table_cell_bold), Paragraph("String", table_cell), Paragraph("Detailed visual and cellular foliar indicators observed on leaves.", table_cell)],
        [Paragraph("cause", table_cell_bold), Paragraph("String", table_cell), Paragraph("Environmental triggers (temperature, humidity, rainfall) favoring infection.", table_cell)],
        [Paragraph("organic_treatment", table_cell_bold), Paragraph("Array[String]", table_cell), Paragraph("List of biological and non-chemical control remedies (Bacillus, Neem oil).", table_cell)],
        [Paragraph("chemical_treatment", table_cell_bold), Paragraph("Array[String]", table_cell), Paragraph("Targeted chemical fungicide/bactericide compounds and spray dosages.", table_cell)],
        [Paragraph("prevention", table_cell_bold), Paragraph("Array[String]", table_cell), Paragraph("Long-term cultural and agronomic preventive field strategies.", table_cell)],
    ]
    t_kb = Table(kb_data, colWidths=[105, 75, 352])
    t_kb.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1b365d")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e0")),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
    ]))
    story.append(t_kb)

    story.append(PageBreak())

    # =========================================================================
    # SECTION 13: TESTING & VERIFICATION
    # =========================================================================
    story.append(Paragraph("13. Testing", h1_style))
    story.append(Paragraph("13.1 Test Cases & Results", h2_style))
    story.append(Paragraph("All core modules undergo rigorous automated unit and integration testing via `tests/test_pipeline.py`. All 7 test cases passed cleanly with 100% success rate:", body_style))

    test_data = [
        [Paragraph("<font color='white'><b>Test ID</b></font>", table_cell_header), Paragraph("<font color='white'><b>Test Module & Objective</b></font>", table_cell_header), Paragraph("<font color='white'><b>Expected Result</b></font>", table_cell_header), Paragraph("<font color='white'><b>Status</b></font>", table_cell_header)],
        [Paragraph("TC-01", table_cell_bold), Paragraph("CNN Forward Pass Tensor", table_cell), Paragraph("Model outputs shape `(batch, 18)` without error.", table_cell), Paragraph("<font color='#16a34a'><b>PASSED</b></font>", table_cell)],
        [Paragraph("TC-02", table_cell_bold), Paragraph("MobileNet Backbone Forward", table_cell), Paragraph("Transfer network outputs shape `(batch, 18)`.", table_cell), Paragraph("<font color='#16a34a'><b>PASSED</b></font>", table_cell)],
        [Paragraph("TC-03", table_cell_bold), Paragraph("Leaf Image Synthesis", table_cell), Paragraph("Generates 224x224 leaf geometry with pathology spots.", table_cell), Paragraph("<font color='#16a34a'><b>PASSED</b></font>", table_cell)],
        [Paragraph("TC-04", table_cell_bold), Paragraph("Predictor Pipeline Inference", table_cell), Paragraph("Returns top-k predictions, confidence %, and remedies.", table_cell), Paragraph("<font color='#16a34a'><b>PASSED</b></font>", table_cell)],
        [Paragraph("TC-05", table_cell_bold), Paragraph("Grad-CAM Saliency Overlay", table_cell), Paragraph("Generates normalized 2D heatmap overlaid on leaf.", table_cell), Paragraph("<font color='#16a34a'><b>PASSED</b></font>", table_cell)],
        [Paragraph("TC-06", table_cell_bold), Paragraph("PDF Report Generation", table_cell), Paragraph("Streams valid `%PDF` byte stream exceeding 1000 bytes.", table_cell), Paragraph("<font color='#16a34a'><b>PASSED</b></font>", table_cell)],
        [Paragraph("TC-07", table_cell_bold), Paragraph("FastAPI Endpoint Suite", table_cell), Paragraph("`/health` returns 200, `/classes` returns 18, `/predict` works.", table_cell), Paragraph("<font color='#16a34a'><b>PASSED</b></font>", table_cell)],
    ]
    t_test = Table(test_data, colWidths=[45, 145, 272, 70])
    t_test.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1b365d")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_test)

    story.append(Paragraph("13.2 Requirements Traceability Matrix (RTM)", h2_style))
    rtm_data = [
        [Paragraph("<font color='white'><b>Requirement ID</b></font>", table_cell_header), Paragraph("<font color='white'><b>Functional Description</b></font>", table_cell_header), Paragraph("<font color='white'><b>Implementing Module</b></font>", table_cell_header), Paragraph("<font color='white'><b>Verification Test</b></font>", table_cell_header)],
        [Paragraph("FR-01 / FR-02", table_cell_bold), Paragraph("Leaf Image Ingestion & Normalization", table_cell), Paragraph("`predictor.py`, `dataset_builder.py`", table_cell), Paragraph("TC-03, TC-04", table_cell)],
        [Paragraph("FR-03 / FR-04", table_cell_bold), Paragraph("CNN Feature Extraction & Classification", table_cell), Paragraph("`cnn_model.py`", table_cell), Paragraph("TC-01, TC-02", table_cell)],
        [Paragraph("FR-05", table_cell_bold), Paragraph("Grad-CAM Visual Saliency Mapping", table_cell), Paragraph("`gradcam.py`", table_cell), Paragraph("TC-05", table_cell)],
        [Paragraph("FR-06 / FR-07", table_cell_bold), Paragraph("Agronomic Advisory & PDF Export", table_cell), Paragraph("`report_generator.py`", table_cell), Paragraph("TC-06", table_cell)],
        [Paragraph("FR-08", table_cell_bold), Paragraph("FastAPI REST Endpoints", table_cell), Paragraph("`api/main.py`", table_cell), Paragraph("TC-07", table_cell)],
        [Paragraph("FR-09 / FR-10", table_cell_bold), Paragraph("Streamlit Web Dashboard & Analytics", table_cell), Paragraph("`web/app.py`", table_cell), Paragraph("Live Verification", table_cell)],
    ]
    t_rtm = Table(rtm_data, colWidths=[80, 180, 162, 110])
    t_rtm.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1b365d")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_rtm)

    story.append(PageBreak())

    # =========================================================================
    # SECTION 14: SYSTEM IMPLEMENTATION & SCREENSHOTS
    # =========================================================================
    story.append(Paragraph("14. System Implementation & Working Verification", h1_style))
    story.append(Paragraph(
        "The following screenshots document the live execution of the AgriScan AI platform on Streamlit Community Cloud "
        "(`https://agriscan-ai.streamlit.app`), demonstrating the user interface, real-time disease diagnosis, Grad-CAM attention localization, "
        "and seasonal pathology news advisories.",
        body_style
    ))

    # Screenshot 1: Working Diagnosis with Grad-CAM
    ss3_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "screenshots", "working_diagnosis_gradcam.png")
    if os.path.exists(ss3_path):
        story.append(KeepTogether([
            Paragraph("<b>Figure 14.1: Live Working Diagnosis & Grad-CAM Explainable AI Saliency Map</b>", h2_style),
            Paragraph("Demonstration of real potato leaf specimen upload (`images (2).jpg`), instant pathology identification (<i>Potato - Early Blight</i>), and side-by-side Grad-CAM heatmap highlighting necrotic lesions.", body_style),
            RLImage(ss3_path, width=530, height=298),
            Spacer(1, 10)
        ]))

    # Screenshot 2: App Overview & Input Panel
    ss1_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "screenshots", "app_overview.png")
    if os.path.exists(ss1_path):
        story.append(KeepTogether([
            Paragraph("<b>Figure 14.2: Main Dashboard, Navigation Tabs & Specimen Acquisition Panel</b>", h2_style),
            Paragraph("Overview of the deployed web application showing the vibrant emerald banner, horizontal navigation tabs, crops available in list order, and multi-mode specimen input selector.", body_style),
            RLImage(ss1_path, width=530, height=298),
            Spacer(1, 10)
        ]))

    story.append(PageBreak())

    # Screenshot 3: News & Precautions Panel
    ss2_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "screenshots", "alerts_precautions.png")
    if os.path.exists(ss2_path):
        story.append(KeepTogether([
            Paragraph("<b>Figure 14.3: Live Outbreak Bulletins & Smart Farming Precautions</b>", h2_style),
            Paragraph("Below-the-fold informative panel providing active seasonal alerts (Potato Late Blight, Tomato Whitefly TYLCV) and cultural prevention tips (Drip irrigation, 3-year crop rotation).", body_style),
            RLImage(ss2_path, width=530, height=298),
            Spacer(1, 10)
        ]))

    # =========================================================================
    # SECTION 15: LIMITATIONS
    # =========================================================================
    story.append(Paragraph("15. Limitations", h1_style))
    story.append(Paragraph("• <b>Lighting & Occlusion Variations:</b> Severe lens flare, motion blur, or heavy soil occlusion on foliage can reduce classification confidence scores.", bullet_style))
    story.append(Paragraph("• <b>Dual-Infection Scenarios:</b> If a single leaf suffers from simultaneous bacterial and fungal co-infection, the model identifies the dominant visual symptom pattern.", bullet_style))
    story.append(Paragraph("• <b>Single Leaf Orientation:</b> Best diagnostic performance is achieved when photographing individual leaf blades rather than broad-angle canopy overviews.", bullet_style))
    story.append(Paragraph("• <b>Hardware Free-Tier Limits:</b> On free cloud hosting tiers, cold starts may introduce a 15-20 second initialization delay upon container wake-up.", bullet_style))

    # =========================================================================
    # SECTION 16: FUTURE SCOPE
    # =========================================================================
    story.append(Paragraph("16. Future Scope", h1_style))
    story.append(Paragraph("• <b>Edge Quantization (ONNX / TensorRT):</b> Convert trained PyTorch weights into INT8 quantized models for real-time inference on Raspberry Pi 5 and NVIDIA Jetson Nano.", bullet_style))
    story.append(Paragraph("• <b>Autonomous Agricultural Drones:</b> Integrate the lightweight CNN into drone video streams for automated hectare-scale overhead crop scouting.", bullet_style))
    story.append(Paragraph("• <b>Multi-Lingual Voice Bot:</b> Integrate audio speech recognition allowing rural farmers to speak queries in Hindi, Spanish, or regional dialects.", bullet_style))
    story.append(Paragraph("• <b>Microclimate IoT Fusion:</b> Combine optical leaf diagnosis with real-time field weather sensors (humidity, leaf wetness hours) for predictive outbreak warnings.", bullet_style))

    # =========================================================================
    # SECTION 17: REFERENCES
    # =========================================================================
    story.append(Paragraph("17. References", h1_style))
    story.append(Paragraph("• Project GitHub Repository: <font color='#2b6cb0'>https://github.com/devuphub/hcl-crop-disease</font>", bullet_style))
    story.append(Paragraph("• Live Application URL: <font color='#2b6cb0'>https://agriscan-ai.streamlit.app</font>", bullet_style))
    story.append(Paragraph("• Selvaraju, R. R., et al. 'Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.' <i>IEEE ICCV</i>, 2017.", bullet_style))
    story.append(Paragraph("• Hughes, D., & Salathé, M. 'An open access repository of images on plant health to enable the development of mobile disease diagnostics.' <i>arXiv:1511.08060</i>, 2015.", bullet_style))
    story.append(Paragraph("• PyTorch Deep Learning Documentation: <font color='#2b6cb0'>https://pytorch.org/docs/</font>", bullet_style))
    story.append(Paragraph("• Streamlit Application Framework: <font color='#2b6cb0'>https://docs.streamlit.io/</font>", bullet_style))
    story.append(Paragraph("• FastAPI High Performance Framework: <font color='#2b6cb0'>https://fastapi.tiangolo.com/</font>", bullet_style))

    # =========================================================================
    # APPENDICES
    # =========================================================================
    story.append(Paragraph("Appendix A: Definitions & Abbreviations", h1_style))
    app_data = [
        [Paragraph("<font color='white'><b>Term</b></font>", table_cell_header), Paragraph("<font color='white'><b>Definition</b></font>", table_cell_header)],
        [Paragraph("CNN", table_cell_bold), Paragraph("Convolutional Neural Network; deep learning architecture specialized for image feature extraction.", table_cell)],
        [Paragraph("Grad-CAM", table_cell_bold), Paragraph("Gradient-weighted Class Activation Mapping; visual explainability technique for spatial feature visualization.", table_cell)],
        [Paragraph("GAP", table_cell_bold), Paragraph("Global Average Pooling; dimensional reduction technique averaging spatial planes into scalar outputs.", table_cell)],
        [Paragraph("XAI", table_cell_bold), Paragraph("Explainable Artificial Intelligence; methodologies providing human-interpretable reasoning for AI outputs.", table_cell)],
        [Paragraph("REST", table_cell_bold), Paragraph("Representational State Transfer; standard architectural style for networked web API services.", table_cell)],
        [Paragraph("Softmax", table_cell_bold), Paragraph("Mathematical activation function normalizing real-valued logits into a multi-class probability distribution.", table_cell)],
    ]
    t_app = Table(app_data, colWidths=[90, 442])
    t_app.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1b365d")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_app)

    story.append(Paragraph("Appendix B: Submission Notes", h1_style))
    story.append(Paragraph(
        "This Software Requirements Specification (SRS) document complies with formal academic project evaluation formats. "
        "The system has been trained, verified via automated test suites (7/7 tests passed), containerized, and deployed to a public cloud URL "
        "accessible at `https://agriscan-ai.streamlit.app`. All source code, dataset scripts, trained model checkpoints, and evaluation graphs "
        "are publicly maintained under MIT License at `https://github.com/devuphub/hcl-crop-disease`.",
        body_style
    ))

    # Build PDF with dynamic NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated academic SRS PDF: {output_pdf_path}")
    return output_pdf_path


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
    os.makedirs(out_dir, exist_ok=True)
    pdf_file = os.path.join(out_dir, "Crop_Disease_Detection_SRS_Report.pdf")
    build_srs_pdf(pdf_file)
