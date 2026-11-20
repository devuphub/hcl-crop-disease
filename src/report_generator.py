"""
Agronomic PDF Health & Treatment Report Generator
Generates exportable diagnostic certificates and treatment schedules for farmers and agronomists.
"""

import os
from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def generate_pdf_report(diagnosis_result: dict) -> bytes:
    """
    Builds a professional PDF advisory report from diagnosis output.
    Returns bytes buffer.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1b5e20"),
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor("#555555"),
        spaceAfter=15
    )

    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#2e7d32"),
        spaceBefore=10,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#222222")
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=13,
        leftIndent=15,
        textColor=colors.HexColor("#2b2b2b"),
        spaceAfter=3
    )

    elements = []

    # Title & Header
    elements.append(Paragraph("AgriScan AI - Crop Disease Diagnostic Advisory", title_style))
    elements.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Deep Learning CNN Inspection", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2e7d32"), spaceAfter=12))

    best = diagnosis_result.get("primary_prediction", {})
    details = diagnosis_result.get("details", {})
    severity = diagnosis_result.get("severity", "Moderate")

    # Summary Table
    table_data = [
        [Paragraph("<b>Crop Inspected:</b>", body_style), Paragraph(str(best.get("crop", "Unknown")), body_style)],
        [Paragraph("<b>Diagnosis:</b>", body_style), Paragraph(f"<b>{best.get('disease', 'Unknown')}</b>", body_style)],
        [Paragraph("<b>Confidence Level:</b>", body_style), Paragraph(f"{best.get('confidence', 0)}%", body_style)],
        [Paragraph("<b>Severity Assessment:</b>", body_style), Paragraph(str(severity), body_style)],
        [Paragraph("<b>Identified Pathogen:</b>", body_style), Paragraph(str(details.get("pathogen", "N/A")), body_style)],
        [Paragraph("<b>Environmental Cause:</b>", body_style), Paragraph(str(details.get("cause", "N/A")), body_style)],
    ]

    summary_table = Table(table_data, colWidths=[150, 380])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f1f8e9")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#c5e1a5")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 12))

    # Symptoms
    elements.append(Paragraph("Observed Foliar Symptoms", section_heading))
    elements.append(Paragraph(str(details.get("symptoms", "No symptom records available.")), body_style))
    elements.append(Spacer(1, 10))

    # Organic Treatments
    elements.append(Paragraph("Recommended Organic & Bio-Control Remedies", section_heading))
    organic_list = details.get("organic_treatment", ["Continue standard care."])
    for item in organic_list:
        elements.append(Paragraph(f"• {item}", bullet_style))
    elements.append(Spacer(1, 8))

    # Chemical Treatments
    elements.append(Paragraph("Recommended Targeted Chemical Control", section_heading))
    chemical_list = details.get("chemical_treatment", ["Consult extension agent."])
    for item in chemical_list:
        elements.append(Paragraph(f"• {item}", bullet_style))
    elements.append(Spacer(1, 8))

    # Prevention
    elements.append(Paragraph("Long-Term Cultural & Preventive Strategies", section_heading))
    prev_list = details.get("prevention", ["Regular monitoring recommended."])
    for item in prev_list:
        elements.append(Paragraph(f"• {item}", bullet_style))
    elements.append(Spacer(1, 15))

    # Footer Disclaimer
    disclaimer_style = ParagraphStyle('Disclaimer', parent=styles['Italic'], fontSize=8, textColor=colors.gray)
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.gray, spaceAfter=6))
    elements.append(Paragraph("Disclaimer: This advisory is generated via Convolutional Neural Network automated visual inspection. Always verify with local agricultural extension officers before applying chemical sprays in high volumes.", disclaimer_style))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
