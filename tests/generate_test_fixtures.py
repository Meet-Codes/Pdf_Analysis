"""
Test Fixture Generator: Creates realistic, high-fidelity PDF test cases using reportlab and Pillow.
Includes native text PDFs, image-based scanned PDFs, and misleading filename PDFs.
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import fitz
from PIL import Image, ImageDraw, ImageFont

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)


def create_motor_cv_pdf() -> Path:
    """Generates TATA ACE commercial vehicle insurance PDF."""
    path = FIXTURES_DIR / "tata_ace_insurance.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 750, "COMMERCIAL VEHICLE PACKAGE POLICY")
    c.setFont("Helvetica", 10)
    c.drawString(50, 725, "Certificate of Insurance cum Schedule")
    c.drawString(50, 700, "Insurer: Reliance General Insurance Co. Ltd.")
    c.drawString(50, 680, "Policy Number: P0023200023/4115/103739")
    c.drawString(50, 660, "Name of Insured: MR. MEET KORAT")
    c.drawString(50, 640, "Address: 104 Ring Road, Rajkot, Gujarat 360005")
    c.drawString(50, 620, "Mobile: +91 98765 43210")
    c.drawString(50, 600, "Registration No: GJ 03 MG 6586")
    c.drawString(50, 580, "Make & Model: TATA ACE GOODS CARRIER")
    c.drawString(50, 560, "Engine No: ENG987654321")
    c.drawString(50, 540, "Chassis No: CHA123456789")
    c.drawString(50, 520, "Period of Insurance: From 29/03/2023 to 28/03/2024")
    c.drawString(50, 500, "Coverage: Third Party Liability Only")
    c.drawString(50, 480, "Third Party Premium: Rs. 15,430/-")
    c.drawString(50, 460, "GST (18%): Rs. 2,777/-")
    c.drawString(50, 440, "Total Premium: Rs. 18,207/-")
    c.save()
    return path


def create_misleading_electricity_pdf() -> Path:
    """
    CRITICAL ACCEPTANCE TEST:
    File named 'PANCARD_17041098095175.pdf' but actual content is an ELECTRICITY BILL.
    Content-based classifier MUST identify as ELECTRICITY_BILL.
    """
    path = FIXTURES_DIR / "PANCARD_17041098095175.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 750, "PASCHIM GUJARAT VIJ COMPANY LIMITED (PGVCL)")
    c.setFont("Helvetica", 10)
    c.drawString(50, 725, "ELECTRICITY BILL - CUM - NOTICE")
    c.drawString(50, 700, "Consumer No: 12345678901")
    c.drawString(50, 680, "Consumer Name: MR MEET KORAT")
    c.drawString(50, 660, "Billing Address: Rajkot, Gujarat")
    c.drawString(50, 640, "Meter No: MTR998877")
    c.drawString(50, 620, "Tariff: RGP Single Phase")
    c.drawString(50, 600, "Bill Date: 10/05/2023")
    c.drawString(50, 580, "Due Date: 25/05/2023")
    c.drawString(50, 560, "Previous Reading: 4500.00")
    c.drawString(50, 540, "Current Reading: 4850.00")
    c.drawString(50, 520, "Units Consumed: 350")
    c.drawString(50, 500, "Energy Charges: Rs. 1,750/-")
    c.drawString(50, 480, "Fixed Charges: Rs. 150/-")
    c.drawString(50, 460, "Fuel Surcharge: Rs. 350/-")
    c.drawString(50, 440, "Electricity Duty: Rs. 250/-")
    c.drawString(50, 420, "Total Bill Amount: Rs. 2,500/-")
    c.save()
    return path


def create_health_policy_pdf() -> Path:
    """Generates Health Insurance PDF."""
    path = FIXTURES_DIR / "health_insurance.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 750, "FAMILY FLOATER HEALTH INSURANCE POLICY")
    c.setFont("Helvetica", 10)
    c.drawString(50, 725, "HDFC ERGO Health Insurance Limited")
    c.drawString(50, 700, "Policy Number: HDFC/HLTH/2023/987654")
    c.drawString(50, 680, "Policyholder Name: Mr Rajesh Patel")
    c.drawString(50, 660, "Plan Name: Health Suraksha Platinum")
    c.drawString(50, 640, "Sum Insured: Rs. 5,00,000/-")
    c.drawString(50, 620, "Policy Period: From 01/04/2023 to 31/03/2024")
    c.drawString(50, 600, "Hospitalization Cover: Up to Sum Insured")
    c.drawString(50, 580, "Total Premium: Rs. 18,500/-")
    c.save()
    return path


def create_scanned_motor_pdf() -> Path:
    """
    Creates an image-based scanned PDF where text exists ONLY as raster pixels.
    Requires OCR to read!
    """
    path = FIXTURES_DIR / "scanned_car_policy.pdf"
    # Create PIL image with printed text
    img = Image.new("RGB", (1200, 1600), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    lines = [
        "PRIVATE CAR LIABILITY ONLY POLICY",
        "Insurer: Bajaj Allianz General Insurance",
        "Policy No: P0023200023/4115/103739",
        "Name of Insured: MR MEET KORAT / 3654789/5",
        "Vehicle: TATA ACE",
        "Registration No: GJ 03 MG 6586",
        "Period: From 29/03/2023 to 28/03/2024",
        "Third Party Liability: Rs. 15,430/-",
        "GST: Rs. 2,777/-",
        "Total Premium: Rs. 18,207/-",
    ]

    y = 100
    for line in lines:
        draw.text((80, y), line, fill=(0, 0, 0))
        y += 60

    # Save as PDF using PyMuPDF or PIL
    img_pdf_path = FIXTURES_DIR / "temp_scanned.pdf"
    img.save(img_pdf_path, "PDF", resolution=150.0)
    if img_pdf_path.exists():
        img_pdf_path.replace(path)
    return path


def create_table_pdf() -> Path:
    """Creates a PDF with structured tables."""
    path = FIXTURES_DIR / "property_with_table.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=letter)
    styles = getSampleStyleSheet()

    story = [
        Paragraph("STANDARD FIRE AND SPECIAL PERILS INSURANCE", styles["Heading1"]),
        Spacer(1, 10),
        Paragraph("Insured Name: ABC Manufacturing Works", styles["Normal"]),
        Paragraph("Policy Number: FIRE/2023/00112233", styles["Normal"]),
        Paragraph("Policy Period: From 01/01/2023 to 31/12/2023", styles["Normal"]),
        Spacer(1, 15),
    ]

    table_data = [
        ["Asset Description", "Sum Insured (INR)"],
        ["Building", "5000000"],
        ["Plant & Machinery", "2500000"],
        ["Furniture & Fixtures", "500000"],
        ["Stocks in Process", "1000000"],
        ["Total Sum Insured", "9000000"],
    ]
    t = Table(table_data, colWidths=[200, 150])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(t)
    story.append(Spacer(1, 15))
    story.append(Paragraph("Total Premium: Rs. 25,000/-", styles["Normal"]))

    doc.build(story)
    return path


def generate_all_fixtures():
    """Generates all test documents."""
    create_motor_cv_pdf()
    create_misleading_electricity_pdf()
    create_health_policy_pdf()
    create_scanned_motor_pdf()
    create_table_pdf()
    print("Test fixtures successfully generated in:", FIXTURES_DIR)


if __name__ == "__main__":
    generate_all_fixtures()
