# app.py
import streamlit as st
import pandas as pd
from datetime import datetime, date
import os
import hashlib
from pathlib import Path
import re
from PIL import Image
import io

# --- Robust imports for PDF & OCR ---
# PyMuPDF can be imported as `fitz` (classic) or `pymupdf` (newer).
try:
    import pymupdf as fitz  # PyMuPDF
except Exception:
    try:
        import fitz  # type: ignore
    except Exception as _e:
        fitz = None

try:
    import pytesseract
except Exception:
    pytesseract = None

st.set_page_config(page_title="AI.Healthcare", layout="wide")

FILES_MANIFEST_CSV = "patient_files.csv"
MEASUREMENTS_CSV = "patient_measurements.csv"
UPLOADS_DIR = "uploads"

# Document type hierarchy
DOCUMENT_SUBTYPES = {
    "Imaging": ["X-ray", "CT", "MRI", "PET", "Ultrasound", "Other"],
    "Labs": ["Blood Work", "Urinalysis", "Culture", "Pathology", "Other"],
    "Vital Signs": ["Blood Pressure", "Heart Rate", "Temperature", "Oxygen Saturation", "Respiratory Rate", "Weight", "Height", "Other"],
    "Media": ["Photo", "Video", "Audio", "Other"],
    "Consultation": ["Specialist Note", "Primary Care", "Emergency", "Other"],
    "Discharge Summary": ["Hospital", "Clinic", "Emergency Department", "Other"],
    "Medication Records": ["Prescription", "Immunization", "Medication List", "Other"],
    "Other": []
}

# Document scanning keywords for type detection
DOC_TYPE_KEYWORDS = {
    "Imaging": ["x-ray", "xray", "radiograph", "ct scan", "mri", "pet scan", "ultrasound", "imaging", "radiology"],
    "Labs": ["laboratory", "lab results", "blood work", "cbc", "urinalysis", "pathology", "culture", "biopsy"],
    "Vital Signs": ["vital signs", "vitals", "blood pressure", "heart rate", "pulse", "temperature", "oxygen saturation", "spo2", "respiratory rate", "weight", "height", "bmi"],
    "Media": ["photo", "photograph", "image", "video", "recording", "audio", "media file"],
    "Consultation": ["consultation", "specialist", "referral", "assessment", "evaluation", "visit note", "provider note"],
    "Discharge Summary": ["discharge", "hospital discharge", "summary", "admission", "released"],
    "Medication Records": ["prescription", "medication", "immunization", "vaccine", "rx", "pharmacy"],
    "Other": []
}

DOC_SUBTYPE_KEYWORDS = {
    "X-ray": ["x-ray", "xray", "radiograph"],
    "CT": ["ct scan", "computed tomography", "cat scan"],
    "MRI": ["mri", "magnetic resonance"],
    "PET": ["pet scan", "positron emission"],
    "Ultrasound": ["ultrasound", "sonogram", "echo"],
    "Blood Work": ["blood work", "blood", "cbc", "complete blood count", "blood test", "hematology", "hemoglobin", "wbc", "rbc"],
    "Urinalysis": ["urinalysis", "urine", "urine test", "ua"],
    "Culture": ["culture", "bacterial culture"],
    "Pathology": ["pathology", "biopsy", "tissue", "histology"],
    "Blood Pressure": ["blood pressure", "bp", "systolic", "diastolic", "mmhg", "hypertension"],
    "Heart Rate": ["heart rate", "pulse", "bpm", "beats per minute", "hr"],
    "Temperature": ["temperature", "temp", "fever", "celsius", "fahrenheit"],
    "Oxygen Saturation": ["oxygen saturation", "spo2", "o2 sat", "pulse ox", "oximetry"],
    "Respiratory Rate": ["respiratory rate", "rr", "respiration", "breaths per minute"],
    "Weight": ["weight", "wt", "kg", "lbs", "pounds", "kilograms", "body weight"],
    "Height": ["height", "ht", "cm", "inches", "feet", "stature"]
}

def _pymupdf_ok() -> bool:
    return fitz is not None

def _tesseract_ok() -> bool:
    return pytesseract is not None

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF using PyMuPDF; if low text, render pages and OCR with Tesseract if available."""
    if not _pymupdf_ok():
        return ""

    try:
        # First, try direct text extraction
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()

        # If little text (likely scanned), try OCR path
        if len(text.strip()) < 100 and _tesseract_ok():
            try:
                ocr_text = ""
                for page in doc:
                    pix = page.get_pixmap(dpi=300)
                    img_data = pix.tobytes("png")
                    image = Image.open(io.BytesIO(img_data))
                    ocr_text += pytesseract.image_to_string(image)
                    ocr_text += "\n"
                doc.close()
                if len(ocr_text.strip()) > len(text.strip()):
                    return ocr_text
            except Exception:
                doc.close()
                return text

        doc.close()
        return text
    except Exception:
        return ""

def extract_text_from_image(file_bytes: bytes) -> str:
    """Extract text from image using OCR (pytesseract)"""
    if not _tesseract_ok():
        return ""
    try:
        image = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(image)
        return text
    except Exception:
        return ""

def extract_text_from_file(uploaded_file) -> str:
    """Extract text from uploaded file based on type"""
    file_ext = Path(uploaded_file.name).suffix.lower()
    try:
        file_bytes = uploaded_file.getvalue()
        if file_ext == ".pdf":
            return extract_text_from_pdf(file_bytes)
        elif file_ext in [".txt", ".csv"]:
            return file_bytes.decode("utf-8", errors="ignore")
        elif file_ext in [".jpg", ".jpeg", ".png", ".tiff"]:
            return extract_text_from_image(file_bytes)
        else:
            return ""
    except Exception:
        return ""

def detect_document_type(text: str):
    """Detect document type based on keyword matching"""
    if not text:
        return "Other", []

    text_lower = text.lower()
    scores = {}

    # Score each document type
    for doc_type, keywords in DOC_TYPE_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            scores[doc_type] = score

    if not scores:
        return "Other", []

    # Best matching type
    best_type = max(scores, key=scores.get)

    # Detect subtypes (>=1 match)
    subtype_scores = {}
    for subtype, keywords in DOC_SUBTYPE_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            subtype_scores[subtype] = score

    detected_subtypes = sorted(subtype_scores.keys(), key=lambda x: subtype_scores[x], reverse=True) if subtype_scores else []
    return best_type, detected_subtypes

def _parse_possible_date(s: str):
    from datetime import datetime
    for fmt in ("%m/%d/%Y", "%m-%d-%Y", "%m/%d/%y", "%m-%d-%y", "%Y-%m-%d", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            continue
    # last resort: dateutil if available
    try:
        from dateutil import parser
        return parser.parse(s, fuzzy=True).date()
    except Exception:
        return None

def extract_dates_from_text(text: str):
    """Extract potential dates from text and return the most recent one if possible."""
    if not text:
        return None

    date_patterns = [
        r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",   # MM/DD/YYYY or MM-DD-YYYY
        r"\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b",     # YYYY-MM-DD
        r"\b([A-Za-z]+ \d{1,2},\s*\d{4})\b",      # Month DD, YYYY
    ]

    matches = []
    for pat in date_patterns:
        matches.extend(re.findall(pat, text))

    # Normalize to most recent
    parsed_dates = []
    for m in matches:
        d = _parse_possible_date(m)
        if d:
            parsed_dates.append(d)

    if parsed_dates:
        most_recent = max(parsed_dates)
        return most_recent.strftime("%m/%d/%Y")

    return None

def scan_document(uploaded_file):
    """Scan document and return detected type, subtypes (list), and date"""
    text = extract_text_from_file(uploaded_file)
    doc_type, doc_subtypes = detect_document_type(text)
    extracted_date = extract_dates_from_text(text)

    return {
        "document_type": doc_type,
        "document_subtypes": doc_subtypes,  # list
        "date": extracted_date,
        "confidence": "medium" if text else "low",
        "text_preview": text[:500] if text else ""
    }

def extract_measurements_from_text(text, document_id, patient_id, document_date):
    """Extract structured measurements (lab values, vital signs) from text"""
    if not text:
        return []

    measurements = []
    text_lower = text.lower()

    # Lab value patterns
    lab_patterns = {
        "Potassium": [r"(?:potassium|(?<![a-z0-9])k)\+?[\s\-–—:;=,]+(\d+\.?\d*)", "mEq/L"],
        "Sodium": [r"(?:sodium|(?<![a-z0-9])na)\+?[\s\-–—:;=,]+(\d+\.?\d*)", "mEq/L"],
        "Hemoglobin": [r"(?:hemoglobin|hgb|(?<![a-z0-9])hb)[\s\-–—:;=,]+(\d+\.?\d*)", "g/dL"],
        "Hematocrit": [r"(?:hematocrit|hct)[\s\-–—:;=,]+(\d+\.?\d*)", "%"],
        "WBC": [r"(?:wbc|white\s+blood\s+cell)[\s\-–—:;=,]+(\d+\.?\d*)", "K/uL"],
        "Platelets": [r"(?:platelets|platelet|plt)[\s\-–—:;=,]+(\d+\.?\d*)", "K/uL"],
        "Glucose": [r"(?:glucose|glu)[\s\-–—:;=,]+(\d+\.?\d*)", "mg/dL"],
        "Creatinine": [r"(?:creatinine|creat|(?<![a-z0-9])cr)[\s\-–—:;=,]+(\d+\.?\d*)", "mg/dL"],
    }

    # Vital signs patterns
    vital_patterns = {
        "Blood Pressure Systolic": [r"(?:bp|blood\s+pressure)[\s\-–—:;=,]+(\d+)\s*/\s*\d+", "mmHg"],
        "Blood Pressure Diastolic": [r"(?:bp|blood\s+pressure)[\s\-–—:;=,]+\d+\s*/\s*(\d+)", "mmHg"],
        "Heart Rate": [r"(?:hr|heart\s+rate|pulse)[\s\-–—:;=,]+(\d+\.?\d*)", "bpm"],
        "Temperature": [r"(?:temp|temperature)[\s\-–—:;=,°]+(\d+\.?\d*)", "°F"],
        "Oxygen Saturation": [r"(?:spo2|o2\s+sat|oxygen\s+saturation|pulse\s+ox)[\s\-–—:;=,]+(\d+\.?\d*)", "%"],
        "Respiratory Rate": [r"(?:rr|resp\s+rate|respiratory\s+rate)[\s\-–—:;=,]+(\d+\.?\d*)", "breaths/min"],
        "Weight": [r"(?:weight|wt)[\s\-–—:;=,]+(\d+\.?\d*)", "lbs"],
        "Height": [r"(?:height|ht)[\s\-–—:;=,]+(\d+\.?\d*)", "inches"],
    }

    # Extract lab values
    for lab_name, (pattern, default_unit) in lab_patterns.items():
        matches = re.findall(pattern, text_lower)
        for match in matches:
            value = match[0] if isinstance(match, tuple) else match
            try:
                measurements.append({
                    "PatientID": patient_id,
                    "DocumentID": document_id,
                    "MeasurementGroup": "Labs",
                    "MeasurementName": lab_name,
                    "ValueNumeric": float(value),
                    "ValueText": value,
                    "Unit": default_unit,
                    "DateTime": document_date,
                    "CreatedAt": datetime.now().isoformat()
                })
            except Exception:
                continue

    # Extract vital signs
    for vital_name, (pattern, default_unit) in vital_patterns.items():
        matches = re.findall(pattern, text_lower)
        for match in matches:
            value = match if isinstance(match, str) else match[0] if isinstance(match, tuple) else match
            try:
                measurements.append({
                    "PatientID": patient_id,
                    "DocumentID": document_id,
                    "MeasurementGroup": "Vital Signs",
                    "MeasurementName": vital_name,
                    "ValueNumeric": float(value),
                    "ValueText": value,
                    "Unit": default_unit,
                    "DateTime": document_date,
                    "CreatedAt": datetime.now().isoformat()
                })
            except Exception:
                continue

    return measurements

def load_measurements():
    if os.path.exists(MEASUREMENTS_CSV):
        return pd.read_csv(MEASUREMENTS_CSV)
    cols = ["PatientID", "DocumentID", "MeasurementGroup", "MeasurementName", "ValueNumeric", "ValueText", "Unit", "DateTime", "CreatedAt"]
    return pd.DataFrame(columns=cols)

def save_measurements(measurements_list):
    if not measurements_list:
        return
    existing_df = load_measurements()
    new_df = pd.DataFrame(measurements_list)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df.to_csv(MEASUREMENTS_CSV, index=False)

def patient_id(name: str, dob: str) -> str:
    return hashlib.md5(f"{name.strip().lower()}_{dob.strip()}".encode()).hexdigest()[:16]

def ensure_uploads_dir():
    if not os.path.exists(UPLOADS_DIR):
        os.makedirs(UPLOADS_DIR, exist_ok=True)

def load_files_manifest():
    if os.path.exists(FILES_MANIFEST_CSV):
        return pd.read_csv(FILES_MANIFEST_CSV)
    cols = ["PatientID", "Name", "DOB", "DocumentType", "DocumentSubType", "OriginalFilename", "StoredPath", "UploadTimestamp", "ResultsReceivedDate", "Notes", "FileSize"]
    return pd.DataFrame(columns=cols)

def save_files_manifest(df: pd.DataFrame):
    df.to_csv(FILES_MANIFEST_CSV, index=False)

def save_uploaded_file(uploaded_file, patient_id: str, patient_name: str, patient_dob: str, doc_type: str, doc_subtype: str, results_date: str, notes: str):
    ensure_uploads_dir()

    # patient folder
    patient_dir = os.path.join(UPLOADS_DIR, patient_id)
    os.makedirs(patient_dir, exist_ok=True)

    # filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_name = Path(uploaded_file.name).name
    safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in original_name).replace(" ", "_")
    safe_filename = f"{timestamp}_{safe_name}"
    file_path = os.path.join(patient_dir, safe_filename)

    # doc id
    document_id = hashlib.md5(f"{patient_id}_{safe_filename}".encode()).hexdigest()[:16]

    # Extract text BEFORE saving (so the file buffer is intact)
    extracted_text = None
    if doc_type in {"Labs", "Vital Signs"}:
        try:
            uploaded_file.seek(0)
            extracted_text = extract_text_from_file(uploaded_file)
            uploaded_file.seek(0)
        except Exception as e:
            print(f"Error extracting text: {e}")

    # Save file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Record in manifest
    manifest = load_files_manifest()
    file_size = getattr(uploaded_file, "size", None)
    if file_size is None:
        try:
            file_size = os.path.getsize(file_path)
        except Exception:
            file_size = 0

    new_record = pd.DataFrame([{
        "PatientID": patient_id,
        "Name": patient_name,
        "DOB": patient_dob,
        "DocumentType": doc_type,
        "DocumentSubType": doc_subtype,
        "OriginalFilename": original_name,
        "StoredPath": file_path,
        "UploadTimestamp": datetime.now().isoformat(),
        "ResultsReceivedDate": results_date,
        "Notes": notes,
        "FileSize": file_size,
    }])
    updated_manifest = pd.concat([manifest, new_record], ignore_index=True)
    save_files_manifest(updated_manifest)

    # Extract measurements
    if extracted_text:
        try:
            measurements = extract_measurements_from_text(extracted_text, document_id, patient_id, results_date)
            if measurements:
                save_measurements(measurements)
                return True, f"Extracted {len(measurements)} measurements"
            else:
                return True, "No measurements found in document"
        except Exception as e:
            return True, f"Error extracting measurements: {e}"
    else:
        if doc_type in ["Labs", "Vital Signs"]:
            return True, "Could not extract text from document"

    return True

def delete_uploaded_file(stored_path: str):
    if os.path.exists(stored_path):
        try:
            os.remove(stored_path)
        except Exception:
            pass
    manifest = load_files_manifest()
    updated_manifest = manifest[manifest["StoredPath"] != stored_path]
    save_files_manifest(updated_manifest)

def format_file_size(size_bytes: float) -> str:
    try:
        size_bytes = float(size_bytes)
    except Exception:
        return "N/A"
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def display_document_card(row, idx):
    doc_type = row.get("DocumentType", "N/A")
    doc_subtype = row.get("DocumentSubType", "")

    title = f"📄 {doc_type} - {row['OriginalFilename']}" if not doc_subtype else f"📄 {doc_type} - {doc_subtype} - {row['OriginalFilename']}"
    with st.expander(title, expanded=False):
        col1, col2 = st.columns([3, 1])

        with col1:
            st.write(f"**Document Type:** {doc_type}")
            if pd.notna(doc_subtype) and doc_subtype:
                st.write(f"**Sub-Type:** {doc_subtype}")
            st.write(f"**Filename:** {row['OriginalFilename']}")
            st.write(f"**Upload Date:** {row['UploadTimestamp']}")
            if pd.notna(row.get("ResultsReceivedDate")) and row.get("ResultsReceivedDate"):
                st.write(f"**Results Received:** {row['ResultsReceivedDate']}")
            st.write(f"**File Size:** {format_file_size(row['FileSize'])}")
            if pd.notna(row.get("Notes")) and row.get("Notes"):
                st.write(f"**Notes:** {row['Notes']}")

        with col2:
            if os.path.exists(row["StoredPath"]):
                try:
                    with open(row["StoredPath"], "rb") as f:
                        file_data = f.read()
                    st.download_button(
                        label="⬇️ Download",
                        data=file_data,
                        file_name=row["OriginalFilename"],
                        mime="application/octet-stream",
                        key=f"download_{idx}",
                    )
                except Exception:
                    st.caption("File unavailable.")
                if st.button("🗑️ Delete", key=f"delete_{idx}"):
                    delete_uploaded_file(row["StoredPath"])
                    st.rerun()

# ---------------- Session State ----------------
if "current_page" not in st.session_state:
    st.session_state.current_page = "upload"

# Sidebar - Patient Information
with st.sidebar:
    st.header("Patient Information")

    # Streamlit's date_input sometimes does not accept None depending on version.
    # We'll provide a sensible default and clear instruction.
    name = st.text_input("Name *")
    default_dob = st.session_state.get("dob_default", date(1990, 1, 1))
    dob_date = st.date_input(
        "DOB *",
        value=default_dob,
        format="MM/DD/YYYY",
        min_value=date(1900, 1, 1),
        max_value=date.today(),
        key="dob_date_input"
    )
    st.caption("Enter Name & DOB to start.")

    st.divider()

    # Navigation
    if name and dob_date:
        st.write("**Navigation:**")
        if st.button("📤 Upload Documents", use_container_width=True):
            st.session_state.current_page = "upload"
            st.rerun()
        if st.button("📂 View Documents", use_container_width=True):
            st.session_state.current_page = "view_documents"
            st.rerun()

# Guard
if not name or not dob_date:
    st.title("AI.Healthcare")
    st.info("Add the patient's Name and DOB in the sidebar to begin.")
    st.stop()

dob = dob_date.strftime("%m/%d/%Y")
pid = patient_id(name, dob)

# Main
st.title("AI.Healthcare")
st.subheader("Hospital Admission Documents")

manifest = load_files_manifest()
patient_files = manifest[manifest["PatientID"] == pid]

# ==================== UPLOAD PAGE ====================
if st.session_state.current_page == "upload":
    st.write("### Upload New Documents")
    st.caption("Upload hospital admission records, medical documents, or related files (PDF, images, CSV).")

    if "scanned_info" not in st.session_state:
        st.session_state.scanned_info = None

    uploaded_files = st.file_uploader(
        "Choose file(s) to upload",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "csv", "txt"],
        accept_multiple_files=True,
        help="Select one or more files to upload. The first file will be automatically scanned for document type, sub-type(s), and date.",
        key="file_uploader",
    )

    # Auto-scan first file
    if uploaded_files:
        current_filename = uploaded_files[0].name
        if st.session_state.get("scan_filename") != current_filename:
            with st.spinner("🔍 Automatically scanning document..."):
                scan_result = scan_document(uploaded_files[0])
                st.session_state.scanned_info = scan_result
                st.session_state.scan_filename = current_filename

                # Pre-fill widgets
                if scan_result["document_type"] in ["Labs", "Imaging", "Vital Signs", "Media", "Consultation", "Discharge Summary", "Medication Records", "Other"]:
                    st.session_state.doc_type_select = scan_result["document_type"]

                if scan_result["document_subtypes"] is not None:
                    st.session_state.doc_subtype_select = scan_result["document_subtypes"]

                if scan_result["date"]:
                    try:
                        from dateutil import parser
                        parsed_date = parser.parse(scan_result["date"])
                        st.session_state.results_date_input = parsed_date.date()
                    except Exception:
                        pass

                st.rerun()

    # Show scan results
    if st.session_state.get("scanned_info"):
        scan_result = st.session_state.scanned_info
        st.write("---")
        st.write("**📄 Document Scan Results:**")
        if scan_result["confidence"] == "medium":
            subtypes_text = ""
            if scan_result.get("document_subtypes"):
                subtypes_list = scan_result["document_subtypes"]
                if subtypes_list:
                    subtypes_text = f" → **{', '.join(subtypes_list)}**"
            st.success(
                f"✓ Detected: **{scan_result['document_type']}**"
                + subtypes_text
                + (f" (Date: **{scan_result['date']}**)" if scan_result["date"] else "")
            )
            st.info("📋 Fields below have been auto-filled. Review and adjust if needed before saving.")
        else:
            st.warning("⚠️ Could not automatically detect document details. Please fill in the fields below manually.")
        st.write("---")

    # Defaults
    default_doc_type = "Labs"
    default_date = date.today()
    if st.session_state.scanned_info:
        scanned = st.session_state.scanned_info
        if scanned["document_type"] in ["Labs", "Imaging", "Vital Signs", "Media", "Consultation", "Discharge Summary", "Medication Records", "Other"]:
            default_doc_type = scanned["document_type"]
        if scanned["date"]:
            try:
                from dateutil import parser
                parsed_date = parser.parse(scanned["date"])
                default_date = parsed_date.date()
            except Exception:
                pass

    col1, col2 = st.columns(2)
    with col1:
        type_options = ["Labs", "Imaging", "Vital Signs", "Media", "Consultation", "Discharge Summary", "Medication Records", "Other"]
        default_index = type_options.index(default_doc_type) if default_doc_type in type_options else 0
        document_type = st.selectbox(
            "Document Type *",
            options=type_options,
            index=default_index,
            help="Select the type of document (auto-scanned where possible)",
            key="doc_type_select",
        )
    with col2:
        if document_type in DOCUMENT_SUBTYPES and DOCUMENT_SUBTYPES[document_type]:
            if "doc_subtype_select" not in st.session_state:
                default_subtypes = []
                if st.session_state.scanned_info and st.session_state.scanned_info.get("document_subtypes"):
                    scanned_subtypes = st.session_state.scanned_info["document_subtypes"]
                    default_subtypes = [s for s in scanned_subtypes if s in DOCUMENT_SUBTYPES[document_type]]
                st.session_state.doc_subtype_select = default_subtypes
            document_subtypes = st.multiselect(
                "Document Sub-Type(s) *",
                options=DOCUMENT_SUBTYPES[document_type],
                help=f"Select one or more types of {document_type}",
                key="doc_subtype_select",
            )
            document_subtype = ", ".join(document_subtypes) if document_subtypes else ""
        else:
            document_subtype = ""
            st.write("")

    if "results_date_input" not in st.session_state:
        st.session_state.results_date_input = default_date

    results_received_date = st.date_input(
        "Results Received Date *",
        value=st.session_state.results_date_input,
        format="MM/DD/YYYY",
        min_value=date(1900, 1, 1),
        max_value=date.today(),
        help="Date when patient received the results",
        key="results_date_input",
    )

    with st.form("upload_form"):
        upload_notes = st.text_area(
            "Notes (optional)",
            placeholder="Add any relevant notes about these documents...",
            height=100,
        )
        submit_upload = st.form_submit_button("📤 Upload Documents")

    if submit_upload:
        if uploaded_files:
            success_count = 0
            extraction_messages = []
            results_date_str = results_received_date.strftime("%m/%d/%Y")
            for uf in uploaded_files:
                try:
                    result = save_uploaded_file(uf, pid, name, dob, document_type, document_subtype, results_date_str, upload_notes)
                    success_count += 1
                    if isinstance(result, tuple) and len(result) == 2:
                        extraction_messages.append(f"**{uf.name}:** {result[1]}")
                except Exception as e:
                    st.error(f"Error uploading {uf.name}: {e}")

            if success_count > 0:
                st.success(f"Successfully uploaded {success_count} file(s)!")
                if extraction_messages:
                    with st.expander("📊 Extraction Status", expanded=True):
                        for msg in extraction_messages:
                            st.write(msg)
                st.rerun()
        else:
            st.warning("Please select at least one file to upload.")

    st.divider()
    if not patient_files.empty:
        total_files = len(patient_files)
        total_size = patient_files["FileSize"].sum()
        st.info(f"**Summary:** {total_files} document(s) uploaded | Total size: {format_file_size(total_size)}")
    else:
        st.info("No documents uploaded yet. Use the form above to upload hospital admission records.")

    st.divider()
    if st.button("Next → View My Documents", type="primary", use_container_width=True):
        st.session_state.current_page = "view_documents"
        st.rerun()

# ==================== VIEW DOCUMENTS PAGE ====================
elif st.session_state.current_page == "view_documents":
    st.write("### Your Documents")

    if patient_files.empty:
        st.info("No documents uploaded yet. Go to Upload Documents to add files.")
    else:
        tabs = st.tabs(["Labs", "Imaging", "Vital Signs", "Media", "Provider Notes", "Discharge Summary", "Medication Records"])

        # Labs
        with tabs[0]:
            measurements_df = load_measurements()
            patient_labs = measurements_df[(measurements_df["PatientID"] == pid) & (measurements_df["MeasurementGroup"] == "Labs")]
            if not patient_labs.empty:
                st.write("#### 📊 Lab Value Trends")
                unique_measurements = patient_labs["MeasurementName"].unique()
                for measurement in unique_measurements:
                    measurement_data = patient_labs[patient_labs["MeasurementName"] == measurement].copy()
                    if len(measurement_data) >= 1:
                        measurement_data["DateTime"] = pd.to_datetime(measurement_data["DateTime"])
                        measurement_data = measurement_data.sort_values("DateTime")
                        latest_value = measurement_data.iloc[-1]["ValueNumeric"]
                        latest_unit = measurement_data.iloc[-1]["Unit"]
                        latest_date = measurement_data.iloc[-1]["DateTime"].strftime("%m/%d/%Y")
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{measurement}**")
                        with col2:
                            st.metric("Latest", f"{latest_value} {latest_unit}", help=f"As of {latest_date}")
                        if len(measurement_data) > 1:
                            import altair as alt
                            chart = alt.Chart(measurement_data).mark_line(point=True).encode(
                                x=alt.X("DateTime:T", title="Date", axis=alt.Axis(format="%m/%d/%Y")),
                                y=alt.Y("ValueNumeric:Q", title=f"{measurement} ({latest_unit})"),
                                tooltip=[
                                    alt.Tooltip("DateTime:T", title="Date", format="%m/%d/%Y"),
                                    alt.Tooltip("ValueNumeric:Q", title="Value"),
                                    alt.Tooltip("Unit:N", title="Unit"),
                                ],
                            ).properties(height=200)
                            st.altair_chart(chart, use_container_width=True)
                        else:
                            st.caption(f"Single measurement: {latest_value} {latest_unit} on {latest_date}")
                        st.divider()
            labs_docs = patient_files[patient_files["DocumentType"] == "Labs"]
            if not labs_docs.empty:
                st.write("#### 📄 Lab Documents")
                for idx, row in labs_docs.iterrows():
                    display_document_card(row, idx)
            else:
                st.info("No lab documents uploaded yet.")

        # Imaging
        with tabs[1]:
            imaging_docs = patient_files[patient_files["DocumentType"] == "Imaging"]
            if not imaging_docs.empty:
                for idx, row in imaging_docs.iterrows():
                    display_document_card(row, idx)
            else:
                st.info("No imaging documents uploaded yet.")

        # Vital Signs
        with tabs[2]:
            measurements_df = load_measurements()
            patient_vitals = measurements_df[(measurements_df["PatientID"] == pid) & (measurements_df["MeasurementGroup"] == "Vital Signs")]
            if not patient_vitals.empty:
                st.write("#### 📊 Vital Signs Trends")
                unique_measurements = patient_vitals["MeasurementName"].unique()
                for measurement in unique_measurements:
                    measurement_data = patient_vitals[patient_vitals["MeasurementName"] == measurement].copy()
                    if len(measurement_data) >= 1:
                        measurement_data["DateTime"] = pd.to_datetime(measurement_data["DateTime"])
                        measurement_data = measurement_data.sort_values("DateTime")
                        latest_value = measurement_data.iloc[-1]["ValueNumeric"]
                        latest_unit = measurement_data.iloc[-1]["Unit"]
                        latest_date = measurement_data.iloc[-1]["DateTime"].strftime("%m/%d/%Y")
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{measurement}**")
                        with col2:
                            st.metric("Latest", f"{latest_value} {latest_unit}", help=f"As of {latest_date}")
                        if len(measurement_data) > 1:
                            import altair as alt
                            chart = alt.Chart(measurement_data).mark_line(point=True).encode(
                                x=alt.X("DateTime:T", title="Date", axis=alt.Axis(format="%m/%d/%Y")),
                                y=alt.Y("ValueNumeric:Q", title=f"{measurement} ({latest_unit})"),
                                tooltip=[
                                    alt.Tooltip("DateTime:T", title="Date", format="%m/%d/%Y"),
                                    alt.Tooltip("ValueNumeric:Q", title="Value"),
                                    alt.Tooltip("Unit:N", title="Unit"),
                                ],
                            ).properties(height=200)
                            st.altair_chart(chart, use_container_width=True)
                        else:
                            st.caption(f"Single measurement: {latest_value} {latest_unit} on {latest_date}")
                        st.divider()
            vitals_docs = patient_files[patient_files["DocumentType"] == "Vital Signs"]
            if not vitals_docs.empty:
                st.write("#### 📄 Vital Signs Documents")
                for idx, row in vitals_docs.iterrows():
                    display_document_card(row, idx)
            else:
                st.info("No vital signs documents uploaded yet.")

        # Media
        with tabs[3]:
            media_docs = patient_files[patient_files["DocumentType"] == "Media"]
            if not media_docs.empty:
                for idx, row in media_docs.iterrows():
                    display_document_card(row, idx)
            else:
                st.info("No media files uploaded yet.")

        # Provider Notes (Consultation)
        with tabs[4]:
            consultation_docs = patient_files[patient_files["DocumentType"] == "Consultation"]
            if not consultation_docs.empty:
                for idx, row in consultation_docs.iterrows():
                    display_document_card(row, idx)
            else:
                st.info("No provider notes uploaded yet.")

        # Discharge Summary
        with tabs[5]:
            discharge_docs = patient_files[patient_files["DocumentType"] == "Discharge Summary"]
            if not discharge_docs.empty:
                for idx, row in discharge_docs.iterrows():
                    display_document_card(row, idx)
            else:
                st.info("No discharge summaries uploaded yet.")

        # Medication Records
        with tabs[6]:
            medication_docs = patient_files[patient_files["DocumentType"] == "Medication Records"]
            if not medication_docs.empty:
                for idx, row in medication_docs.iterrows():
                    display_document_card(row, idx)
            else:
                st.info("No medication records uploaded yet.")

# --- Optional environment diagnostics ---
with st.sidebar:
    st.divider()
    st.caption("Environment checks")
    st.write(f"PyMuPDF available: {'✅' if _pymupdf_ok() else '❌'}")
    st.write(f"Tesseract OCR available: {'✅' if _tesseract_ok() else '❌'}")
    if not _tesseract_ok():
        st.caption("Install Tesseract if you need OCR (scanned PDFs/images).")
