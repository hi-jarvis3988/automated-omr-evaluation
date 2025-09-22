# OMR Processing Web Application

This web application provides a complete solution for processing, evaluating, and reporting on Optical Mark Recognition (OMR) answer sheets.

## Quick Start (Recommended)

The project includes an automated deployment script that handles all setup steps.

1.  **Clone the Repository:**
    ```bash
    git clone https://huggingface.co/spaces/Jarvis3988/OMR_Processing_Web_Application
    cd OMR_Processing_Web_Application
    ```

    **Verifying the Clone:**
    After running the clone command, you can confirm it was successful if:
    1.  The command completes without any "fatal" error messages.
    2.  A new directory named `OMR_Processing_Web_Application` is created.
    3.  You can navigate into it (`cd OMR_Processing_Web_Application`) and see the project files by running `ls` (Linux/macOS) or `dir` (Windows). You should see files like `app.py`, `deploy_omr_app.py`, etc.

2.  **Run the Deployment Script:**
    This script will create a virtual environment, install dependencies, and generate necessary configuration and sample files.
    ```bash
    python deploy_omr_app.py
    ```

3.  **Run the Application:**
    Use the generated startup scripts.
    -   On Windows: `start_app.bat`
    -   On Linux/macOS: `./start_app.sh`

4.  **Access the Application:**
    Open your browser and go to: `http://localhost:5000`

## Manual Installation

1.  **Create Virtual Environment & Install Dependencies:**
   ```bash
   python -m venv omr_env
   source omr_env/bin/activate  # On Windows: omr_env\Scripts\activate
   pip install -r requirements.txt
   ```

2.  **Project Structure:**
    Ensure your project has the following structure. The `uploads` and `results` directories will be created automatically by the app if they don't exist.
   ```
   omr_webapp/
   ├── app.py                 # Backend Flask application
   ├── requirements.txt       # Python dependencies
   ├── templates/
   │   └── index.html        # Frontend HTML (or serve separately)
   ├── uploads/              # Upload directory (auto-created)
   └── results/              # Results directory (auto-created)
   ```

3.  **Run the Application:**
    ```bash
    python app.py
    ```
### Excel Answer Key Format:

Create an Excel file with the following structure:

**Sheet 1: SET_A**
| Question_No | Subject | Answer |
|-------------|---------|--------|
| 1           | PYTHON  | a      |
| 2           | PYTHON  | c      |
| ...         | ...     | ...    |
| 21          | DATA_ANALYSIS | b |
| ...         | ...     | ...    |

**Sheet 2: SET_B**
| Question_No | Subject | Answer |
|-------------|---------|--------|
| 1           | PYTHON  | b      |
| 2           | PYTHON  | d      |
| ...         | ...     | ...    |

### Alternative Excel Format (Single Sheet):
| Question_No | SET_A_Answer | SET_B_Answer |
|-------------|-------------|-------------|
| 1           | a           | b           |
| 2           | c           | d           |
| ...         | ...         | ...         |

### Features:

✅ **Dashboard with 5 Tabs:**
- Upload Answer Key (Excel processing)
- Student Details Entry
- OMR Sheet Upload with cropping
- Processing & Evaluation
- Results Display & Export

✅ **Image Processing:**
- Grayscale conversion
- Gaussian blur
- Adaptive thresholding
- ROI detection and cropping
- Grid-based bubble detection

✅ **JSON Storage Format:**
```json
{
  "SET_A": {
    "exam_info": {
      "set": "A",
      "subjects": ["PYTHON", "DATA_ANALYSIS", "MySQL", "POWER_BI", "Adv_STATS"],
      "total_questions": 100,
      "format": "multiple_choice"
    },
    "answer_key": {
      "PYTHON": {"1": "a", "2": "c", ...},
      "DATA_ANALYSIS": {"21": "a", ...},
      ...
    }
  }
}
```

✅ **Scoring System:**
- Total Score calculation
- Subject-wise scoring
- Attempted/Unattempted tracking
- Incorrect answer identification

✅ **Export Options:**
- PDF Report generation
- Excel Report export
- Detailed score breakdown

### API Endpoints:

1. **POST /api/process-answer-key**
   - Upload Excel file with answer keys
   - Returns JSON formatted answer key data

2. **POST /api/process-omr**
   - Upload OMR image and paper set
   - Returns extracted student answers in JSON

3. **POST /api/evaluate**
   - Compare student answers with answer key
   - Returns detailed evaluation results

4. **POST /api/export-pdf**
   - Generate PDF report
   - Returns downloadable PDF file

5. **POST /api/export-excel**
   - Generate Excel report
   - Returns downloadable Excel file

### Usage Workflow:

1. **Step 1: Upload Answer Key**
   - Select Excel file with SET_A and SET_B answers
   - System processes and stores in JSON format
   - Validates question numbering and subjects

2. **Step 2: Enter Student Details**
   - Roll Number (required)
   - Name (optional)
   - Exam Date (optional)

3. **Step 3: Upload OMR Sheet**
   - Select paper set (A or B)
   - Upload OMR image (JPG/PNG)
   - Crop to show only bubble section
   - System processes and extracts answers

4. **Step 4: Process & Evaluate**
   - Compare student answers with answer key
   - Calculate scores and statistics
   - Generate evaluation report

5. **Step 5: View & Export Results**
   - View detailed score breakdown
   - Export to PDF or Excel format
   - Subject-wise performance analysis

### Image Processing Pipeline:

```
Original Image → Grayscale → Gaussian Blur → Adaptive Threshold → 
ROI Detection → Grid-based Sampling → Bubble Classification → 
JSON Output
```

### Bubble Detection Logic:

- **Grid Division**: 5 subjects × 20 questions × 4 options
- **Sample Areas**: Extract regions around expected bubble positions
- **Fill Detection**: Analyze pixel intensity and binary fill ratio
- **Classification**: Mark bubbles as filled/unfilled based on thresholds

### Error Handling:

- File format validation
- Image processing error recovery
- Missing data handling
- Graceful degradation for poor image quality

### Performance Considerations:

- Maximum file size: 16MB
- Supported formats: XLSX, XLS, JPG, JPEG, PNG
- Processing time: ~2-5 seconds per OMR sheet
- Concurrent user support via Flask

### Security Features:

- File type validation
- Secure filename handling
- Input sanitization
- CORS enabled for cross-origin requests

### Troubleshooting:

**Common Issues:**

1. **"Could not read image file"**
   - Check image format (JPG/PNG only)
   - Ensure image is not corrupted
   - Try reducing image size

2. **"Answer key processing failed"**
   - Verify Excel file format
   - Check column names (Question_No, Answer)
   - Ensure proper sheet naming (SET_A, SET_B)

3. **"Low accuracy in bubble detection"**
   - Ensure good image quality
   - Crop tightly around bubble area
   - Check lighting and contrast
   - Align image properly

4. **"Missing required data"**
   - Complete all steps in sequence
   - Check that all files uploaded successfully
   - Verify student details are entered

### Advanced Configuration:

**Bubble Detection Parameters:**
```python
# Adjust these in OMRProcessor class
sample_size = min(option_width, question_height) // 3  # Sampling area
binary_fill_threshold = 0.3  # Binary fill ratio threshold  
gray_darkness_threshold = 120  # Grayscale intensity threshold
```

**Grid Parameters:**
```python
subjects = 5           # Number of subject columns
total_questions = 20   # Questions per subject
options = 4           # Options per question (A,B,C,D)
```

### Production Deployment:

For production use, consider:

1. **Use Production WSGI Server:**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

2. **Environment Variables:**
   ```bash
   export FLASK_ENV=production
   export UPLOAD_FOLDER=/path/to/uploads
   export RESULTS_FOLDER=/path/to/results
   ```

3. **Database Integration:**
   - Add SQLite/PostgreSQL for persistent storage
   - Store processing history
   - User management system

4. **Enhanced Security:**
   - Add authentication
   - Rate limiting
   - Input validation
   - Virus scanning for uploads

This web application provides a complete OMR processing solution with high accuracy focus, user-friendly interface, and comprehensive reporting capabilities.