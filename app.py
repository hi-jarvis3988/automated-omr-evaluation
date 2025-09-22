from flask import Flask, request, jsonify, send_file, render_template, render_template
from flask_cors import CORS
import pandas as pd
import cv2
import numpy as np
import json
import os
from datetime import datetime
import tempfile
from werkzeug.utils import secure_filename
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
import xlsxwriter

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'jpg', 'jpeg', 'png'}

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class OMRProcessor:
    def __init__(self):
        self.subjects = ["PYTHON", "DATA_ANALYSIS", "MySQL", "POWER_BI", "Adv_STATS"]
        self.questions_per_subject = 20
        self.total_questions = 100
        
    def process_answer_key_excel(self, excel_file):
        """Process Excel file containing answer keys for sets A & B"""
        try:
            # Read Excel file
            df = pd.read_excel(excel_file, sheet_name=None)  # Read all sheets
            
            answer_key_data = {}
            
            # Process each sheet (assuming sheets are named SET_A, SET_B, etc.)
            for sheet_name, sheet_df in df.items():
                if 'SET' in sheet_name.upper():
                    set_name = sheet_name.upper()
                    if set_name not in answer_key_data:
                        answer_key_data[set_name] = {
                            "exam_info": {
                                "set": set_name.split('_')[1],
                                "subjects": self.subjects,
                                "total_questions": self.total_questions,
                                "format": "multiple_choice"
                            },
                            "answer_key": {}
                        }
                    
                    # Process the sheet data
                    # Assuming columns: Question_No, Subject, Answer
                    for _, row in sheet_df.iterrows():
                        if pd.notna(row.get('Question_No')) and pd.notna(row.get('Answer')):
                            q_no = int(row['Question_No'])
                            answer = str(row['Answer']).lower()
                            
                            # Determine subject based on question number
                            subject = self.get_subject_by_question(q_no)
                            
                            if subject not in answer_key_data[set_name]["answer_key"]:
                                answer_key_data[set_name]["answer_key"][subject] = {}
                            
                            answer_key_data[set_name]["answer_key"][subject][str(q_no)] = answer
            
            # If no SET sheets found, create from main sheet
            if not answer_key_data and len(df) > 0:
                main_sheet = list(df.values())[0]
                answer_key_data = self.create_default_sets(main_sheet)
            
            # Save to JSON file
            json_filename = f"answer_keys_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            json_path = os.path.join(RESULTS_FOLDER, json_filename)
            
            with open(json_path, 'w') as f:
                json.dump(answer_key_data, f, indent=2)
            
            return answer_key_data
            
        except Exception as e:
            raise Exception(f"Error processing answer key: {str(e)}")
    
    def get_subject_by_question(self, q_no):
        """Determine subject based on question number"""
        if 1 <= q_no <= 20:
            return "PYTHON"
        elif 21 <= q_no <= 40:
            return "DATA_ANALYSIS"
        elif 41 <= q_no <= 60:
            return "MySQL"
        elif 61 <= q_no <= 80:
            return "POWER_BI"
        elif 81 <= q_no <= 100:
            return "Adv_STATS"
        else:
            return "UNKNOWN"
    
    def create_default_sets(self, df):
        """Create SET_A and SET_B from main sheet"""
        answer_key_data = {}
        
        for set_name in ['SET_A', 'SET_B']:
            answer_key_data[set_name] = {
                "exam_info": {
                    "set": set_name.split('_')[1],
                    "subjects": self.subjects,
                    "total_questions": self.total_questions,
                    "format": "multiple_choice"
                },
                "answer_key": {}
            }
            
            # Initialize subjects
            for subject in self.subjects:
                answer_key_data[set_name]["answer_key"][subject] = {}
            
            # Process DataFrame
            set_column = f'{set_name}_Answer' if f'{set_name}_Answer' in df.columns else 'Answer'
            
            for _, row in df.iterrows():
                if pd.notna(row.get('Question_No')) and pd.notna(row.get(set_column)):
                    q_no = int(row['Question_No'])
                    answer = str(row[set_column]).lower()
                    subject = self.get_subject_by_question(q_no)
                    
                    answer_key_data[set_name]["answer_key"][subject][str(q_no)] = answer
        
        return answer_key_data
    
    def preprocess_image(self, image):
        """Preprocess the OMR image"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply threshold to get binary image
            binary = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            return gray, binary
            
        except Exception as e:
            raise Exception(f"Error in image preprocessing: {str(e)}")
    
    def detect_omr_region(self, binary_image):
        """Detect and extract the main OMR region"""
        try:
            # Find contours
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find the largest rectangular contour
            largest_area = 0
            best_contour = None
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > largest_area and area > binary_image.shape[0] * binary_image.shape[1] * 0.1:
                    # Check if contour is roughly rectangular
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    if len(approx) >= 4:
                        largest_area = area
                        best_contour = contour
            
            if best_contour is not None:
                x, y, w, h = cv2.boundingRect(best_contour)
                return (x, y, w, h)
            else:
                # Use entire image as fallback
                return (0, 0, binary_image.shape[1], binary_image.shape[0])
                
        except Exception as e:
            # Fallback to entire image
            return (0, 0, binary_image.shape[1], binary_image.shape[0])
    
    def extract_bubbles_grid_based(self, binary_image, gray_image, roi):
        """Extract bubble data using grid-based sampling"""
        try:
            x, y, w, h = roi
            
            # Extract ROI
            binary_roi = binary_image[y:y+h, x:x+w]
            gray_roi = gray_image[y:y+h, x:x+w]
            
            # Grid parameters
            subjects = 5
            total_questions = 20
            options = 4
            
            # Calculate grid dimensions
            subject_width = w // subjects
            question_height = h // total_questions
            option_width = subject_width // options
            
            # Initialize results
            results = np.zeros((subjects, total_questions, options), dtype=int)
            
            # Sample each grid cell
            for subject in range(subjects):
                for question in range(total_questions):
                    for option in range(options):
                        # Calculate cell center
                        cell_x = subject * subject_width + option * option_width + option_width // 2
                        cell_y = question * question_height + question_height // 2
                        
                        # Sample area around the expected bubble location
                        sample_size = min(option_width, question_height) // 3
                        
                        # Extract region
                        x1 = max(0, cell_x - sample_size)
                        x2 = min(w, cell_x + sample_size)
                        y1 = max(0, cell_y - sample_size)
                        y2 = min(h, cell_y + sample_size)
                        
                        if x2 > x1 and y2 > y1:
                            # Check binary region for filled bubble
                            binary_region = binary_roi[y1:y2, x1:x2]
                            gray_region = gray_roi[y1:y2, x1:x2]
                            
                            if binary_region.size > 0 and gray_region.size > 0:
                                # Calculate fill ratio
                                binary_fill_ratio = np.sum(binary_region == 255) / binary_region.size
                                gray_darkness = np.mean(gray_region)
                                
                                # Bubble is filled if high binary fill ratio or low gray intensity
                                is_filled = binary_fill_ratio > 0.3 or gray_darkness < 120
                                results[subject, question, option] = 1 if is_filled else 0
            
            return results
            
        except Exception as e:
            raise Exception(f"Error in bubble extraction: {str(e)}")
    
    def convert_results_to_json(self, results_3d, paper_set):
        """Convert 3D results array to JSON format"""
        try:
            json_data = {
                "paper_set": paper_set,
                "extraction_timestamp": datetime.now().isoformat(),
                "student_answers": {}
            }
            
            for subject_idx, subject_name in enumerate(self.subjects):
                json_data["student_answers"][subject_name] = {}
                
                for question_idx in range(20):
                    # Calculate global question number
                    global_q_num = subject_idx * 20 + question_idx + 1
                    
                    # Find selected options
                    selected_options = []
                    for option_idx in range(4):
                        if results_3d[subject_idx, question_idx, option_idx] == 1:
                            selected_options.append(chr(65 + option_idx))  # A, B, C, D
                    
                    # Store result
                    if len(selected_options) == 1:
                        json_data["student_answers"][subject_name][str(global_q_num)] = selected_options[0].lower()
                    elif len(selected_options) > 1:
                        json_data["student_answers"][subject_name][str(global_q_num)] = "multiple"
                    else:
                        json_data["student_answers"][subject_name][str(global_q_num)] = "not_attempted"
            
            return json_data
            
        except Exception as e:
            raise Exception(f"Error converting results to JSON: {str(e)}")
    
    def process_omr_image(self, image_file, paper_set):
        """Main OMR processing function"""
        try:
            # Read image
            image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
            
            if image is None:
                raise Exception("Could not read image file")
            
            # Preprocess image
            gray, binary = self.preprocess_image(image)
            
            # Detect OMR region
            roi = self.detect_omr_region(binary)
            
            # Extract bubbles
            results_3d = self.extract_bubbles_grid_based(binary, gray, roi)
            
            # Convert to JSON format
            json_data = self.convert_results_to_json(results_3d, paper_set)
            
            # Save results
            json_filename = f"omr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            json_path = os.path.join(RESULTS_FOLDER, json_filename)
            
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            return json_data
            
        except Exception as e:
            raise Exception(f"Error processing OMR image: {str(e)}")

# Initialize OMR processor
omr_processor = OMRProcessor()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/process-answer-key', methods=['POST'])
def process_answer_key():
    try:
        if 'answerKey' not in request.files:
            return jsonify({'error': 'No answer key file provided'}), 400
        
        file = request.files['answerKey']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format'}), 400
        
        # Process the answer key
        answer_key_data = omr_processor.process_answer_key_excel(file)
        
        return jsonify({
            'success': True,
            'message': 'Answer key processed successfully',
            'data': answer_key_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/process-omr', methods=['POST'])
def process_omr():
    try:
        if 'omrImage' not in request.files:
            return jsonify({'error': 'No OMR image provided'}), 400
        
        if 'paperSet' not in request.form:
            return jsonify({'error': 'Paper set not specified'}), 400
        
        image_file = request.files['omrImage']
        paper_set = request.form['paperSet']
        
        if image_file.filename == '' or not allowed_file(image_file.filename):
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Process OMR image
        omr_data = omr_processor.process_omr_image(image_file, paper_set)
        
        return jsonify({
            'success': True,
            'message': 'OMR sheet processed successfully',
            'data': omr_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    try:
        data = request.get_json()
        
        answer_key = data.get('answerKey')
        student_data = data.get('studentData')
        omr_data = data.get('omrData')
        
        if not all([answer_key, student_data, omr_data]):
            return jsonify({'error': 'Missing required data'}), 400
        
        # Get the correct answer key for the paper set
        paper_set = omr_data['paper_set']
        if paper_set not in answer_key:
            return jsonify({'error': f'Answer key for {paper_set} not found'}), 400
        
        correct_answers = answer_key[paper_set]['answer_key']
        student_answers = omr_data['student_answers']
        
        # Calculate scores
        results = {
            'totalScore': 0,
            'attempted': 0,
            'unattempted': 0,
            'incorrect': 0,
            'subjectScores': {}
        }
        
        for subject in omr_processor.subjects:
            subject_score = {'score': 0, 'total': 20, 'attempted': 0, 'correct': 0}
            
            for q_num in range(1, 101):  # Questions 1-100
                q_str = str(q_num)
                subject_range = omr_processor.get_subject_by_question(q_num)
                
                if subject_range == subject:
                    if q_str in correct_answers.get(subject, {}):
                        correct_answer = correct_answers[subject][q_str]
                        student_answer = student_answers.get(subject, {}).get(q_str, 'not_attempted')
                        
                        if student_answer != 'not_attempted':
                            results['attempted'] += 1
                            subject_score['attempted'] += 1
                            
                            if student_answer == correct_answer:
                                results['totalScore'] += 1
                                subject_score['score'] += 1
                                subject_score['correct'] += 1
                            else:
                                results['incorrect'] += 1
                        else:
                            results['unattempted'] += 1
            
            results['subjectScores'][subject] = subject_score
        
        # Save evaluation results
        evaluation_result = {
            'student': student_data,
            'paper_set': paper_set,
            'evaluation_timestamp': datetime.now().isoformat(),
            'results': results,
            'detailed_comparison': {
                'correct_answers': correct_answers,
                'student_answers': student_answers
            }
        }
        
        eval_filename = f"evaluation_{student_data['rollNumber']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        eval_path = os.path.join(RESULTS_FOLDER, eval_filename)
        
        with open(eval_path, 'w') as f:
            json.dump(evaluation_result, f, indent=2)
        
        return jsonify({
            'success': True,
            'message': 'Evaluation completed successfully',
            'data': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export-pdf', methods=['POST'])
def export_pdf():
    try:
        data = request.get_json()
        results = data.get('results')
        student = data.get('student')
        
        if not all([results, student]):
            return jsonify({'error': 'Missing required data'}), 400
        
        # Create PDF
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        
        # Build content
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title = Paragraph("OMR Evaluation Report", styles['Title'])
        story.append(title)
        
        # Student details
        story.append(Paragraph(f"Roll Number: {student.get('rollNumber', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"Name: {student.get('name', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"Exam Date: {student.get('examDate', 'N/A')}", styles['Normal']))
        story.append(Paragraph("<br/><br/>", styles['Normal']))
        
        # Scores table
        score_data = [
            ['Metric', 'Value'],
            ['Total Score', str(results['totalScore'])],
            ['Attempted Questions', str(results['attempted'])],
            ['Unattempted Questions', str(results['unattempted'])],
            ['Incorrect Answers', str(results['incorrect'])]
        ]
        
        score_table = Table(score_data)
        score_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(score_table)
        story.append(Paragraph("<br/><br/>", styles['Normal']))
        
        # Subject-wise scores
        story.append(Paragraph("Subject-wise Scores", styles['Heading2']))
        subject_data = [['Subject', 'Score', 'Total']]
        
        for subject, score_info in results['subjectScores'].items():
            subject_data.append([subject, str(score_info['score']), str(score_info['total'])])
        
        subject_table = Table(subject_data)
        subject_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(subject_table)
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f"OMR_Result_{student['rollNumber']}.pdf",
            mimetype='application/pdf'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export-excel', methods=['POST'])
def export_excel():
    try:
        data = request.get_json()
        results = data.get('results')
        student = data.get('student')
        
        if not all([results, student]):
            return jsonify({'error': 'Missing required data'}), 400
        
        # Create Excel file
        buffer = io.BytesIO()
        workbook = xlsxwriter.Workbook(buffer)
        worksheet = workbook.add_worksheet('OMR Results')
        
        # Formats
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#D7E4BC',
            'border': 1
        })
        
        cell_format = workbook.add_format({
            'border': 1
        })
        
        # Write student details
        row = 0
        worksheet.write(row, 0, 'Student Details', header_format)
        row += 1
        worksheet.write(row, 0, 'Roll Number:', cell_format)
        worksheet.write(row, 1, student.get('rollNumber', 'N/A'), cell_format)
        row += 1
        worksheet.write(row, 0, 'Name:', cell_format)
        worksheet.write(row, 1, student.get('name', 'N/A'), cell_format)
        row += 1
        worksheet.write(row, 0, 'Exam Date:', cell_format)
        worksheet.write(row, 1, student.get('examDate', 'N/A'), cell_format)
        
        row += 3
        
        # Write overall scores
        worksheet.write(row, 0, 'Overall Results', header_format)
        row += 1
        
        score_items = [
            ('Total Score', results['totalScore']),
            ('Attempted Questions', results['attempted']),
            ('Unattempted Questions', results['unattempted']),
            ('Incorrect Answers', results['incorrect'])
        ]
        
        for label, value in score_items:
            worksheet.write(row, 0, label, cell_format)
            worksheet.write(row, 1, value, cell_format)
            row += 1
        
        row += 2
        
        # Write subject-wise scores
        worksheet.write(row, 0, 'Subject-wise Scores', header_format)
        row += 1
        
        # Headers
        worksheet.write(row, 0, 'Subject', header_format)
        worksheet.write(row, 1, 'Score', header_format)
        worksheet.write(row, 2, 'Total', header_format)
        worksheet.write(row, 3, 'Percentage', header_format)
        row += 1
        
        # Subject data
        for subject, score_info in results['subjectScores'].items():
            percentage = (score_info['score'] / score_info['total'] * 100) if score_info['total'] > 0 else 0
            worksheet.write(row, 0, subject, cell_format)
            worksheet.write(row, 1, score_info['score'], cell_format)
            worksheet.write(row, 2, score_info['total'], cell_format)
            worksheet.write(row, 3, f"{percentage:.1f}%", cell_format)
            row += 1
        
        workbook.close()
        buffer.seek(0)
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f"OMR_Result_{student['rollNumber']}.xlsx",
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting OMR Processing Web Application...")
    print("Server will be available at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)