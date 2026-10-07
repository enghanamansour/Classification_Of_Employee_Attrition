"""
Interactive Web Server for Employee Attrition Classification Model
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys
import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import joblib

# Import prediction engine
from predict import predict_single, load_model

PORT = 8080
MODEL_DATA = None

class AttritionRequestHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ['/', '/index.html']:
            self.serve_file('index.html', 'text/html; charset=utf-8')
        elif path == '/api/metadata':
            if os.path.exists('model_metadata.json'):
                with open('model_metadata.json', 'r', encoding='utf-8') as f:
                    data = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(data.encode('utf-8'))
            else:
                self.send_error(404, "model_metadata.json not found")
        elif path == '/api/sample-employees':
            samples = {
                'junior_burnout': {
                    'name': 'Alex Rivera (Junior Sales Rep - High Burnout Risk)',
                    'data': {
                        'Age': 24, 'BusinessTravel': 'Travel_Frequently', 'DailyRate': 420,
                        'Department': 'Sales', 'DistanceFromHome': 24, 'Education': 2,
                        'EducationField': 'Marketing', 'EnvironmentSatisfaction': 1, 'Gender': 'Female',
                        'HourlyRate': 45, 'JobInvolvement': 1, 'JobLevel': 1, 'JobRole': 'Sales Representative',
                        'JobSatisfaction': 1, 'MaritalStatus': 'Single', 'MonthlyIncome': 2200,
                        'MonthlyRate': 12000, 'NumCompaniesWorked': 3, 'OverTime': 'Yes',
                        'PercentSalaryHike': 11, 'PerformanceRating': 3, 'RelationshipSatisfaction': 2,
                        'StockOptionLevel': 0, 'TotalWorkingYears': 2, 'TrainingTimesLastYear': 1,
                        'WorkLifeBalance': 1, 'YearsAtCompany': 1, 'YearsInCurrentRole': 0,
                        'YearsSinceLastPromotion': 0, 'YearsWithCurrManager': 0
                    }
                },
                'mid_stagnant': {
                    'name': 'David Chen (Lab Tech - Career Stagnation)',
                    'data': {
                        'Age': 34, 'BusinessTravel': 'Travel_Rarely', 'DailyRate': 750,
                        'Department': 'Research & Development', 'DistanceFromHome': 18, 'Education': 3,
                        'EducationField': 'Medical', 'EnvironmentSatisfaction': 2, 'Gender': 'Male',
                        'HourlyRate': 60, 'JobInvolvement': 2, 'JobLevel': 2, 'JobRole': 'Laboratory Technician',
                        'JobSatisfaction': 2, 'MaritalStatus': 'Single', 'MonthlyIncome': 3800,
                        'MonthlyRate': 16000, 'NumCompaniesWorked': 4, 'OverTime': 'Yes',
                        'PercentSalaryHike': 12, 'PerformanceRating': 3, 'RelationshipSatisfaction': 2,
                        'StockOptionLevel': 0, 'TotalWorkingYears': 8, 'TrainingTimesLastYear': 2,
                        'WorkLifeBalance': 2, 'YearsAtCompany': 6, 'YearsInCurrentRole': 5,
                        'YearsSinceLastPromotion': 5, 'YearsWithCurrManager': 1
                    }
                },
                'senior_stable': {
                    'name': 'Dr. Eleanor Vance (Research Director - Highly Retained)',
                    'data': {
                        'Age': 49, 'BusinessTravel': 'Non-Travel', 'DailyRate': 1350,
                        'Department': 'Research & Development', 'DistanceFromHome': 3, 'Education': 4,
                        'EducationField': 'Life Sciences', 'EnvironmentSatisfaction': 4, 'Gender': 'Female',
                        'HourlyRate': 88, 'JobInvolvement': 4, 'JobLevel': 4, 'JobRole': 'Research Director',
                        'JobSatisfaction': 4, 'MaritalStatus': 'Married', 'MonthlyIncome': 17500,
                        'MonthlyRate': 21000, 'NumCompaniesWorked': 2, 'OverTime': 'No',
                        'PercentSalaryHike': 19, 'PerformanceRating': 4, 'RelationshipSatisfaction': 4,
                        'StockOptionLevel': 2, 'TotalWorkingYears': 26, 'TrainingTimesLastYear': 3,
                        'WorkLifeBalance': 4, 'YearsAtCompany': 19, 'YearsInCurrentRole': 12,
                        'YearsSinceLastPromotion': 1, 'YearsWithCurrManager': 11
                    }
                },
                'average_employee': {
                    'name': 'Jordan Taylor (Standard Sales Executive)',
                    'data': {
                        'Age': 36, 'BusinessTravel': 'Travel_Rarely', 'DailyRate': 850,
                        'Department': 'Sales', 'DistanceFromHome': 8, 'Education': 3,
                        'EducationField': 'Life Sciences', 'EnvironmentSatisfaction': 3, 'Gender': 'Male',
                        'HourlyRate': 68, 'JobInvolvement': 3, 'JobLevel': 2, 'JobRole': 'Sales Executive',
                        'JobSatisfaction': 3, 'MaritalStatus': 'Married', 'MonthlyIncome': 5800,
                        'MonthlyRate': 14000, 'NumCompaniesWorked': 2, 'OverTime': 'No',
                        'PercentSalaryHike': 14, 'PerformanceRating': 3, 'RelationshipSatisfaction': 3,
                        'StockOptionLevel': 1, 'TotalWorkingYears': 10, 'TrainingTimesLastYear': 3,
                        'WorkLifeBalance': 3, 'YearsAtCompany': 6, 'YearsInCurrentRole': 4,
                        'YearsSinceLastPromotion': 1, 'YearsWithCurrManager': 4
                    }
                }
            }
            res_str = json.dumps(samples)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(res_str.encode('utf-8'))
        elif path.startswith('/static/') or path == '/evaluation_plots.png':
            clean_path = path.lstrip('/')
            if os.path.exists(clean_path):
                ext = os.path.splitext(clean_path)[1].lower()
                mime = 'image/png' if ext == '.png' else 'text/css' if ext == '.css' else 'application/javascript' if ext == '.js' else 'application/octet-stream'
                self.serve_file(clean_path, mime)
            else:
                self.send_error(404, f"File {clean_path} not found")
        else:
            self.send_error(404, "Page not found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == '/api/predict':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            try:
                emp_dict = json.loads(post_data.decode('utf-8'))
                global MODEL_DATA
                if MODEL_DATA is None:
                    MODEL_DATA = load_model()
                result = predict_single(emp_dict, MODEL_DATA)
                
                resp_bytes = json.dumps(result).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(resp_bytes)
            except Exception as e:
                err_resp = json.dumps({'error': str(e)}).encode('utf-8')
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(err_resp)
        else:
            self.send_error(404, "Endpoint not found")

    def serve_file(self, filename, content_type):
        if not os.path.exists(filename):
            self.send_error(404, f"File {filename} not found")
            return
        with open(filename, 'rb') as f:
            content = f.read()
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format, *args):
        # Keep logs clean
        sys.stderr.write("%s - - [%s] %s\n" %
                         (self.address_string(),
                          self.log_date_time_string(),
                          format % args))

def run_server(port=PORT):
    global MODEL_DATA
    print(f"[*] Preloading trained model pipeline...")
    MODEL_DATA = load_model()
    server_address = ('', port)
    httpd = HTTPServer(server_address, AttritionRequestHandler)
    print(f"[+] HR Attrition Classification Server running at http://localhost:{port}/")
    print(f"[+] Press Ctrl+C to terminate.")
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()
