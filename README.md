# spamDetector
An AI-powered web app that detects spam emails using machine learning. The Flask backend (Naïve Bayes + TF-IDF) analyzes email text, while the React frontend provides a user-friendly interface. Deployed on Render, it predicts whether an email is spam (❌) or legit (✅) with confidence scores.

🛠️ Tech Stack

    Backend: Python, Flask, Scikit-Learn, Pandas

    Frontend: React.js, Axios

    Deployment: Render (Backend & Frontend)

    
💻 Setup & Installation

Follow these steps to run the project locally:
1️⃣ Clone the Repository

git clone https://github.com/YOUR_USERNAME/spamDetector.git
cd spamDetector

2️⃣ Create a Virtual Environment (Optional)
python -m venv venv
source venv/bin/activate  # For macOS/Linux
venv\Scripts\activate     # For Windows

3️⃣ Install Dependencies
cd backend
pip install -r requirements.txt


4️⃣ Run the Flask Server

python app.py

    
