"""
Smart Campus Chatbot Engine — v3
────────────────────────────────
NLP Stack:
  • spaCy  — tokenisation, lemmatisation, NER
  • NLTK   — stopword removal, PorterStemmer
  • TF-IDF + Cosine Similarity (scikit-learn)
  • Naive-Bayes Intent Classifier (scikit-learn) — trained at startup
  • Fuzzy keyword fallback

All original responses are preserved.
"""

import re, numpy as np

# ── NLP imports ────────────────────────────────────────────
import nltk
from nltk.corpus   import stopwords
from nltk.stem     import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise        import cosine_similarity
from sklearn.naive_bayes             import MultinomialNB
from sklearn.pipeline                import Pipeline
from sklearn.preprocessing           import LabelEncoder

# Download NLTK data silently
for pkg in ("stopwords", "punkt", "punkt_tab"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

# Optional spaCy
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    SPACY_OK = True
except Exception:
    SPACY_OK = False

_stemmer    = PorterStemmer()
_stop_words = set()
try:
    _stop_words = set(stopwords.words("english"))
except Exception:
    pass

# ─── Training Corpus ─────────────────────────────────────
CORPUS = [
    # Greetings
    ("hello hi hey good morning good evening welcome",
     "👋 Hello! Welcome to Smart Campus Assistant! How can I help you today?<br><br>"
     "Choose an option:<br>"
     "<div class=\"chips\">"
     "<button class=\"chip\" onclick=\"send('1')\">Student Enquiry</button>"
     "<button class=\"chip\" onclick=\"send('2')\">Faculty Enquiry</button>"
     "<button class=\"chip\" onclick=\"send('3')\">Parent Enquiry</button>"
     "<button class=\"chip\" onclick=\"send('4')\">Visitor Enquiry</button>"
     "</div>",
     "greet"),
    ("how are you how do you do what's up",
     "😊 I'm doing great! Ready to assist you with all your Smart Campus queries. What do you need?",
     "greet"),

    # Menu navigation
    ("1 student enquiry i am a student student help",
     "🎓 <b>Student Enquiry</b><br>What would you like to know?<br>"
     "<div class=\"chips\">"
     "<button class=\"chip\" onclick=\"send('1.1')\">Admission process</button>"
     "<button class=\"chip\" onclick=\"send('1.2')\">Courses offered</button>"
     "<button class=\"chip\" onclick=\"send('1.3')\">Exam schedule</button>"
     "<button class=\"chip\" onclick=\"send('1.4')\">Scholarships</button>"
     "<button class=\"chip\" onclick=\"send('1.5')\">Hostel facilities</button>"
     "<button class=\"chip\" onclick=\"send('1.6')\">Library</button>"
     "<button class=\"chip\" onclick=\"send('1.7')\">Placement</button>"
     "</div>",
     "menu"),
    ("2 faculty enquiry i am a teacher professor faculty",
     "👨‍🏫 <b>Faculty Enquiry</b><br>What would you like to know?<br>"
     "<div class=\"chips\">"
     "<button class=\"chip\" onclick=\"send('2.1')\">Faculty directory</button>"
     "<button class=\"chip\" onclick=\"send('2.2')\">Department contacts</button>"
     "<button class=\"chip\" onclick=\"send('2.3')\">Academic calendar</button>"
     "<button class=\"chip\" onclick=\"send('2.4')\">Research facilities</button>"
     "</div>",
     "menu"),
    ("3 parent enquiry i am a parent guardian mother father",
     "👨‍👩‍👧 <b>Parent Enquiry</b><br>What would you like to know?<br>"
     "<div class=\"chips\">"
     "<button class=\"chip\" onclick=\"send('3.1')\">Student progress</button>"
     "<button class=\"chip\" onclick=\"send('3.2')\">Fee structure</button>"
     "<button class=\"chip\" onclick=\"send('3.3')\">Hostel info</button>"
     "<button class=\"chip\" onclick=\"send('3.4')\">Grievance</button>"
     "</div>",
     "menu"),
    ("4 visitor enquiry i am visiting campus tour",
     "🏫 <b>Visitor Enquiry</b><br>What would you like to know?<br>"
     "<div class=\"chips\">"
     "<button class=\"chip\" onclick=\"send('4.1')\">Campus location</button>"
     "<button class=\"chip\" onclick=\"send('4.2')\">Courses overview</button>"
     "<button class=\"chip\" onclick=\"send('4.3')\">Events & seminars</button>"
     "<button class=\"chip\" onclick=\"send('4.4')\">Contact</button>"
     "</div>",
     "menu"),

    # Admission
    ("admission apply how to join enroll application process joining 1.1",
     "📄 <b>Admission Process</b><br><br>"
     "1. Fill the online application at our website<br>"
     "2. Upload documents: 10th, 12th marksheets, ID proof<br>"
     "3. Pay application fee:&nbsp;₹500<br>"
     "4. Entrance test / merit-based selection<br>"
     "5. Attend counselling and confirm seat<br><br>"
     "📧<a href='https://mail.google.com/mail/?view=cm&to=admission@jec.ac.in'>admission@jec.ac.in</a><br>"
     "📧<a href='https://mail.google.com/mail/?view=cm&to=info@jec.ac.in'>info@jec.ac.in</a><br>"
     "📧<a href='https://mail.google.com/mail/?view=cm&to=principal@jec.ac.in'>principal@jec.ac.in</a><br>"
     "📞 <span class=\"number\">044-26300982</span>, <span class=\"number\">044-26341264</span>, <span class=\"number\">044-26390041</span>",
     "admission"),
    ("documents required admission what to bring certificates eligibility",
     "📋 <b>Documents Required for Admission</b><br><br>"
     "• 10th Marksheet (original + 2 copies)<br>"
     "• 12th Marksheet (original + 2 copies)<br>"
     "• Transfer Certificate (TC)<br>"
     "• Community Certificate (SC/ST/OBC)<br>"
     "• Passport-size photos (6 copies)<br>"
     "• Aadhar Card / ID proof<br>"
     "• Migration Certificate (if applicable)",
     "admission"),

    # Courses
    ("courses offered programs degrees branches departments 1.2 4.2 what courses",
     "📚 <b>Courses Offered</b><br><br>"
     "<b>Undergraduate (4 years – BE):</b><br>"
     "• Computer Science & Engineering (CSE)<br>"
     "• Electronics & Communication (ECE)<br>"
     "• Mechanical Engineering (MECH)<br>"
     "• Civil Engineering (CIVIL)<br>"
     "• Electrical & Electronics (EEE)<br><br>"
     "<b>Postgraduate (2 years):</b><br>"
     "• M.E. Computer Science<br>"
     "• MBA – Business Administration<br><br>"
     "Total intake: 480 students/year | Anna University Affiliated",
     "courses"),

    # Fees
    ("fee fees tuition cost payment how much fee structure 3.2",
     "💰 <b>Fee Structure (per year)</b><br><br>"
     "•&nbsp;B.E.&nbsp;Engineering:&nbsp;₹85,000&nbsp;–&nbsp;₹1,10,000<br>"
     "•&nbsp;M.E.:&nbsp;₹60,000/year<br>"
     "•&nbsp;MBA:&nbsp;₹70,000/year<br><br>"
     "<b>Payment modes:</b> Online / DD / Net Banking<br>"
     "<b>Installments:</b> 2 per year allowed<br>"
     "🏦 Scholarships available for merit and SC/ST/OBC students",
     "fees"),

    # Scholarship
    ("scholarship financial aid merit sc st obc concession free seat 1.4",
     "🏅 <b>Scholarships Available</b><br><br>"
     "• 7.5% Govt School Quota<br>"
     "• Tamil Nadu Govt Scholarship (SC/ST/OBC) – up to 100%<br>"
     "• AICTE Pragati Scholarship for Girls<br>"
     "• National Merit Scholarship<br>"
     "• Management Merit (top 10 students) – 25% waiver<br>"
     "• Sports Quota – up to 50% concession<br>"
     "• First-generation graduate incentive<br>"
     "• For more information: <a href='https://scholarships.gov.in/'>https://scholarships.gov.in/</a><br>"
     "Apply within 30 days of joining via the scholarship portal.",
     "scholarship"),

    # Hostel
    ("hostel accommodation room boarding stay 1.5 3.3 dormitory",
     "🏠 <b>Hostel Facilities</b><br><br>"
     "• Separate boys' & girls' hostels<br>"
     "• 24/7 CCTV security & warden<br>"
     "• Wi-Fi enabled rooms<br>"
     "• Mess: veg & non-veg options<br>"
     "• Gym, recreation room, indoor games<br><br>"
     "💰&nbsp;Fee:&nbsp;₹60,000/year&nbsp;(room&nbsp;+&nbsp;mess)<br>"
     "📞&nbsp;Boys Warden:&nbsp;<span class=\"number\">044-9876554321</span><br>"
     "📞&nbsp;Girls Warden:&nbsp;<span class=\"number\">044-9876556748</span>",
     "hostel"),

    # Exams
    ("exam examination schedule timetable test result marks 1.3 hall ticket",
     "📝 <b>Examination Schedule</b><br><br>"
     "• Internal Assessment: Every 45 days (3 per semester)<br>"
     "• Semester End Exams: November & April<br>"
     "• Results: Within 30 days after exams<br>"
     "• Hall tickets: Issued 2 weeks before exam<br><br>"
     "Anna University affiliated — follow Anna University calendar.<br>"
     "📋&nbsp;Check&nbsp;results&nbsp;at:&nbsp;<a href='https://coe.annauniv.edu/home/index.php'>click for results</a>",
     "exams"),

    # Placement
    ("placement job career recruit company package salary hire offer 1.7",
     "📁 <b>Placement Cell</b><br><br>"
     "🎯 <b>Placement Highlights (2023–24)</b><br>"
     "• Placement Rate: 92%<br>"
     "• Average Package: ₹4.8 LPA<br>"
     "• Highest Package: ₹18 LPA (Amazon)<br><br>"
     "🏢 <b>Top Recruiters</b><br>"
     "TCS, Infosys, Wipro, Cognizant (CTS), HCL, Zoho, Amazon, Capgemini, Accenture<br><br>"
     "📊 <b>Training & Support</b><br>"
     "• Aptitude & Technical Training Programs<br>"
     "• Mock Interviews & Group Discussions<br>"
     "• Resume Building Workshops<br>"
     "• Soft Skills & Communication Training<br><br>"
     "🚀 <b>Career Opportunities</b><br>"
     "Students are placed in IT, Core Engineering, Banking, and Startup sectors.<br><br>"
     "🌐 For detailed placement reports, company-wise offers, and eligibility criteria, visit the official placement portal.<br><br>"
     "📞 Contact Placement Cell: <span class=\"number\">044-23456800</span><br><br>"
     "<a href='https://jec.ac.in/placement-record/' target='_blank' class='placement-link'>🔗 View Detailed Placement Report</a>",
     "placement"),

    # Library
    ("library book digital e-book journal read 1.6",
     "📚 <b>Library</b><br><br>"
     "• 60,000+ books & reference materials<br>"
     "• Digital access: NPTEL, IEEE Xplore, Springer<br>"
     "• Open: 8 AM – 7.30 PM (Mon–Sat)<br>"
     "• Issue: 3 books per student, 14-day period<br>"
     "•&nbsp;E-library&nbsp;portal:&nbsp;elibrary<a href='https://jec.ac.in/library/'> click here</a><br><br>",
     "library"),

    # Faculty
    ("faculty directory professor hod head of department 2.1",
     "👨‍🏫 <b>Faculty Directory</b><br><br>""•&nbsp;CSE&nbsp;HOD:&nbsp;Dr.&nbsp;R.&nbsp;Suresh&nbsp;—&nbsp;<a href='https://mail.google.com/mail/?view=cm&to=csehod@jec.ac.in'>csehod@jec.ac.in</a><br>"
     "•&nbsp;ECE&nbsp;HOD:&nbsp;Dr.&nbsp;P.&nbsp;Meena&nbsp;—&nbsp;<a href='https://mail.google.com/mail/?view=cm&to=ecehod@jec.ac.in'>ecehod@jec.ac.in</a><br>"
     "•&nbsp;MECH&nbsp;HOD:&nbsp;Dr.&nbsp;K.&nbsp;Rajan&nbsp;—&nbsp;<a href='https://mail.google.com/mail/?view=cm&to=mechhod@jec.ac.in'>mechhod@jec.ac.in</a><br>"
     "•&nbsp;CIVIL&nbsp;HOD:&nbsp;Dr.&nbsp;S.&nbsp;Priya&nbsp;—&nbsp;<a href='https://mail.google.com/mail/?view=cm&to=civilhod@jec.ac.in'>civilhod@jec.ac.in</a><br><br>"
     "Total: 120+ faculty | PhD holders: 45%",
     "faculty"),

    # Calendar
    ("academic calendar semester holiday schedule 2.3",
     "📅 <b>Academic Calendar 2024–25</b><br><br>"
     "• Odd Sem: July – Nov 2024<br>"
     "• Odd Sem Exams: Nov 18 – Dec 6, 2024<br>"
     "• Even Sem: Jan – April 2025<br>"
     "• Even Sem Exams: April 21 – May 9, 2025<br><br>"
     "Holidays follow TN Government notification.",
     "calendar"),

    # Research
    ("research lab laboratory project publication 2.4",
     "🔬 <b>Research Facilities</b><br><br>"
     "• AI & Machine Learning Lab<br>"
     "• IoT Research Centre<br>"
     "• Robotics & Automation Lab<br>"
     "• VLSI Design Lab<br>"
     "• Centre for Green Energy Research<br><br>"
     "Publications: 200+ papers (2023–24)<a href=https://jec.ac.in/research-papers-published/'> click here</a><br>"
     "Research labs: <a href=https://jec.ac.in/research-labs/> click here</a><br>"
     "Funded by: DST, AICTE, TNSCST",
     "research"),

    # Contact
    ("contact phone email address location where campus 4.1 4.4",
     "📞 <b>Contact Information</b><br><br>"
     "🏫 Smart Campus College of Engineering<br>"
     "📍 123 Campus Road, Chennai – 600 001, Tamil Nadu<br><br>"
     "📧<a href='https://mail.google.com/mail/?view=cm&to=admission@jec.ac.in'>admission@jec.ac.in</a><br>"
     "📧<a href='https://mail.google.com/mail/?view=cm&to=info@jec.ac.in'>info@jec.ac.in</a><br>"
     "📧<a href='https://mail.google.com/mail/?view=cm&to=principal@jec.ac.in'>principal@jec.ac.in</a><br>"
     "📞 <span class=\"number\">044-26300982</span>, <span class=\"number\">044-26341264</span>, <span class=\"number\">044-26390041</span><br>"
     "🌐&nbsp;<a href='https://jec.ac.in/'>click here</a><br><br>"
     "⏰ Office: Mon–Fri, 9 AM – 5 PM",
     "contact"),

    # Events
    ("events seminar workshop fest symposium conference 4.3",
     "🎉 <b>Upcoming Events</b><br><br>"
     "• Tech Symposium 'InnovateMind 2025' — Feb 14–15<br>"
     "• National Hackathon — March 1<br>"
     "• Annual Sports Meet — Jan 20–22<br>"
     "• Cultural Fest 'Utsav 2025' — April 5–6<br><br>"
     "Register via student portal or events committee.",
     "events"),

    # Parent/Progress
    ("student progress attendance marks report performance 3.1",
     "📊 <b>Student Progress (Parent Portal)</b><br><br>"
     "🌐&nbsp;parents.smartcampus.edu.in<br><br>"
     "• Live attendance (updated weekly)<br>"
     "• Internal assessment marks<br>"
     "• Arrear / backlog status<br>"
     "• Fee payment history<br><br>"
     "Login credentials given during orientation.",
     "progress"),

    # Grievance
    ("grievance complaint issue problem redressal 3.4",
     "📋 <b>Grievance Redressal</b><br><br>"
     "Step 1: Raise with class tutor<br>"
     "Step 2: Escalate to HOD (if unresolved in 3 days)<br>"
     "Step 3: Principal's office<br><br>"
     "📧&nbsp;<a href='https://mail.google.com/mail/?view=cm&to=info@jec.ac.in'>info@jec.ac.in</a><br>"
     "📧<a href='https://mail.google.com/mail/?view=cm&to=principal@jec.ac.in'>principal@jec.ac.in</a><br>"
     "📞&nbsp;Helpline:&nbsp;<span class=\"number\">044-23456790</span>",
     "grievance"),

    # Department contacts
    ("department contact office number 2.2",
     "📞 <b>Department Contacts</b><br><br>"
     "•&nbsp;CSE:&nbsp;<span class=\"number\">044-23456791</span><br>"
     "•&nbsp;ECE:&nbsp;<span class=\"number\">044-23456792</span><br>"
     "•&nbsp;MECH:&nbsp;<span class=\"number\">044-23456793</span><br>"
     "•&nbsp;CIVIL:&nbsp;<span class=\"number\">044-23456794</span><br>"
     "•&nbsp;MBA/MCA:&nbsp;<span class=\"number\">044-23456795</span><br>"
     "•&nbsp;Library:&nbsp;<span class=\"number\">044-23456796</span>",
     "contacts"),

    # Thanks / Bye
    ("thank thanks thank you helpful great good excellent",
     "😊 You're welcome! Is there anything else I can help you with?",
     "thanks"),
    ("bye goodbye exit quit see you that's all done",
     "👋 Thank you for visiting Smart Campus! Have a great day. Come back anytime! 🎓",
     "bye"),
]

# ─── Multilingual responses ───────────────────────────────
MULTILINGUAL = {
    "ta": {
        "greet": "👋 வணக்கம்! Smart Campus உதவியாளரிடம் வரவேற்கிறோம்!\n\n"
                 "1️⃣ மாணவர் விசாரணை\n2️⃣ ஆசிரியர் விசாரணை\n"
                 "3️⃣ பெற்றோர் விசாரணை\n4️⃣ பார்வையாளர் விசாரணை",
        "admission": "📄 <b>சேர்க்கை செயல்முறை</b><br><br>"
                     "1. ஆன்லைன் விண்ணப்பத்தை நிரப்பவும்<br>"
                     "2. ஆவணங்களை பதிவேற்றவும்<br>"
                     "3. விண்ணப்ப கட்டணம்: ₹500<br>"
                     "4. நுழைவுத் தேர்வு / தகுதி அடிப்படையில் தேர்வு<br>"
                     "5. கலந்தாய்வு மற்றும் இடத்தை உறுதிப்படுத்தவும்<br><br>"
                     "📅 சேர்க்கை: ஜூன் – ஆகஸ்ட்",
        "fees": "💰 <b>கட்டண அமைப்பு (ஆண்டுக்கு)</b><br><br>"
                "• B.E. பொறியியல்: ₹85,000 – ₹1,10,000<br>"
                "• M.E.: ₹60,000<br>"
                "• MBA: ₹70,000<br><br>"
                "SC/ST/OBC மாணவர்களுக்கு உதவித்தொகை உண்டு.",
        "contact": "📞 <b>தொடர்பு தகவல்</b><br><br>"
                   "🏫 Smart Campus பொறியியல் கல்லூரி<br>"
                   "📍 123 கேம்பஸ் சாலை, சென்னை – 600 001<br>"
                   "📞 <span class=\"number\">044-23456789</span><br>"
                   "📧<a href='https://mail.google.com/mail/?view=cm&to=info@jec.ac.in>info@jec.ac.in</a>",
        "default": "🙏 மன்னிக்கவும், என்னால் புரிந்துகொள்ள முடியவில்லை. தயவுசெய்து ஆங்கிலத்தில் கேட்கவும்.",
    },
    "hi": {
        "greet": "👋 नमस्ते! Smart Campus सहायक में आपका स्वागत है!\n\n"
                 "1️⃣ छात्र पूछताछ\n2️⃣ शिक्षक पूछताछ\n"
                 "3️⃣ अभिभावक पूछताछ\n4️⃣ आगंतुक पूछताछ",
        "admission": "📄 <b>प्रवेश प्रक्रिया</b><br><br>"
                     "1. ऑनलाइन आवेदन भरें<br>"
                     "2. दस्तावेज़ अपलोड करें<br>"
                     "3. आवेदन शुल्क: ₹500<br>"
                     "4. प्रवेश परीक्षा / मेरिट-आधारित चयन<br>"
                     "5. काउंसलिंग में भाग लें<br><br>"
                     "📅 प्रवेश: जून – अगस्त",
        "fees": "💰 <b>शुल्क संरचना (प्रति वर्ष)</b><br><br>"
                "• B.E. इंजीनियरिंग: ₹85,000 – ₹1,10,000<br>"
                "• M.E.: ₹60,000<br>"
                "• MBA: ₹70,000<br><br>"
                "SC/ST/OBC छात्रों के लिए छात्रवृत्ति उपलब्ध है।",
        "contact": "📞 <b>संपर्क जानकारी</b><br><br>"
                   "🏫 Smart Campus इंजीनियरिंग कॉलेज<br>"
                   "📍 123 कैंपस रोड, चेन्नई – 600 001<br>"
                   "📞 <span class=\"number\">044-23456789</span><br>"
                   "📧 <a href='https://mail.google.com/mail/?view=cm&to=info@jec.ac.in>info@jec.ac.in</a>",
        "default": "🙏 माफ़ कीजिए, मैं समझ नहीं सका। कृपया अंग्रेजी में पूछें।",
    },
}

FALLBACK = "Please enter a valid option like 1, 1.1, 2.3, etc."

# ─── NLP Preprocessing ───────────────────────────────────
def _preprocess(text: str) -> str:
    """Tokenise, remove stopwords, stem using NLTK; lemmatise using spaCy if available."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)

    if SPACY_OK:
        doc = _nlp(text)
        tokens = [
            token.lemma_ for token in doc
            if not token.is_stop and not token.is_punct and token.text.strip()
        ]
    else:
        tokens = text.split()
        tokens = [t for t in tokens if t not in _stop_words]
        tokens = [_stemmer.stem(t) for t in tokens]

    return " ".join(tokens)

# ─── Build TF-IDF model ───────────────────────────────────
_questions     = [c[0] for c in CORPUS]
_responses     = [c[1] for c in CORPUS]
_tags          = [c[2] for c in CORPUS]
_processed_qs  = [_preprocess(q) for q in _questions]

_vectorizer   = TfidfVectorizer(ngram_range=(1, 2))
_tfidf_matrix = _vectorizer.fit_transform(_processed_qs)

# ─── Naive-Bayes Intent Classifier ───────────────────────
# Augment training data for each intent tag
_NB_SAMPLES = {
    "greet":      ["hello", "hi there", "good morning", "hey bot", "welcome", "how are you"],
    "admission":  ["how to apply", "admission process", "how do I join", "enrollment", "apply now",
                   "application form", "when does admission open", "entrance test"],
    "fees":       ["what is the fee", "how much does it cost", "tuition fee", "fee structure",
                   "payment options", "how to pay fees", "fee per year"],
    "courses":    ["what courses are available", "list of programs", "engineering branches",
                   "MBA available", "which departments", "degree programs"],
    "hostel":     ["hostel available", "accommodation details", "room and board", "stay on campus",
                   "hostel fees", "boys hostel", "girls hostel"],
    "exams":      ["when are exams", "exam schedule", "timetable", "hall ticket", "results",
                   "internal assessment", "semester exam"],
    "placement":  ["placement stats", "job offers", "campus recruitment", "top recruiters",
                   "salary package", "career cell", "hiring companies"],
    "contact":    ["how to contact", "phone number", "email address", "campus location",
                   "where is the college", "office hours"],
    "scholarship":["scholarship available", "financial aid", "fee waiver", "merit scholarship",
                   "SC ST concession", "free education"],
    "library":    ["library timings", "how to borrow books", "e-library", "digital books",
                   "library access", "book issue"],
    "faculty":    ["faculty list", "professor details", "HOD contact", "department head",
                   "staff directory"],
    "events":     ["upcoming events", "tech fest", "hackathon", "sports meet", "cultural fest",
                   "seminars", "workshop"],
    "grievance":  ["how to complain", "raise a complaint", "grievance cell", "issue redressal"],
    "thanks":     ["thank you", "thanks", "that was helpful", "great", "awesome"],
    "bye":        ["bye", "goodbye", "see you", "exit", "quit", "that's all"],
    "menu":       ["student", "faculty", "parent", "visitor", "main menu", "options"],
    "progress":   ["student marks", "attendance", "academic performance", "parent portal"],
    "research":   ["research facilities", "lab", "publication", "funded projects", "IoT lab"],
    "calendar":   ["academic calendar", "semester dates", "holiday list", "exam dates"],
    "contacts":   ["department phone", "office number", "CSE department", "ECE department"],
}

_nb_texts, _nb_labels = [], []
for intent, samples in _NB_SAMPLES.items():
    for s in samples:
        _nb_texts.append(_preprocess(s))
        _nb_labels.append(intent)

_le = LabelEncoder()
_nb_y = _le.fit_transform(_nb_labels)

_nb_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("clf",   MultinomialNB(alpha=0.5)),
])
_nb_pipeline.fit(_nb_texts, _nb_y)

def _predict_intent(text: str) -> tuple[str, float]:
    """Return (intent_label, confidence) using Naive-Bayes."""
    processed = _preprocess(text)
    proba = _nb_pipeline.predict_proba([processed])[0]
    best_idx   = int(np.argmax(proba))
    confidence = float(proba[best_idx])
    intent     = _le.inverse_transform([best_idx])[0]
    return intent, confidence



# ─── Named Entity Recognition (spaCy) ────────────────────
def _extract_entities(text: str) -> dict:
    """Extract named entities – useful for logging & analytics."""
    if not SPACY_OK:
        return {}
    doc = _nlp(text)
    return {ent.label_: ent.text for ent in doc.ents}

# ─── Main response function ────────────────────────────────
def get_response(user_input: str) -> str:
    text = user_input.strip()
    if not text:
        return FALLBACK

    text_lower = text.lower()
    
    # Exact role match
    if text_lower in ["student"]:
        return _responses[2]
    elif text_lower in ["admission", "admissions", "admission process"]:
        return _responses[6]
    elif text_lower in ["courses", "course", "courses offered", "courses overview"]:
        return _responses[8]
    elif text_lower in ["exams", "exam", "schedule", "exam schedule"]:
        return _responses[12]
    elif text_lower in ["scholarships", "scholarship"]:
        return _responses[10]
    elif text_lower in ["hostel", "hostels", "hostel facilities", "hostel info"]:
        return _responses[11]
    elif text_lower in ["library"]:
        return _responses[14]
    elif text_lower in ["placement", "placements"]:
        return _responses[13]
    elif text_lower in ["faculty"]:
        return _responses[3]
    elif text_lower in ["directory", "faculty directory"]:
        return _responses[15]
    elif text_lower in ["department contacts", "contacts", "department contact"]:
        return _responses[22]
    elif text_lower in ["calendar", "academic calendar"]:
        return _responses[16]
    elif text_lower in ["research", "research facilities"]:
        return _responses[17]
    elif text_lower in ["parent"]:
        return _responses[4]
    elif text_lower in ["progress", "student progress"]:
        return _responses[20]
    elif text_lower in ["fees", "fee structure", "fee"]:
        return _responses[9]
    elif text_lower in ["grievance"]:
        return _responses[21]
    elif text_lower in ["visitor"]:
        return _responses[5]
    elif text_lower in ["campus location", "location"]:
        return _responses[18]
    elif text_lower in ["events", "seminars", "events & seminars", "events and seminars"]:
        return _responses[19]
    elif text_lower in ["contact", "contact info"]:
        return _responses[18]

    # Preprocess for NLP
    processed = _preprocess(text)

    # ── Step 1: Naive-Bayes intent classification ──────────
    intent, confidence = _predict_intent(text)

    if confidence >= 0.55:
        # Find the best corpus entry matching this intent
        intent_indices = [i for i, t in enumerate(_tags) if t == intent]
        if intent_indices:
            if len(intent_indices) == 1:
                return _responses[intent_indices[0]]
            # Among matching intent entries pick highest TF-IDF similarity
            sub_matrix = _tfidf_matrix[intent_indices]
            try:
                vec  = _vectorizer.transform([processed])
                sims = cosine_similarity(vec, sub_matrix).flatten()
                best = intent_indices[int(np.argmax(sims))]
                return _responses[best]
            except Exception:
                return _responses[intent_indices[0]]

    # ── Step 2: TF-IDF cosine similarity ──────────────────
    try:
        vec  = _vectorizer.transform([processed])
        sims = cosine_similarity(vec, _tfidf_matrix).flatten()
        best_idx   = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        if best_score >= 0.10:
            return _responses[best_idx]
    except Exception:
        pass

    # ── Step 3: Keyword / fuzzy fallback ──────────────────
    kw_map = {
        "admission":   ["admission","apply","join","enroll","application"],
        "fees":        ["fee","fees","cost","tuition","pay","payment"],
        "courses":     ["course","program","degree","branch","department"],
        "hostel":      ["hostel","room","accommodation","boarding","stay"],
        "exams":       ["exam","test","schedule","result","marks","timetable","hall ticket"],
        "placement":   ["placement","job","career","package","recruit","salary"],
        "contact":     ["contact","phone","address","location","email"],
        "scholarship": ["scholarship","financial","merit","concession","sc","st"],
        "library":     ["library","book","e-book","journal","digital"],
        "events":      ["event","fest","seminar","workshop","symposium","hackathon"],
        "grievance":   ["grievance","complaint","issue","problem","redressal"],
        "faculty":     ["faculty","professor","hod","department","staff"],
        "greet":       ["hello","hi","hey","morning","evening","welcome"],
    }
    for intent_kw, kws in kw_map.items():
        if any(kw in text_lower for kw in kws):
            for i, tag in enumerate(_tags):
                if tag == intent_kw:
                    return _responses[i]

    return FALLBACK


# ── Expose entity extractor for analytics ─────────────────
def extract_entities(text: str) -> dict:
    return _extract_entities(text)


# ── Device-aware response wrapper ─────────────────────────
def get_response_for_device(user_input: str,
                             device: str = "desktop") -> str:
    """
    Convenience wrapper: get_response() + lightweight pre-formatting
    that the engine can apply before the response reaches app.py.

    The heavy formatting (line-wrap, section breaks, mobile truncation)
    is handled by response_formatter.py in app.py.  This wrapper only
    applies engine-level heuristics:

      • On mobile, prefer shorter corpus entries when multiple intent
        matches have similar TF-IDF scores.
      • On all devices, strip trailing whitespace from each response.
    """
    resp = get_response(user_input)

    # Strip trailing whitespace from every <br>-terminated segment
    resp = resp.strip()

    # On mobile, if the response is a long HTML block, prefer the
    # short greeting variant (already handled by FALLBACK & corpus design,
    # but this is a safety net for future corpus additions).
    if device == "mobile" and len(resp) > 2000:
        # Truncate at a safe <br> boundary — final clean-up done in formatter
        last_br = resp[:1800].rfind("<br>")
        if last_br > 800:
            resp = resp[:last_br] + "<br>📱 <i>Type a keyword for details.</i>"

    return resp
