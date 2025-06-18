from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Frame
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.colors import HexColor
from reportlab.platypus.flowables import HRFlowable
import markdown2

text = '''Here's a detailed analysis and strategic recommendations for Hasnaoui Private Hospital's Facebook content and engagement:

---

### **1. Post Analysis & Engagement Opportunities**  
**a. Announcement Posts (e.g., Clinica Salon, Marathon Participation)**  
- **Tone/Clarity**: Formal and informative but lacks emotional appeal.  
- **Missed Opportunities**:  
  - No call-to-action (e.g., "Visit us at Booth G!" or "Share your marathon photos with #HasnaouiSupports").  
  - Minimal interaction despite community relevance (e.g., tagging local partners/sponsors could boost visibility).  

**b. Service Updates (e.g., ER Availability, Surgical Techniques)**  
- **Strengths**: Clear, professional, and highlights expertise.  
- **Gaps**:  
  - Comments like "????" or "@للجميع" suggest confusion or lack of context (e.g., no FAQs on "CPRE" or "DBS" procedures).  
  - No replies to comments (e.g., "@89" or "@followers" tags go unanswered).  

**c. Emotional/National Posts (Martyrs' Day, Paramedical Tribute)**  
- **Impact**: High emotional resonance (patriotism, gratitude) but superficial engagement (e.g., 🇩🇿 emojis without deeper dialogue).  

**d. Promotional Posts (Free Cardiology Consultation, Facility Tour Video)**  
- **Effectiveness**: Strong value proposition (free care, advanced tech).  
- **Improvements**:  
  - Address barriers to sign-ups (e.g., "No referral letter? Contact us for guidance!").  
  - Video tour could include patient testimonials or live Q&A sessions.  

---

### **2. Recurring Themes & Needs**  
- **Patient Education**: Questions about procedures (e.g., "out-in technique," "DBS") indicate demand for explainer content.  
- **Accessibility**: Limited slots for free consultations may frustrate users; transparency about waitlists could help.  
- **Local Pride**: Engagement spikes with local events (marathon) and Arabic content—leverage regional identity more.  

---

### **3. Content Strategy Improvements**  
**a. Boost Engagement:**  
- **Reply to ALL comments**, even simple ones (e.g., "Thank you! 🇩🇿" to patriotic replies).  
- **Use polls/questions**: "Which health topic should we cover next?"  

**b. Educational Content:**  
- **FAQ Series**: Short videos/posts explaining terms like "arthroscopy" or "varicose laser treatment."  
- **Patient Stories**: Highlight successful treatments (with consent) to humanize services.  

**c. Event Posts:**  
- **Pre- and post-event coverage**: "Meet our team at Clinica Salon! Drop by for a free blood pressure check."  

---

### **4. Suggested FAQ/Responses**  
**For Surgical Technique Posts (e.g., "CPRE"):**  
> *"La CPRE (Cholangiopancréatographie Rétrograde Endoscopique) est une procédure mini-invasive pour traiter les calculs biliaires ou les tumeurs. Elle évite une chirurgie ouverte. Contactez-nous pour plus de détails!"*  

**For Free Consultation Posts:**  
> *"Pas de lettre d'orientation? Notre secrétariat peut vous aider à obtenir un avis médical préalable. Appelez-nous!"*  

---

### **5. New Post Ideas**  
- **"Meet Our Team" Series**: Introduce doctors with credentials and patient quotes.  
- **Behind-the-Scenes**: Show sterilization protocols or lab techs at work to build trust.  
- **Health Tips**: Seasonal advice (e.g., "5 Signs You Need a Cardiac Check-Up").  
- **Community Partnerships**: Feature local NGOs or sports teams sponsored by the hospital.  

---

### **6. Benchmarking Algerian Private Hospitals**  
- **Best Practices Observed**:  
  - **Pasteur Clinic (Algiers)**: Uses Instagram Reels for quick FAQs.  
  - **Villa Mélissa (Oran)**: Shares live surgery success rates to showcase transparency.  
  - **Sidi Maarouf Clinic (Casablanca)**: Offers chatbot appointments via Messenger.  

**Recommendation**: Adopt a hybrid approach—combine Hasnaoui's local authenticity with tech-savvy engagement tools.  

---

### **Key Takeaway**  
Hasnaoui's content is professionally crafted but needs **more two-way dialogue, patient-centric storytelling, and proactive education** to convert passive followers into engaged patients. Prioritize responsiveness and demystify medical jargon to stand out in Algeria's competitive healthcare market.  

Would you like a tailored response template for specific comment threads?'''

def clean_markdown(text):
    """Enhanced markdown cleaner with better HTML handling"""
    html = markdown2.markdown(text)
    html = (html
            .replace('<li>', '• ')
            .replace('</li>', '<br/>')
            .replace('<ul>', '')
            .replace('</ul>', '')
            .replace('<p>', '')
            .replace('</p>', '<br/>')
            .replace('<h1>', '<font size="16" color="#82CBE8"><b>')
            .replace('</h1>', '</b></font><br/>')
            .replace('<h2>', '<font size="14" color="#82CBE8"><b>')
            .replace('</h2>', '</b></font><br/>')
            .replace('<h3>', '<font size="12" color="#82CBE8"><b>')
            .replace('</h3>', '</b></font><br/>')
            .replace('<blockquote>', '<font color="#555555"><i>')
            .replace('</blockquote>', '</i></font>')
            .replace('<hr/>', '<br/><hr width="50%" color="#82CBE8"/><br/>')
            .replace('<em>', '<i>').replace('</em>', '</i>')
            .replace('<strong>', '<b>').replace('</strong>', '</b>')
            .replace('<p><i>', '<i>').replace('</i></p>', '</i>') 
            )
    return html

def create_pdf_from_markdown(text, output_filename):
    doc = SimpleDocTemplate(output_filename, pagesize=letter,
                          rightMargin=50, leftMargin=50,
                          topMargin=50, bottomMargin=50)
    
    styles = getSampleStyleSheet()
    
    main_style = ParagraphStyle(
        name="MainStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=HexColor("#333333"),
        spaceAfter=6,
        alignment=TA_LEFT
    )
    
    header_style = ParagraphStyle(
        name="HeaderStyle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=16,
        textColor=HexColor("#82CBE8"),
        spaceAfter=12,
        alignment=TA_CENTER
    )
    
    formatted_text = clean_markdown(text)
    
    story = []
    
    title = Paragraph("Hasnaoui Hospital Social Media Analysis", header_style)
    story.append(title)
    story.append(HRFlowable(width="100%", thickness=1, lineCap='round', 
                          color=HexColor("#82CBE8"), spaceAfter=20))
    
    story.append(Paragraph(formatted_text, main_style))
    
    doc.build(story)
    print(f"Successfully created {output_filename}")

create_pdf_from_markdown(text, "hospital_analysis_enhanced.pdf")