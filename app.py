import streamlit as st
import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain.chains import ConversationChain

# Inisialisasi model Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=os.environ.get("GEMINI_API_KEY")
)

# Fungsi untuk menghasilkan konten
def generate_content(prompt):
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

# Parsing kuis dari respons Gemini
def parse_quiz(response):
    questions = []
    blocks = response.split('\n\n')
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 6:
            question = lines[0]
            options = lines[1:5]
            correct_line = lines[5]
            explanation = lines[6] if len(lines) > 6 else ""
            correct = correct_line.split(':')[1].strip()
            questions.append({
                'question': question,
                'options': options,
                'correct': correct,
                'explanation': explanation
            })
    return questions

# Menghasilkan kuis diagnostik
def generate_diagnostic_quiz():
    prompt = "Generate a 10-question multiple-choice quiz covering the full syllabus of AI and programming. Each question should have four options (a, b, c, d), indicate the correct answer, and provide an explanation for why the correct answer is right and why others are wrong."
    response = generate_content(prompt)
    return parse_quiz(response)

# Menghasilkan kuis per modul
def generate_module_quiz(module_name, num_questions=5):
    prompt = f"Generate a {num_questions}-question multiple-choice quiz for the module: {module_name}. Each question should have four options (a, b, c, d), indicate the correct answer, and provide an explanation for why the correct answer is right and why others are wrong."
    response = generate_content(prompt)
    return parse_quiz(response)

# Menilai kuis
def assess_quiz(user_answers, questions):
    score = 0
    feedback = []
    for i, q in enumerate(questions):
        user_answer = user_answers.get(f'q{i}')
        if user_answer == q['correct']:
            score += 1
            feedback.append(f"{q['question']} - Benar!")
        else:
            feedback.append(f"{q['question']} - Salah. Penjelasan: {q['explanation']}")
    return score, feedback

# Menghasilkan konten adaptif
def generate_adaptive_content(profile, module):
    prompt = f"Provide detailed learning content suitable for a {profile} level student on the topic: {module}. Include explanations, examples, and key concepts."
    return generate_content(prompt)

# Menghasilkan aktivitas pembelajaran
def generate_learning_activity(module, is_technical):
    if is_technical:
        prompt = f"Generate a hands-on coding activity for the module: {module}. Provide a task description and an example solution."
    else:
        prompt = f"Generate a reflective writing or analysis activity for the module: {module}. Provide a task description."
    return generate_content(prompt)

# Menghasilkan rekap
def generate_recap(module):
    prompt = f"Provide a summary and key takeaways for the module: {module}."
    return generate_content(prompt)

# Dukungan percakapan
def get_conversational_response(query):
    if 'conversation' not in st.session_state:
        st.session_state.conversation = ConversationChain(llm=llm)
    return st.session_state.conversation.run(query)

# Halaman Utama
def main_page():
    st.title("AI Engineering Academy")
    st.write("Selamat datang di AI Tutor adaptif Anda!")
    if st.button("Mulai Kuis Diagnostik"):
        st.session_state.page = "diagnostic_quiz"
        st.session_state.quiz_generated = False

# Halaman Kuis Diagnostik
def diagnostic_quiz_page():
    st.title("Kuis Diagnostik")
    if not st.session_state.get('quiz_generated', False):
        st.session_state.quiz = generate_diagnostic_quiz()
        st.session_state.quiz_generated = True
    questions = st.session_state.quiz
    user_answers = {}
    for i, q in enumerate(questions):
        st.write(q['question'])
        user_answers[f'q{i}'] = st.radio(f"Pertanyaan {i+1}", q['options'], key=f"diag_q{i}")
    if st.button("Kirim Jawaban"):
        score, feedback = assess_quiz(user_answers, questions)
        st.write("\n".join(feedback))
        if score >= 8:
            profile = "Advanced"
        elif score >= 5:
            profile = "Intermediate"
        else:
            profile = "Beginner"
        st.session_state.profile = profile
        st.session_state.format_prefs = {"video": 0, "text": 0, "visuals": 0}
        st.write(f"Skor Anda: {score}/10. Profil Anda: {profile}")
        if st.button("Lanjut ke Dashboard Modul"):
            st.session_state.page = "module_dashboard"

# Halaman Dashboard Modul
def module_dashboard_page():
    st.title("Dashboard Modul")
    st.write(f"Profil: {st.session_state.profile}")
    modules = [
        {"name": "Intro to AI", "type": "flexible"},
        {"name": "Python for AI Engineering", "type": "technical"},
        {"name": "Data Cleaning & EDA", "type": "technical"},
        {"name": "ML Foundations & Evaluation", "type": "technical"},
        {"name": "Deep Learning & Transformers Basics", "type": "technical"},
        {"name": "Prompting, RAG & Agentic Apps", "type": "flexible"},
        {"name": "Ethics, Portfolio, Professional Growth", "type": "flexible"}
    ]
    for module in modules:
        if st.button(module["name"]):
            st.session_state.current_module = module
            st.session_state.page = "module_page"
            st.session_state.module_state = {
                "entry_quiz_done": False,
                "content_viewed": False,
                "activity_done": False,
                "final_quiz_done": False,
                "start_time": time.time(),
                "time_spent": 0
            }

# Halaman Modul
def module_page():
    module = st.session_state.current_module
    st.title(module["name"])
    profile = st.session_state.profile
    module_state = st.session_state.module_state
    is_technical = module["type"] == "technical"

    # Kuis Masuk
    if not module_state["entry_quiz_done"]:
        st.subheader("Kuis Masuk")
        if "entry_quiz" not in st.session_state:
            st.session_state.entry_quiz = generate_module_quiz(module["name"])
        questions = st.session_state.entry_quiz
        user_answers = {}
        for i, q in enumerate(questions):
            st.write(q['question'])
            user_answers[f'q{i}'] = st.radio(f"Pertanyaan {i+1}", q['options'], key=f"entry_q{i}")
        if st.button("Kirim Kuis Masuk"):
            score, feedback = assess_quiz(user_answers, questions)
            st.write("\n".join(feedback))
            st.write(f"Skor Anda: {score}/{len(questions)}")
            module_state["entry_quiz_done"] = True
            st.session_state.module_state = module_state
    else:
        # Pengiriman Konten
        st.subheader("Konten Pembelajaran")
        if "content" not in st.session_state:
            st.session_state.content = generate_adaptive_content(profile, module["name"])
        content = st.session_state.content
        format_choice = st.selectbox("Pilih format:", ["Teks", "Video (placeholder)", "Visual (placeholder)"])
        if format_choice == "Teks":
            st.session_state.format_prefs["text"] += 1
            st.write(content)
        elif format_choice == "Video (placeholder)":
            st.session_state.format_prefs["video"] += 1
            st.write(f"Video: Konten untuk {module['name']} (tersedia dalam format video)")
        else:
            st.session_state.format_prefs["visuals"] += 1
            st.write(f"Visual: Diagram dan ilustrasi untuk {module['name']}")
        if st.button("Tandai Konten Dilihat"):
            module_state["content_viewed"] = True
            st.session_state.module_state = module_state

        # Aktivitas Pembelajaran
        if module_state["content_viewed"]:
            st.subheader("Aktivitas Pembelajaran")
            if "activity" not in st.session_state:
                st.session_state.activity = generate_learning_activity(module["name"], is_technical)
            st.write(st.session_state.activity)
            response = st.text_area("Masukkan jawaban Anda:")
            if st.button("Kirim Aktivitas") and response:
                module_state["activity_done"] = True
                st.session_state.module_state = module_state

        # Mode Explorer atau Kuis Akhir
        if module_state["content_viewed"] and module_state["activity_done"] or profile == "Advanced":
            st.subheader("Kuis Akhir")
            if profile == "Advanced" and not module_state["final_quiz_done"]:
                st.write("Mode Explorer: Anda dapat mencoba kuis akhir sekarang.")
            if "final_quiz" not in st.session_state:
                st.session_state.final_quiz = generate_module_quiz(module["name"])
            questions = st.session_state.final_quiz
            user_answers = {}
            for i, q in enumerate(questions):
                st.write(q['question'])
                user_answers[f'q{i}'] = st.radio(f"Pertanyaan {i+1}", q['options'], key=f"final_q{i}")
            if st.button("Kirim Kuis Akhir"):
                score, feedback = assess_quiz(user_answers, questions)
                st.write("\n".join(feedback))
                passing_score = len(questions) * 0.7
                if score >= passing_score:
                    st.write(f"Skor Anda: {score}/{len(questions)}. Anda lulus!")
                    module_state["final_quiz_done"] = True
                    st.session_state.module_state = module_state
                else:
                    st.write(f"Skor Anda: {score}/{len(questions)}. Anda belum lulus. Silakan tinjau kembali konten.")
                    remediation = generate_content(f"Provide remediation content for {module['name']} where the student scored {score}/{len(questions)}.")
                    st.write(remediation)
                    del st.session_state["final_quiz"]  # Reset kuis untuk dicoba lagi

        # Refleksi dan Konsolidasi
        if module_state["final_quiz_done"]:
            st.subheader("Rekap dan Refleksi")
            recap = generate_recap(module["name"])
            st.write(recap)
            st.write("Refleksikan apa yang telah Anda pelajari dan bagaimana Anda dapat menerapkannya.")
            reflection = st.text_area("Tulis refleksi Anda:")
            if st.button("Kirim Refleksi") and reflection:
                st.write("Terima kasih atas refleksi Anda!")
                st.session_state.page = "module_dashboard"

    # Pemantauan Kemajuan
    current_time = time.time()
    module_state["time_spent"] = current_time - module_state["start_time"]
    st.write(f"Waktu yang dihabiskan: {int(module_state['time_spent'] / 60)} menit")
    recommended_time = 8 * 60  # 8 jam dalam menit untuk contoh
    if module_state["time_spent"] < recommended_time * 0.5:
        st.write("Dorongan: Pertimbangkan untuk menghabiskan lebih banyak waktu di modul ini.")

# Dukungan Percakapan
def chat_support():
    st.sidebar.title("AI Tutor")
    query = st.sidebar.text_input("Tanyakan apa saja:")
    if query:
        response = get_conversational_response(query)
        st.sidebar.write(response)

# Simulasi Interaksi Peer-to-Peer
def peer_interaction():
    st.sidebar.subheader("Interaksi Peer")
    if st.button("Cari Teman Belajar"):
        st.sidebar.write("Rekomendasi: Anda dapat terhubung dengan 'Pengguna A' yang memiliki kesulitan serupa.")

# Logika Navigasi
if "page" not in st.session_state:
    st.session_state.page = "main"

if st.session_state.page == "main":
    main_page()
elif st.session_state.page == "diagnostic_quiz":
    diagnostic_quiz_page()
elif st.session_state.page == "module_dashboard":
    module_dashboard_page()
elif st.session_state.page == "module_page":
    module_page()

# Komponen Sidebar
chat_support()
peer_interaction()
