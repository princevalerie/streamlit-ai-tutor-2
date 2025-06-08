import streamlit as st
import json
import time
from datetime import datetime, timedelta
import pandas as pd

# Try importing Google Generative AI
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    st.warning("google-generativeai not installed. AI features will be limited.")

# Try importing LangChain components with fallbacks
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    LANGCHAIN_GOOGLE_AVAILABLE = True
except ImportError:
    LANGCHAIN_GOOGLE_AVAILABLE = False

try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    try:
        from langchain.schema import HumanMessage, SystemMessage
    except ImportError:
        # Fallback: create simple message classes
        class HumanMessage:
            def __init__(self, content):
                self.content = content
        
        class SystemMessage:
            def __init__(self, content):
                self.content = content
        
        LANGCHAIN_GOOGLE_AVAILABLE = False

# Configure page
st.set_page_config(
    page_title="AI Engineering Academy - AI Tutor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def init_session_state():
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = None
    if 'current_module' not in st.session_state:
        st.session_state.current_module = None
    if 'quiz_attempts' not in st.session_state:
        st.session_state.quiz_attempts = {}
    if 'module_progress' not in st.session_state:
        st.session_state.module_progress = {}
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""

init_session_state()

# AI Tutor Configuration
class AITutor:
    def __init__(self, api_key):
        self.api_key = api_key
        self.llm = None
        
        if api_key and GENAI_AVAILABLE:
            try:
                genai.configure(api_key=api_key)
                if LANGCHAIN_GOOGLE_AVAILABLE:
                    self.llm = ChatGoogleGenerativeAI(
                        model="gemini-pro",
                        google_api_key=api_key,
                        temperature=0.7
                    )
                else:
                    # Direct Gemini API usage as fallback
                    self.model = genai.GenerativeModel('gemini-pro')
            except Exception as e:
                st.error(f"Error initializing AI: {str(e)}")
    
    def get_tutor_response(self, user_message, context=""):
        if not self.api_key:
            return "Please configure your Gemini API key first."
        
        if not GENAI_AVAILABLE:
            return "Google Generative AI library not available. Please install: pip install google-generativeai"
        
        system_prompt = f"""
        You are an AI Tutor for the AI Engineering Academy. Your role is to:
        1. Adapt your teaching style based on student level (Beginner/Intermediate/Advanced)
        2. Provide clear, engaging explanations
        3. Offer multiple learning formats (visual, text, code examples)
        4. Give constructive feedback and motivation
        5. Guide students through their learning journey
        
        Current context: {context}
        Student level: {st.session_state.user_profile.get('level', 'Unknown') if st.session_state.user_profile else 'Unknown'}
        
        Respond in a helpful, encouraging manner. Keep responses concise but comprehensive.
        """
        
        try:
            if self.llm and LANGCHAIN_GOOGLE_AVAILABLE:
                # Use LangChain if available
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_message)
                ]
                response = self.llm(messages)
                return response.content
            else:
                # Direct Gemini API usage
                full_prompt = f"{system_prompt}\n\nUser: {user_message}\nAI Tutor:"
                response = self.model.generate_content(full_prompt)
                return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}. Please check your API key and try again."

# Module and Quiz Data
MODULES = {
    "Block 1": {
        "Module 1": {
            "title": "Introduction to AI",
            "type": "flexible",
            "estimated_time": "4-6 hours",
            "description": "Foundational concepts in Artificial Intelligence"
        },
        "Module 2": {
            "title": "Python for AI Engineering",
            "type": "technical",
            "estimated_time": "8-12 hours",
            "description": "Python programming fundamentals for AI"
        },
        "Module 3": {
            "title": "Math Essentials",
            "type": "technical",
            "estimated_time": "8-10 hours",
            "description": "Mathematical foundations for AI"
        },
        "Module 4": {
            "title": "Text EDA",
            "type": "technical",
            "estimated_time": "6-8 hours",
            "description": "Exploratory Data Analysis for text data"
        }
    },
    "Block 2": {
        "Module 1": {
            "title": "ML Foundations",
            "type": "technical",
            "estimated_time": "12-16 hours",
            "description": "Machine Learning fundamentals"
        },
        "Module 2": {
            "title": "ML Evaluation",
            "type": "technical",
            "estimated_time": "12-16 hours",
            "description": "Model evaluation and validation"
        },
        "Module 3": {
            "title": "ML Pipelines & Scikit-learn",
            "type": "technical",
            "estimated_time": "12-16 hours",
            "description": "Building ML pipelines with Scikit-learn"
        }
    }
}

DIAGNOSTIC_QUIZ = [
    {
        "question": "What is your experience with Python programming?",
        "options": ["No experience", "Basic (variables, loops)", "Intermediate (OOP, libraries)", "Advanced (frameworks, optimization)"],
        "category": "programming"
    },
    {
        "question": "How familiar are you with machine learning concepts?",
        "options": ["Never heard of it", "Basic understanding", "Can explain algorithms", "Have built ML models"],
        "category": "ml"
    },
    {
        "question": "What is your experience with data analysis?",
        "options": ["No experience", "Basic (Excel, simple stats)", "Intermediate (pandas, visualization)", "Advanced (complex analysis, big data)"],
        "category": "data"
    },
    {
        "question": "How comfortable are you with mathematics (linear algebra, statistics)?",
        "options": ["Not comfortable", "Basic understanding", "Comfortable with concepts", "Advanced mathematical background"],
        "category": "math"
    },
    {
        "question": "What is your experience with deep learning?",
        "options": ["No experience", "Heard about it", "Basic understanding", "Have implemented neural networks"],
        "category": "dl"
    }
]

def calculate_user_level(responses):
    score = 0
    for response in responses:
        score += response
    
    if score <= 5:
        return "Beginner"
    elif score <= 10:
        return "Intermediate"
    else:
        return "Advanced"

# Sidebar Configuration
def setup_sidebar():
    with st.sidebar:
        st.title("🤖 AI Tutor Settings")
        
        # API Key Input
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=st.session_state.api_key,
            help="Enter your Google Gemini API key"
        )
        
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
            st.rerun()
        
        if st.session_state.user_profile:
            st.success(f"Level: {st.session_state.user_profile['level']}")
            
            st.subheader("📚 Progress Overview")
            total_modules = sum(len(modules) for modules in MODULES.values())
            completed = len([m for m in st.session_state.module_progress.values() if m.get('completed')])
            progress_percentage = (completed / total_modules) * 100 if total_modules > 0 else 0
            
            st.progress(progress_percentage / 100)
            st.write(f"Completed: {completed}/{total_modules} modules")
            
            if st.button("🔄 Reset Profile"):
                for key in ['user_profile', 'current_module', 'quiz_attempts', 'module_progress']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

# Main Application Functions
def diagnostic_quiz():
    st.title("🎯 Global Diagnostic Quiz")
    st.markdown("This quiz will help us understand your current knowledge level and customize your learning experience.")
    
    if 'quiz_responses' not in st.session_state:
        st.session_state.quiz_responses = []
    
    with st.form("diagnostic_quiz"):
        responses = []
        for i, q in enumerate(DIAGNOSTIC_QUIZ):
            st.subheader(f"Question {i+1}")
            st.write(q["question"])
            response = st.radio(
                "Select your answer:",
                options=range(len(q["options"])),
                format_func=lambda x, opts=q["options"]: opts[x],
                key=f"q_{i}"
            )
            responses.append(response)
        
        submitted = st.form_submit_button("Complete Assessment")
        
        if submitted:
            level = calculate_user_level(responses)
            st.session_state.user_profile = {
                'level': level,
                'responses': responses,
                'timestamp': datetime.now()
            }
            st.success(f"Assessment complete! Your level: {level}")
            st.rerun()

def module_selection():
    st.title("📚 Module Selection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Available Modules")
        
        for block_name, modules in MODULES.items():
            with st.expander(f"📖 {block_name}", expanded=True):
                for module_key, module_info in modules.items():
                    col_mod, col_btn = st.columns([3, 1])
                    
                    with col_mod:
                        status = "✅" if st.session_state.module_progress.get(f"{block_name}_{module_key}", {}).get('completed') else "⏳"
                        st.write(f"{status} **{module_info['title']}**")
                        st.write(f"*{module_info['description']}*")
                        st.write(f"📊 Type: {module_info['type'].title()} | ⏱️ {module_info['estimated_time']}")
                    
                    with col_btn:
                        if st.button(f"Start", key=f"start_{block_name}_{module_key}"):
                            st.session_state.current_module = {
                                'block': block_name,
                                'module': module_key,
                                'info': module_info
                            }
                            st.rerun()
    
    with col2:
        if st.session_state.user_profile:
            st.markdown("### 👤 Your Profile")
            st.info(f"**Level:** {st.session_state.user_profile['level']}")
            
            # Recommendations based on level
            if st.session_state.user_profile['level'] == 'Beginner':
                st.markdown("**Recommended Path:**")
                st.write("- Start with Block 1, Module 1")
                st.write("- Focus on fundamentals")
                st.write("- Take time with each concept")
            elif st.session_state.user_profile['level'] == 'Intermediate':
                st.markdown("**Recommended Path:**")
                st.write("- You may skip basic concepts")
                st.write("- Focus on practical applications")
                st.write("- Use Explorer Mode when ready")
            else:
                st.markdown("**Recommended Path:**")
                st.write("- Use Explorer Mode")
                st.write("- Focus on advanced topics")
                st.write("- Consider peer mentoring")

def module_content():
    if not st.session_state.current_module:
        st.error("No module selected. Please go back to module selection.")
        return
    
    module = st.session_state.current_module
    module_key = f"{module['block']}_{module['module']}"
    
    st.title(f"📖 {module['info']['title']}")
    st.markdown(f"**Block:** {module['block']} | **Type:** {module['info']['type'].title()}")
    
    # Progress tracking
    if module_key not in st.session_state.module_progress:
        st.session_state.module_progress[module_key] = {
            'started': datetime.now(),
            'entry_quiz_completed': False,
            'content_viewed': False,
            'activity_completed': False,
            'final_quiz_completed': False,
            'completed': False
        }
    
    progress = st.session_state.module_progress[module_key]
    
    # Module steps
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Entry Quiz", "📚 Content", "🛠️ Activity", "✅ Final Quiz"])
    
    with tab1:
        st.subheader("Entry Quiz")
        if not progress['entry_quiz_completed']:
            st.info("Complete this quick assessment to customize your learning path.")
            
            with st.form("entry_quiz"):
                st.write("How familiar are you with the topics in this module?")
                familiarity = st.slider("Familiarity Level", 0, 10, 5)
                
                st.write("What specific aspects would you like to focus on?")
                focus_areas = st.multiselect(
                    "Select focus areas:",
                    ["Theory", "Practical Applications", "Code Examples", "Best Practices", "Advanced Concepts"]
                )
                
                if st.form_submit_button("Complete Entry Quiz"):
                    progress['entry_quiz_completed'] = True
                    progress['familiarity'] = familiarity
                    progress['focus_areas'] = focus_areas
                    st.success("Entry quiz completed!")
                    st.rerun()
        else:
            st.success("✅ Entry quiz completed!")
            st.write(f"Familiarity level: {progress.get('familiarity', 0)}/10")
            st.write(f"Focus areas: {', '.join(progress.get('focus_areas', []))}")
    
    with tab2:
        st.subheader("Learning Content")
        if not progress['entry_quiz_completed']:
            st.warning("Please complete the entry quiz first.")
        else:
            # Content formats
            format_choice = st.selectbox(
                "Choose your preferred learning format:",
                ["📹 Video", "📖 Reading", "🖼️ Visual", "💻 Code Examples"]
            )
            
            # Simulated content based on format
            if format_choice == "📹 Video":
                st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
                st.write("Watch this comprehensive video covering the key concepts.")
            elif format_choice == "📖 Reading":
                st.markdown(f"""
                ## {module['info']['title']} - Comprehensive Guide
                
                This module covers essential concepts that every AI engineer should master.
                
                ### Key Learning Objectives:
                - Understand core principles
                - Apply theoretical knowledge
                - Develop practical skills
                - Build real-world applications
                
                ### Content Overview:
                {module['info']['description']}
                
                *Continue reading the full material...*
                """)
            elif format_choice == "🖼️ Visual":
                st.image("https://via.placeholder.com/600x400/4CAF50/FFFFFF?text=Learning+Diagram", 
                        caption="Conceptual diagram for this module")
                st.markdown("Visual learners benefit from diagrams, charts, and infographics.")
            else:  # Code Examples
                st.code("""
# Example code for this module
def main_concept():
    \"\"\"
    This function demonstrates the key concept
    covered in this module.
    \"\"\"
    result = perform_operation()
    return process_result(result)

# Practice implementing this yourself!
                """, language="python")
            
            if st.button("Mark Content as Viewed"):
                progress['content_viewed'] = True
                st.success("Content marked as viewed!")
                st.rerun()
    
    with tab3:
        st.subheader("Learning Activity")
        if not progress['content_viewed']:
            st.warning("Please view the content first.")
        else:
            if module['info']['type'] == 'technical':
                st.markdown("### 💻 Coding Activity")
                st.write("Complete the following coding exercise:")
                
                code_input = st.text_area(
                    "Write your code here:",
                    height=200,
                    placeholder="# Your code here\n\n"
                )
                
                if st.button("Submit Code"):
                    if code_input.strip():
                        progress['activity_completed'] = True
                        progress['activity_code'] = code_input
                        st.success("Activity completed!")
                        st.rerun()
                    else:
                        st.error("Please write some code before submitting.")
            
            else:  # Flexible module
                st.markdown("### 📝 Reflection Activity")
                st.write("Complete the following reflection exercise:")
                
                reflection = st.text_area(
                    "Your reflection:",
                    height=150,
                    placeholder="Share your thoughts and insights..."
                )
                
                if st.button("Submit Reflection"):
                    if reflection.strip():
                        progress['activity_completed'] = True
                        progress['activity_reflection'] = reflection
                        st.success("Activity completed!")
                        st.rerun()
                    else:
                        st.error("Please write your reflection before submitting.")
            
            if progress['activity_completed']:
                st.success("✅ Activity completed!")
    
    with tab4:
        st.subheader("Final Quiz")
        if not progress['activity_completed']:
            st.warning("Please complete the learning activity first.")
        else:
            quiz_key = f"{module_key}_final"
            
            # Check for quiz cooldown
            last_attempt = st.session_state.quiz_attempts.get(quiz_key, {}).get('last_attempt')
            if last_attempt:
                time_diff = datetime.now() - last_attempt
                if time_diff < timedelta(hours=2):
                    remaining = timedelta(hours=2) - time_diff
                    st.warning(f"Quiz cooldown active. Please wait {remaining} before retrying.")
                    return
            
            st.info("Pass this quiz to complete the module and advance.")
            
            with st.form("final_quiz"):
                # Sample quiz questions
                q1 = st.radio(
                    "Question 1: Which concept is most important in this module?",
                    ["Option A", "Option B", "Option C", "Option D"]
                )
                
                q2 = st.radio(
                    "Question 2: How would you apply this knowledge?",
                    ["Option A", "Option B", "Option C", "Option D"]
                )
                
                q3 = st.text_area(
                    "Question 3: Explain the main concept in your own words:",
                    height=100
                )
                
                submitted = st.form_submit_button("Submit Final Quiz")
                
                if submitted:
                    # Simple scoring logic
                    score = 75 + (len(q3.split()) if q3 else 0)  # Base score + bonus for explanation
                    passed = score >= 70
                    
                    # Record attempt
                    if quiz_key not in st.session_state.quiz_attempts:
                        st.session_state.quiz_attempts[quiz_key] = {'attempts': 0}
                    
                    st.session_state.quiz_attempts[quiz_key]['attempts'] += 1
                    st.session_state.quiz_attempts[quiz_key]['last_attempt'] = datetime.now()
                    st.session_state.quiz_attempts[quiz_key]['last_score'] = score
                    
                    if passed:
                        progress['final_quiz_completed'] = True
                        progress['completed'] = True
                        st.success(f"🎉 Congratulations! You passed with {score}%")
                        st.balloons()
                    else:
                        st.error(f"Score: {score}%. You need 70% to pass. Review the material and try again in 2 hours.")
                    
                    st.rerun()

def ai_chat():
    st.title("💬 Chat with AI Tutor")
    
    if not st.session_state.api_key:
        st.warning("Please configure your Gemini API key in the sidebar to use the AI Tutor chat.")
        st.info("You can get a free API key from: https://makersuite.google.com/app/apikey")
        return
    
    if not GENAI_AVAILABLE:
        st.error("Google Generative AI library not installed. Please install it with:")
        st.code("pip install google-generativeai")
        return
    
    tutor = AITutor(st.session_state.api_key)
    
    # Display chat history
    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask your AI Tutor anything..."):
        # Add user message
        st.session_state.conversation_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get AI response
        context = f"Current module: {st.session_state.current_module}" if st.session_state.current_module else "Module selection"
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = tutor.get_tutor_response(prompt, context)
                st.write(response)
        
        # Add AI response
        st.session_state.conversation_history.append({"role": "assistant", "content": response})

# Main App
def main():
    setup_sidebar()
    
    # Main navigation
    if not st.session_state.user_profile:
        diagnostic_quiz()
    else:
        # Navigation menu
        if st.session_state.current_module:
            nav_option = st.selectbox(
                "Navigation:",
                ["📖 Current Module", "📚 Module Selection", "💬 AI Tutor Chat"],
                key="main_nav"
            )
        else:
            nav_option = st.selectbox(
                "Navigation:",
                ["📚 Module Selection", "💬 AI Tutor Chat"],
                key="main_nav"
            )
        
        if nav_option == "📚 Module Selection":
            st.session_state.current_module = None
            module_selection()
        elif nav_option == "📖 Current Module":
            module_content()
        elif nav_option == "💬 AI Tutor Chat":
            ai_chat()

if __name__ == "__main__":
    main()
