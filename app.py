import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, SystemMessage
import json
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Configure Streamlit page
st.set_page_config(
    page_title="AI Engineering Academy - Adaptive AI Tutor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'student_profile' not in st.session_state:
    st.session_state.student_profile = None
if 'current_module' not in st.session_state:
    st.session_state.current_module = 0
if 'quiz_attempts' not in st.session_state:
    st.session_state.quiz_attempts = {}
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'progress_tracking' not in st.session_state:
    st.session_state.progress_tracking = {}
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ""

# Sample course modules
MODULES = [
    {
        "id": 1,
        "title": "Introduction to AI",
        "description": "Fundamentals of Artificial Intelligence",
        "content_formats": ["video", "text", "interactive"],
        "difficulty": "beginner"
    },
    {
        "id": 2,
        "title": "Machine Learning Basics",
        "description": "Core concepts of Machine Learning",
        "content_formats": ["video", "text", "code"],
        "difficulty": "intermediate"
    },
    {
        "id": 3,
        "title": "Deep Learning",
        "description": "Neural Networks and Deep Learning",
        "content_formats": ["video", "text", "code", "interactive"],
        "difficulty": "advanced"
    },
    {
        "id": 4,
        "title": "Natural Language Processing",
        "description": "Text processing and language models",
        "content_formats": ["video", "text", "code"],
        "difficulty": "advanced"
    }
]

# Global diagnostic quiz questions
GLOBAL_QUIZ = [
    {
        "question": "What is the primary goal of supervised learning?",
        "options": [
            "To find hidden patterns in data without labels",
            "To learn from labeled examples to make predictions",
            "To optimize system performance through trial and error",
            "To reduce the dimensionality of data"
        ],
        "correct": 1,
        "difficulty": "beginner"
    },
    {
        "question": "Which algorithm is commonly used for classification tasks?",
        "options": [
            "K-means clustering",
            "Principal Component Analysis",
            "Random Forest",
            "Autoencoder"
        ],
        "correct": 2,
        "difficulty": "intermediate"
    },
    {
        "question": "What is backpropagation in neural networks?",
        "options": [
            "A method to initialize weights",
            "An algorithm to update weights by propagating errors backward",
            "A technique to prevent overfitting",
            "A way to select the best features"
        ],
        "correct": 1,
        "difficulty": "advanced"
    },
    {
        "question": "In Python, which library is most commonly used for machine learning?",
        "options": [
            "NumPy",
            "Pandas",
            "Scikit-learn",
            "Matplotlib"
        ],
        "correct": 2,
        "difficulty": "beginner"
    },
    {
        "question": "What is the vanishing gradient problem?",
        "options": [
            "When gradients become too large during training",
            "When gradients become very small in deep networks",
            "When the loss function doesn't converge",
            "When the model overfits to training data"
        ],
        "correct": 1,
        "difficulty": "advanced"
    }
]

def initialize_gemini(api_key):
    """Initialize Gemini AI with the provided API key"""
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=api_key,
            temperature=0.7,
            convert_system_message_to_human=True
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing Gemini: {str(e)}")
        return None

def create_ai_tutor_chain(llm, student_level):
    """Create LangChain for AI Tutor responses"""
    if student_level == "beginner":
        personality = "patient, encouraging, and uses simple explanations with lots of examples"
    elif student_level == "intermediate":
        personality = "supportive, provides moderate detail, and connects concepts"
    else:
        personality = "concise, technical, and focuses on advanced applications"
    
    template = f"""
You are an adaptive AI tutor for an AI Engineering Academy. You are {personality}.

You have access to detailed information about the student's current learning context:
{{context}}

Previous conversation: {{chat_history}}
Student Question/Input: {{input}}

IMPORTANT: Always acknowledge the student's current module and activities when relevant. 
For example, if they're working on "Machine Learning Basics" and haven't completed the video yet, 
mention this in your response and provide targeted guidance.

Provide a helpful, educational response that:
1. Acknowledges their current learning context
2. Matches their student level
3. Provides specific guidance for their current activities
4. Encourages progress on incomplete activities when appropriate

Response:
"""
    
    prompt = PromptTemplate(
        input_variables=["context", "input", "chat_history"],
        template=template
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=False)
    
    return chain

def calculate_student_level(quiz_results):
    """Calculate student level based on quiz performance"""
    correct_answers = sum(quiz_results)
    total_questions = len(quiz_results)
    score_percentage = (correct_answers / total_questions) * 100
    
    if score_percentage >= 80:
        return "advanced"
    elif score_percentage >= 60:
        return "intermediate"
    else:
        return "beginner"

def generate_module_quiz(module_id, student_level):
    """Generate module-specific entry quiz"""
    module = MODULES[module_id - 1]
    
    # Sample questions based on module and level
    questions = [
        {
            "question": f"What is your prior experience with {module['title']}?",
            "options": ["No experience", "Basic knowledge", "Some experience", "Advanced knowledge"],
            "correct": None  # Self-assessment
        },
        {
            "question": f"How confident are you in applying {module['title']} concepts?",
            "options": ["Not confident", "Somewhat confident", "Confident", "Very confident"],
            "correct": None
        }
    ]
    
    return questions

def display_progress_dashboard():
    """Display student progress dashboard"""
    st.subheader("📊 Your Learning Progress")
    
    if st.session_state.student_profile:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Student Level", st.session_state.student_profile['level'].capitalize())
        
        with col2:
            completed_modules = len([m for m in st.session_state.progress_tracking.values() if m.get('completed', False)])
            st.metric("Modules Completed", f"{completed_modules}/{len(MODULES)}")
        
        with col3:
            avg_score = sum([m.get('score', 0) for m in st.session_state.progress_tracking.values()]) / max(len(st.session_state.progress_tracking), 1)
            st.metric("Average Score", f"{avg_score:.1f}%")
        
        # Progress visualization
        if st.session_state.progress_tracking:
            modules_data = []
            for module_id, progress in st.session_state.progress_tracking.items():
                module_name = MODULES[module_id - 1]['title']
                modules_data.append({
                    'Module': module_name,
                    'Progress': progress.get('progress', 0),
                    'Score': progress.get('score', 0)
                })
            
            df = pd.DataFrame(modules_data)
            fig = px.bar(df, x='Module', y='Progress', title='Module Progress')
            st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("🤖 AI Engineering Academy - Adaptive AI Tutor")
    st.markdown("*Personalized AI learning experience that adapts to your pace and level*")
    
    # Sidebar for API key and navigation
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Enter your Gemini API Key:",
            type="password",
            value=st.session_state.gemini_api_key,
            help="Get your API key from Google AI Studio"
        )
        
        if api_key != st.session_state.gemini_api_key:
            st.session_state.gemini_api_key = api_key
        
        st.markdown("---")
        
        # Navigation
        st.header("📚 Navigation")
        page = st.selectbox(
            "Choose Section:",
            ["Home", "Global Assessment", "Learning Modules", "AI Tutor Chat", "Progress Dashboard"]
        )
    
    # Initialize Gemini if API key is provided
    llm = None
    if st.session_state.gemini_api_key:
        llm = initialize_gemini(st.session_state.gemini_api_key)
    else:
        st.warning("Please enter your Gemini API key in the sidebar to enable AI features.")
    
    # Main content based on selected page
    if page == "Home":
        display_home_page()
    elif page == "Global Assessment":
        display_global_assessment(llm)
    elif page == "Learning Modules":
        display_learning_modules(llm)
    elif page == "AI Tutor Chat":
        display_ai_tutor_chat(llm)
    elif page == "Progress Dashboard":
        display_progress_dashboard()

def display_home_page():
    """Display the home page with overview"""
    st.header("Welcome to AI Engineering Academy! 🎓")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Your Adaptive Learning Journey
        
        Our AI Tutor adapts to your learning style and pace, providing:
        
        - **Personalized Assessment**: Global diagnostic quiz to determine your level
        - **Adaptive Content**: Multiple formats (video, text, interactive) based on your preferences
        - **Smart Quizzes**: Intelligent feedback and adaptive difficulty
        - **Explorer Mode**: Skip ahead if you're advanced, with safety nets
        - **24/7 Support**: Always-available conversational AI tutor
        - **Progress Tracking**: Monitor your learning journey
        
        ### Core Features:
        - 🧠 **Intelligent Adaptation**: Content adjusts to your skill level
        - 📊 **Comprehensive Assessment**: Both global and module-specific quizzes
        - 💬 **Conversational Support**: Ask questions anytime
        - 🎯 **Hands-On Learning**: Practical activities in every module
        - 📈 **Progress Monitoring**: Track your improvement over time
        """)
    
    with col2:
        st.info("""
        ### Quick Start:
        1. Complete Global Assessment
        2. Begin Learning Modules
        3. Use AI Tutor for help
        4. Track your progress
        """)
        
        if not st.session_state.student_profile:
            if st.button("🚀 Start Global Assessment", type="primary"):
                st.rerun()

def display_global_assessment(llm):
    """Display and handle the global diagnostic quiz"""
    st.header("🔍 Global Diagnostic Assessment")
    
    if st.session_state.student_profile:
        st.success(f"✅ Assessment completed! Your level: **{st.session_state.student_profile['level'].capitalize()}**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Score", f"{st.session_state.student_profile['score']:.1f}%")
        with col2:
            st.metric("Correct Answers", f"{st.session_state.student_profile['correct']}/{st.session_state.student_profile['total']}")
        
        if st.button("🔄 Retake Assessment"):
            st.session_state.student_profile = None
            st.rerun()
        return
    
    st.markdown("""
    This comprehensive assessment will evaluate your current knowledge in AI and programming.
    Your results will help us personalize your learning experience.
    
    **Instructions:**
    - Answer all questions to the best of your ability
    - Don't worry about getting everything right - this helps us understand your starting point
    - Take your time and think through each question
    """)
    
    with st.form("global_quiz"):
        answers = []
        
        for i, q in enumerate(GLOBAL_QUIZ):
            st.subheader(f"Question {i+1}")
            st.write(q["question"])
            
            answer = st.radio(
                f"Select your answer for Question {i+1}:",
                options=range(len(q["options"])),
                format_func=lambda x, opts=q["options"]: opts[x],
                key=f"q_{i}"
            )
            answers.append(answer)
        
        submitted = st.form_submit_button("📝 Submit Assessment", type="primary")
        
        if submitted:
            # Calculate results
            correct_answers = [1 if answers[i] == q["correct"] else 0 for i, q in enumerate(GLOBAL_QUIZ)]
            total_correct = sum(correct_answers)
            score_percentage = (total_correct / len(GLOBAL_QUIZ)) * 100
            level = calculate_student_level(correct_answers)
            
            # Store student profile
            st.session_state.student_profile = {
                "level": level,
                "score": score_percentage,
                "correct": total_correct,
                "total": len(GLOBAL_QUIZ),
                "timestamp": datetime.now(),
                "quiz_results": correct_answers
            }
            
            st.success("Assessment completed! Your profile has been created.")
            st.rerun()

def display_learning_modules(llm):
    """Display learning modules with adaptive content"""
    st.header("📚 Learning Modules")
    
    if not st.session_state.student_profile:
        st.warning("Please complete the Global Assessment first to personalize your learning experience.")
        return
    
    student_level = st.session_state.student_profile['level']
    
    # Module selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Available Modules")
        for i, module in enumerate(MODULES):
            progress = st.session_state.progress_tracking.get(module['id'], {})
            completion_status = "✅" if progress.get('completed', False) else "⏳"
            
            if st.button(f"{completion_status} {module['title']}", key=f"module_{i}"):
                st.session_state.current_module = module['id']
    
    with col2:
        if st.session_state.current_module > 0:
            current_module = MODULES[st.session_state.current_module - 1]
            display_module_content(current_module, student_level, llm)

def display_module_content(module, student_level, llm):
    """Display content for a specific module"""
    st.subheader(f"📖 {module['title']}")
    st.write(module['description'])
    
    # Module entry quiz
    if f"module_{module['id']}_entry" not in st.session_state.progress_tracking:
        st.info("Before starting this module, let's assess your current knowledge:")
        
        entry_quiz = generate_module_quiz(module['id'], student_level)
        
        with st.form(f"entry_quiz_{module['id']}"):
            st.write("**Quick Self-Assessment:**")
            
            answers = []
            for i, q in enumerate(entry_quiz):
                answer = st.select_slider(
                    q['question'],
                    options=q['options'],
                    key=f"entry_q_{i}"
                )
                answers.append(answer)
            
            if st.form_submit_button("Continue to Module"):
                st.session_state.progress_tracking[module['id']] = {
                    'entry_completed': True,
                    'entry_answers': answers,
                    'progress': 25
                }
                st.rerun()
        return
    
    # Content delivery based on student level and preferences
    tabs = st.tabs(["📺 Video", "📝 Text", "💻 Interactive", "🧪 Hands-On"])
    
    with tabs[0]:  # Video
        st.markdown(f"""
        ### Video Content - {module['title']}
        
        *Video content would be embedded here*
        
        **For {student_level} level:**
        - {'Basic concepts with step-by-step explanations' if student_level == 'beginner' else 'Detailed technical content' if student_level == 'intermediate' else 'Advanced applications and case studies'}
        """)
        
        if st.button("▶️ Mark Video as Watched"):
            update_progress(module['id'], 'video_watched', True)
    
    with tabs[1]:  # Text
        if student_level == "beginner":
            content = f"""
            ### {module['title']} - Beginner Guide
            
            Let's start with the basics! {module['title']} is an important concept in AI that helps us...
            
            **Key Points:**
            - Simple explanation of core concepts
            - Real-world examples you can relate to
            - Step-by-step breakdown of processes
            
            **Remember:** Take your time to understand each concept before moving on!
            """
        elif student_level == "intermediate":
            content = f"""
            ### {module['title']} - Intermediate Deep Dive
            
            Building on your existing knowledge, let's explore {module['title']} in more detail...
            
            **Advanced Concepts:**
            - Technical implementation details
            - Common challenges and solutions
            - Best practices and optimization techniques
            
            **Next Steps:** Try to connect these concepts with what you already know!
            """
        else:
            content = f"""
            ### {module['title']} - Advanced Applications
            
            {module['title']} at an advanced level involves sophisticated techniques and cutting-edge research...
            
            **Expert Topics:**
            - Latest research developments
            - Complex implementation strategies
            - Industry applications and case studies
            
            **Challenge:** Consider how you might apply these concepts in your own projects!
            """
        
        st.markdown(content)
        
        if st.button("📚 Mark Reading as Complete"):
            update_progress(module['id'], 'reading_complete', True)
    
    with tabs[2]:  # Interactive
        st.markdown("### Interactive Learning Experience")
        
        # Simple interactive elements
        col1, col2 = st.columns(2)
        
        with col1:
            user_input = st.text_area(
                "Reflect on what you've learned:",
                placeholder="Write your thoughts about this module..."
            )
        
        with col2:
            if user_input and llm:
                try:
                    chain = create_ai_tutor_chain(llm, student_level)
                    response = chain.predict(
                        context=f"Student is reflecting on {module['title']}",
                        input=user_input,
                        chat_history=""
                    )
                    st.write("**AI Tutor Feedback:**")
                    st.write(response)
                except Exception as e:
                    st.error(f"Error getting AI feedback: {str(e)}")
                    st.write("Please check your API key configuration.")
        
        if st.button("💡 Complete Reflection"):
            update_progress(module['id'], 'reflection_complete', True)
    
    with tabs[3]:  # Hands-On
        st.markdown("### Hands-On Activity")
        
        if "code" in module['content_formats']:
            st.code(f"""
# {module['title']} - Practical Exercise
# Complete this code based on what you've learned

def solve_{module['title'].lower().replace(' ', '_')}_problem():
    # Your code here
    pass

# Test your solution
result = solve_{module['title'].lower().replace(' ', '_')}_problem()
print(f"Result: {{result}}")
            """, language="python")
            
            user_code = st.text_area(
                "Write your solution:",
                placeholder="Enter your Python code here..."
            )
            
            if st.button("🧪 Test Solution") and user_code:
                st.success("Great job! Your solution shows understanding of the concepts.")
                update_progress(module['id'], 'hands_on_complete', True)
        else:
            st.write("**Practical Assignment:**")
            st.write(f"Apply the concepts from {module['title']} to a real-world scenario of your choice.")
            
            assignment = st.text_area(
                "Describe your application:",
                placeholder="How would you apply these concepts in practice?"
            )
            
            if st.button("📋 Submit Assignment") and assignment:
                update_progress(module['id'], 'assignment_complete', True)
    
    # Module completion quiz
    progress = st.session_state.progress_tracking.get(module['id'], {})
    if progress.get('progress', 0) >= 75:
        st.markdown("---")
        st.subheader("🎯 Module Completion Quiz")
        
        if st.button("Take Final Quiz", type="primary"):
            # Simulate quiz completion
            import random
            score = random.randint(70, 95)
            st.session_state.progress_tracking[module['id']].update({
                'completed': True,
                'score': score,
                'progress': 100
            })
            st.success(f"Module completed! Your score: {score}%")
            st.rerun()

def update_progress(module_id, activity, status):
    """Update student progress for a module"""
    if module_id not in st.session_state.progress_tracking:
        st.session_state.progress_tracking[module_id] = {'progress': 0}
    
    st.session_state.progress_tracking[module_id][activity] = status
    
    # Calculate progress percentage
    activities = ['video_watched', 'reading_complete', 'reflection_complete', 'hands_on_complete']
    completed = sum([1 for act in activities if st.session_state.progress_tracking[module_id].get(act, False)])
    progress = min(75, (completed / len(activities)) * 75)  # Max 75% before final quiz
    
    st.session_state.progress_tracking[module_id]['progress'] = progress
    st.success(f"Progress updated! ({progress:.0f}% complete)")

def get_current_user_context():
    """Get detailed current user context"""
    context = {
        'student_level': st.session_state.student_profile['level'] if st.session_state.student_profile else 'Unknown',
        'current_module': None,
        'current_activities': [],
        'progress_summary': {},
        'completed_activities': [],
        'next_activities': []
    }
    
    # Get current module information
    if st.session_state.current_module > 0:
        current_module = MODULES[st.session_state.current_module - 1]
        context['current_module'] = current_module
        
        # Get progress for current module
        module_progress = st.session_state.progress_tracking.get(st.session_state.current_module, {})
        
        # Determine completed activities
        activities = ['video_watched', 'reading_complete', 'reflection_complete', 'hands_on_complete']
        completed = [act for act in activities if module_progress.get(act, False)]
        remaining = [act for act in activities if not module_progress.get(act, False)]
        
        context['completed_activities'] = completed
        context['next_activities'] = remaining
        context['progress_summary'] = module_progress
    
    return context

def display_user_context_sidebar():
    """Display current user context in sidebar"""
    user_context = get_current_user_context()
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📍 Current Context")
    
    # Student level
    level_emoji = {"beginner": "🌱", "intermediate": "📚", "advanced": "🚀"}
    level = user_context['student_level']
    st.sidebar.markdown(f"**Level:** {level_emoji.get(level, '📖')} {level.capitalize()}")
    
    # Current module
    if user_context['current_module']:
        module = user_context['current_module']
        st.sidebar.markdown(f"**Current Module:**")
        st.sidebar.markdown(f"📖 {module['title']}")
        
        # Progress bar
        progress = user_context['progress_summary'].get('progress', 0)
        st.sidebar.progress(progress / 100)
        st.sidebar.markdown(f"Progress: {progress:.0f}%")
        
        # Completed activities
        if user_context['completed_activities']:
            st.sidebar.markdown("**✅ Completed:**")
            for activity in user_context['completed_activities']:
                activity_names = {
                    'video_watched': '📺 Video',
                    'reading_complete': '📝 Reading',
                    'reflection_complete': '💭 Reflection',
                    'hands_on_complete': '💻 Hands-On'
                }
                st.sidebar.markdown(f"• {activity_names.get(activity, activity)}")
        
        # Next activities
        if user_context['next_activities']:
            st.sidebar.markdown("**⏳ Next Steps:**")
            for activity in user_context['next_activities']:
                activity_names = {
                    'video_watched': '📺 Watch Video',
                    'reading_complete': '📝 Complete Reading',
                    'reflection_complete': '💭 Write Reflection',
                    'hands_on_complete': '💻 Do Hands-On Activity'
                }
                st.sidebar.markdown(f"• {activity_names.get(activity, activity)}")
    else:
        st.sidebar.markdown("**Current Module:** None selected")
        st.sidebar.markdown("💡 *Select a module to start learning*")
    
    # Overall progress
    total_modules = len(MODULES)
    completed_modules = len([m for m in st.session_state.progress_tracking.values() if m.get('completed', False)])
    st.sidebar.markdown(f"**Overall Progress:** {completed_modules}/{total_modules} modules")

def display_ai_tutor_chat(llm):
    """Display conversational AI tutor interface"""
    st.header("💬 AI Tutor Chat")
    
    if not llm:
        st.error("Please configure your Gemini API key to use the AI Tutor.")
        return
    
    if not st.session_state.student_profile:
        st.warning("Complete the Global Assessment first for personalized tutoring.")
        return
    
    # Display user context sidebar
    display_user_context_sidebar()
    
    # Main chat interface with context-aware layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        student_level = st.session_state.student_profile['level']
        user_context = get_current_user_context()
        
        st.markdown(f"""
        **Your AI Tutor is ready to help!** 
        
        I can see you're currently working on: **{user_context['current_module']['title'] if user_context['current_module'] else 'No module selected'}**
        
        Ask me anything about:
        - Your current module and activities
        - Course concepts and explanations
        - Help with assignments
        - Study strategies
        - Career guidance in AI
        """)
        
        # Display conversation history
        for message in st.session_state.conversation_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask your AI tutor anything..."):
            # Add user message to history
            st.session_state.conversation_history.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate AI response with detailed context
            try:
                chain = create_ai_tutor_chain(llm, student_level)
                
                # Create detailed context for AI
                detailed_context = f"""
Student Level: {student_level}
Current Module: {user_context['current_module']['title'] if user_context['current_module'] else 'None'}
Module Description: {user_context['current_module']['description'] if user_context['current_module'] else 'N/A'}
Module Progress: {user_context['progress_summary'].get('progress', 0)}%
Completed Activities: {', '.join(user_context['completed_activities']) if user_context['completed_activities'] else 'None'}
Next Activities: {', '.join(user_context['next_activities']) if user_context['next_activities'] else 'None'}
Overall Progress: {len([m for m in st.session_state.progress_tracking.values() if m.get('completed', False)])}/{len(MODULES)} modules completed
"""
                
                response = chain.predict(
                    context=detailed_context, 
                    input=prompt,
                    chat_history=""
                )
                
                # Add AI response to history
                st.session_state.conversation_history.append({"role": "assistant", "content": response})
                
                with st.chat_message("assistant"):
                    st.write(response)
                    
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                st.write("Please check your API key and try again.")
    
    with col2:
        st.subheader("🎯 Context-Aware Actions")
        
        user_context = get_current_user_context()
        
        # Current module specific actions
        if user_context['current_module']:
            module = user_context['current_module']
            st.markdown(f"**Current: {module['title']}**")
            
            if st.button("❓ Explain Current Module", key="explain_current"):
                explanation_prompt = f"Please explain {module['title']} - {module['description']} in detail for my level"
                st.session_state.conversation_history.append({"role": "user", "content": explanation_prompt})
                st.rerun()
            
            # Activity-specific help
            if user_context['next_activities']:
                st.markdown("**Next Activity Help:**")
                for activity in user_context['next_activities'][:2]:  # Show max 2
                    activity_names = {
                        'video_watched': 'Video Help',
                        'reading_complete': 'Reading Help',
                        'reflection_complete': 'Reflection Help',
                        'hands_on_complete': 'Hands-On Help'
                    }
                    if st.button(f"💡 {activity_names.get(activity, activity)}", key=f"help_{activity}"):
                        help_prompt = f"I need help with the {activity.replace('_', ' ')} activity for {module['title']}. What should I focus on?"
                        st.session_state.conversation_history.append({"role": "user", "content": help_prompt})
                        st.rerun()
        
        st.markdown("---")
        
        # General quick actions
        st.markdown("**Quick Actions:**")
        
        if st.button("📚 Study Tips", key="study_tips"):
            tips_prompt = f"Give me specific study tips for {student_level} level students learning AI"
            st.session_state.conversation_history.append({"role": "user", "content": tips_prompt})
            st.rerun()
        
        if st.button("🎯 What's Next?", key="whats_next"):
            next_prompt = "Based on my current progress, what should I focus on next?"
            st.session_state.conversation_history.append({"role": "user", "content": next_prompt})
            st.rerun()
        
        if st.button("📊 Progress Review", key="progress_review"):
            review_prompt = "Can you review my learning progress and give me feedback?"
            st.session_state.conversation_history.append({"role": "user", "content": review_prompt})
            st.rerun()
        
        if st.button("🚀 Career Guidance", key="career_guidance"):
            career_prompt = "What career paths are available in AI engineering for someone at my level?"
            st.session_state.conversation_history.append({"role": "user", "content": career_prompt})
            st.rerun()
        
        # Learning statistics
        st.markdown("---")
        st.markdown("**📈 Quick Stats:**")
        completed_modules = len([m for m in st.session_state.progress_tracking.values() if m.get('completed', False)])
        st.metric("Modules Completed", f"{completed_modules}/{len(MODULES)}")
        
        if user_context['current_module']:
            current_progress = user_context['progress_summary'].get('progress', 0)
            st.metric("Current Module", f"{current_progress:.0f}%")

if __name__ == "__main__":
    main()
