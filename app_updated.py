import streamlit as st
from PyPDF2 import PdfReader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from fpdf import FPDF

# Import the 3 different LLM providers
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic


# 1. Define the structured output format using Pydantic
class ResumeAnalysis(BaseModel):
    match_percentage: int = Field(
        description="ATS match score from 0 to 100 based on the job description"
    )
    matched_skills: list[str] = Field(
        description="Keywords and skills found in the resume that match the job description"
    )
    missing_skills: list[str] = Field(
        description="Important keywords and skills from the job description missing in the resume"
    )
    recommendations: list[str] = Field(
        description="3 actionable bullet points to improve the resume for this job role"
    )


# Custom PDF Class to handle standard formatting
class ResumePDF(FPDF):
    def header(self):
        pass  # Empty header to allow more space for the resume content

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


# Streamlit page configuration
st.set_page_config(
    page_title="Multi-Model AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume Analyzer & Generator")
st.write("Upload your resume and a job description to check your ATS compatibility and automatically generate an optimized PDF resume if needed.")


# Sidebar for Provider Selection and API Key input
with st.sidebar:
    st.header("Configuration")
    
    # Dropdown to select the AI Provider
    provider = st.selectbox(
        "Select AI Provider",
        ("Google Gemini", "OpenAI", "Anthropic Claude")
    )
    
    # Dynamic API Key Input
    api_key = st.text_input(
        f"Enter your {provider} API Key",
        type="password"
    )
    
    # Dynamic help links based on selected provider
    if provider == "Google Gemini":
        st.markdown("[Get your free Gemini API key here](https://aistudio.google.com/app/apikey)")
    elif provider == "OpenAI":
        st.markdown("[Get your OpenAI API key here](https://platform.openai.com/api-keys)")
    elif provider == "Anthropic Claude":
        st.markdown("[Get your Claude API key here](https://console.anthropic.com/settings/keys)")


# Main Interface: Two-column layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload Resume")
    uploaded_file = st.file_uploader(
        "Upload your Resume (PDF format only)",
        type=["pdf"]
    )

with col2:
    st.subheader("2. Job Description")
    job_description = st.text_area(
        "Paste the Target Job Description Here",
        height=200
    )


# Process analysis when button is clicked
if st.button("Analyze Resume", type="primary"):
    if not api_key:
        st.error(f"Please enter your {provider} API Key in the sidebar.")
    elif not uploaded_file:
        st.error("Please upload a resume PDF.")
    elif not job_description.strip():
        st.error("Please paste a job description.")
    else:
        with st.spinner(f"Extracting text and analyzing with {provider}..."):
            try:
                # Extract text from PDF
                reader = PdfReader(uploaded_file)
                resume_text = ""

                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        resume_text += text + "\n"

                if not resume_text.strip():
                    st.error(
                        "Could not extract text from this PDF. "
                        "Please upload a text-based PDF."
                    )
                    st.stop()

                # --- DYNAMIC MODEL INITIALIZATION ---
                if provider == "Google Gemini":
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-3.5-flash",
                        google_api_key=api_key,
                        temperature=0.2
                    )
                elif provider == "OpenAI":
                    llm = ChatOpenAI(
                        model="gpt-4o-mini",
                        api_key=api_key,
                        temperature=0.2
                    )
                elif provider == "Anthropic Claude":
                    llm = ChatAnthropic(
                        model_name="claude-3-5-sonnet-latest",
                        anthropic_api_key=api_key,
                        temperature=0.2
                    )

                # Structured JSON parser for evaluation
                parser = JsonOutputParser(
                    pydantic_object=ResumeAnalysis
                )

                # Prompt for ATS analysis
                prompt = ChatPromptTemplate.from_messages([
                    (
                        "system",
                        "You are an expert ATS (Applicant Tracking System) "
                        "optimizer. Compare the candidate's resume text against "
                        "the provided job description. Provide an unbiased "
                        "assessment. Return output strictly in JSON format "
                        "matching the schema instructions.\n{format_instructions}"
                    ),
                    (
                        "human",
                        "RESUME:\n{resume}\n\nJOB DESCRIPTION:\n{job_description}"
                    )
                ])

                # Chain: Prompt -> Selected LLM -> JSON Parser
                chain = prompt | llm | parser

                # Execute analysis
                result = chain.invoke({
                    "resume": resume_text,
                    "job_description": job_description,
                    "format_instructions": parser.get_format_instructions()
                })

                # Render results
                st.success(f"Analysis Complete (Powered by {provider})!")
                st.divider()

                score = result.get("match_percentage", 0)

                if score >= 80:
                    st.balloons()
                    st.metric(
                        label="ATS Match Score",
                        value=f"{score}%",
                        delta="Strong Match (>= 80%)"
                    )
                elif score >= 50:
                    st.metric(
                        label="ATS Match Score",
                        value=f"{score}%",
                        delta="Needs Improvement (< 80%)",
                        delta_color="off"
                    )
                else:
                    st.metric(
                        label="ATS Match Score",
                        value=f"{score}%",
                        delta="Weak Match (< 80%)",
                        delta_color="inverse"
                    )

                res_col1, res_col2 = st.columns(2)

                with res_col1:
                    st.subheader("✅ Matched Skills & Keywords")
                    for skill in result.get("matched_skills", []):
                        st.markdown(f"- {skill}")

                with res_col2:
                    st.subheader("❌ Missing Critical Keywords")
                    for skill in result.get("missing_skills", []):
                        st.markdown(
                            f"- <span style='color:#ff4b4b'>"
                            f"**{skill}**</span>",
                            unsafe_allow_html=True
                        )

                st.subheader("💡 Recommendations to Optimize Your Resume")
                for rec in result.get("recommendations", []):
                    st.markdown(f"* {rec}")

                # Check score threshold to trigger tailored resume generation
                if score < 80:
                    st.divider()
                    st.warning(
                        f"Your ATS match score is **{score}%**. "
                        f"Automatically generating a fully tailored PDF resume using {provider}..."
                    )

                    with st.spinner(f"Analyzing original structure and generating tailored PDF with {provider}..."):
                        # Prompt for HTML generation mirroring original structure
                        gen_prompt = ChatPromptTemplate.from_messages([
                            (
                                "system",
                                "You are an expert resume writer and ATS specialist. "
                                "Your task is to rewrite the candidate's original resume to perfectly align with the target job description while STRICTLY maintaining the original resume's structural format. "
                                "Deduce the order of sections (e.g., Summary, Experience, Education), the style of headers, and how bullet points were used in the ORIGINAL RESUME, and mimic that structure exactly. "
                                "Integrate the missing skills seamlessly into experience bullets and summaries. Keep facts accurate. "
                                "OUTPUT INSTRUCTIONS: Output ONLY clean, standard HTML code using basic tags (<h1>, <h2>, <h3>, <p>, <ul>, <li>, <b>, <strong>, <br>). "
                                "Do not include any CSS styling, do not use Markdown, and do NOT wrap the output in ```html code blocks. Output purely the raw HTML."
                            ),
                            (
                                "human",
                                "ORIGINAL RESUME (Mimic this structure):\n{resume}\n\n"
                                "JOB DESCRIPTION:\n{job_description}\n\n"
                                "MISSING KEYWORDS TO INCLUDE:\n{missing_skills}"
                            )
                        ])

                        gen_chain = gen_prompt | llm | StrOutputParser()

                        raw_html_output = gen_chain.invoke({
                            "resume": resume_text,
                            "job_description": job_description,
                            "missing_skills": ", ".join(result.get("missing_skills", []))
                        })

                        # Clean output just in case the LLM includes code block wrappers
                        clean_html = raw_html_output.replace("```html", "").replace("```", "").strip()

                        # Sanitize Unicode characters for standard PDF font rendering
                        unicode_replacements = {
                            '–': '-', '—': '-',   # En and Em dashes
                            '“': '"', '”': '"',   # Smart double quotes
                            '‘': "'", '’': "'",   # Smart single quotes
                            '•': '-',             # Bullet points
                            '…': '...',           # Ellipsis
                            '\u2028': '\n',       # Line separator
                            '\u2029': '\n',       # Paragraph separator
                            '\xa0': ' '           # Non-breaking space
                        }
                        
                        for char, replacement in unicode_replacements.items():
                            clean_html = clean_html.replace(char, replacement)

                        try:
                            # Generate PDF from the HTML content
                            pdf = ResumePDF()
                            pdf.add_page()
                            # Set default font that renders cleanly
                            pdf.set_font("helvetica", size=11)
                            
                            # Write HTML to PDF
                            pdf.write_html(clean_html)
                            
                            # Output bytes
                            pdf_bytes = bytes(pdf.output())

                            st.success("✅ Tailored PDF Resume Generated Successfully!")
                            
                            # Provide download button for the new PDF
                            st.download_button(
                                label=f"📥 Download Tailored Resume (.pdf)",
                                data=pdf_bytes,
                                file_name="Tailored_ATS_Resume.pdf",
                                mime="application/pdf",
                                type="primary",
                                use_container_width=True
                            )
                            
                        except Exception as pdf_error:
                            st.error(f"Failed to compile PDF structure: {str(pdf_error)}")
                            st.info("The generated structure may have contained complex elements. Try analyzing again.")

            except Exception as e:
                error_msg = str(e)
                st.error(f"An error occurred: {error_msg}")
                if "429" in error_msg or "quota" in error_msg.lower():
                    st.info("You hit the API rate limit or ran out of credits for the selected provider. Please wait a moment, check your billing, or switch to a different provider.")