import streamlit as st
import requests
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL")

st.set_page_config(page_title="AI Interviewer", page_icon="brain")

BACKEND_START_ENDPOINT = f"{BACKEND_URL}/start_interview"
BACKEND_CONTINUE_ENDPOINT = f"{BACKEND_URL}/continue_interview"

for key in [
    "interview_started",
    "messages",
    "thread_id",
    "show_report",
    "final_report",
    "candidate_name",
    "job_title",
    "cv_filename",
    "question_style",
    "qa_pairs",
]:
    if key not in st.session_state:
        st.session_state[key] = (
            False
            if key != "messages" and key != "final_report" and key != "qa_pairs"
            else [] if key in ["messages", "qa_pairs"] else ""
        )

if st.session_state.show_report:
    st.title(f"Final Report: {st.session_state.job_title}")

    st.markdown(
        f"""
    **Candidate:** {st.session_state.candidate_name or 'Not Provided'}  
    **Position:** {st.session_state.job_title}  
    **CV:** {st.session_state.cv_filename or 'Not uploaded'}  
    **Question Style:** {st.session_state.question_style}  
    **Date:** {datetime.now().strftime('%B %d, %Y')}
    """
    )
    st.divider()

    st.markdown(st.session_state.final_report, unsafe_allow_html=True)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start New Interview"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
    with col2:
        if st.download_button(
            label="Download Report (Markdown)",
            data=st.session_state.final_report,
            file_name=f"report_{st.session_state.candidate_name}_{st.session_state.job_title}.md".replace(
                " ", "_"
            ),
            mime="text/markdown",
        ):
            st.success("Downloaded!")

elif not st.session_state.interview_started:
    st.title("AI Interviewer Setup")

    st.session_state.candidate_name = st.text_input(
        "Your Name", placeholder="e.g. Muhammad"
    )
    st.session_state.job_title = st.text_input(
        "Job Title", placeholder="e.g. JS Intern"
    )
    cv_file = st.file_uploader("Upload CV (required)", type=["pdf"])

    st.subheader("Choose Question Style")
    style = st.radio(
        "Question Style",
        [
            "1. Broad, follow-up",
            "2. Narrow, follow-up",
            "3. Broad, non-follow-up",
            "4. Narrow, non-follow-up",
        ],
    )
    qmap = {
        "1. Broad, follow-up": "broad_followup",
        "2. Narrow, follow-up": "narrow_followup",
        "3. Broad, non-follow-up": "broad_nonfollowup",
        "4. Narrow, non-follow-up": "narrow_nonfollowup",
    }
    style_display = style.split(". ", 1)[1] if ". " in style else style

    if st.button("Start Interview"):
        if not st.session_state.job_title:
            st.warning("Enter job title")
        elif not cv_file:
            st.warning("Upload your CV to continue")
        else:
            cv_name = cv_file.name
            files = {"cv": ("cv.pdf", cv_file, "application/pdf")}
            data = {
                "job_title": st.session_state.job_title,
                "question_type": qmap[style],
            }

            try:
                r = requests.post(BACKEND_START_ENDPOINT, data=data, files=files)
                if r.status_code == 200:
                    d = r.json()
                    st.session_state.interview_started = True
                    st.session_state.show_report = False
                    st.session_state.thread_id = d.get("thread_id")
                    st.session_state.cv_filename = cv_name
                    st.session_state.question_style = style_display
                    st.session_state.messages = [
                        {
                            "role": "assistant",
                            "content": d.get("message", "Let's begin!"),
                        }
                    ]
                    st.session_state.qa_pairs = []
                    st.rerun()
                else:
                    st.error(f"Error: {r.status_code}")
            except Exception as e:
                st.error(f"Backend error: {e}")

else:
    st.title(f"Interview: {st.session_state.job_title}")

    st.info(
        f"""
    **Candidate:** {st.session_state.candidate_name or 'Not set'}  
    **CV:** {st.session_state.cv_filename}  
    **Question Style:** {st.session_state.question_style}
    """
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Type your answer..."):
        last_q = None
        for m in reversed(st.session_state.messages):
            if m["role"] == "assistant" and m["content"] != "*Thinking...*":
                last_q = m["content"]
                break
        if last_q:
            st.session_state.qa_pairs.append(
                {"question": last_q.strip(), "answer": user_input.strip()}
            )

        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append(
            {"role": "assistant", "content": "*Thinking...*"}
        )
        st.rerun()

    elif (
        st.session_state.messages
        and st.session_state.messages[-1]["content"] == "*Thinking...*"
    ):
        user_msg = [
            m["content"]
            for m in reversed(st.session_state.messages[:-1])
            if m["role"] == "user"
        ][0]

        try:
            resp = requests.post(
                BACKEND_CONTINUE_ENDPOINT,
                json={
                    "user_response": user_msg,
                    "thread_id": st.session_state.thread_id,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                reply = ""

                feedback_list = data.get("feedback_list", [])
                if feedback_list:
                    fb = feedback_list[-1]
                    qf = fb.get("question_feedback", {})
                    af = fb.get("answer_feedback", {})

                    reply += f"### Feedback for Your Last Answer\n"
                    reply += f"**Question Rating:** {qf.get('rating', 'N/A')}\n"
                    qf_dict = qf.get("feedback", {})
                    if isinstance(qf_dict, dict):
                        for k, v in qf_dict.items():
                            reply += f"- **{k.title()}**: {v}\n"
                    else:
                        reply += f"{qf_dict}\n"
                    reply += "\n"
                    reply += f"**Your Answer Rating:** {af.get('rating', 'N/A')}\n"
                    reply += f"{af.get('feedback', 'No feedback')}\n\n"
                    reply += "---\n"

                if msg := data.get("message"):
                    reply += f"{msg}\n\n"

                if final := data.get("final_evaluation"):
                    report = f"# Final Interview Report\n\n"
                    report += f"**Candidate:** {st.session_state.candidate_name or 'Not Provided'}\n"
                    report += f"**Position:** {st.session_state.job_title}\n"
                    report += f"**CV:** {st.session_state.cv_filename}\n"
                    report += f"**Question Style:** {st.session_state.question_style}\n"
                    report += f"**Date:** {datetime.now().strftime('%B %d, %Y')}\n\n"
                    report += "---\n\n"

                    report += "## Interview Transcript & Feedback\n\n"
                    for i, pair in enumerate(st.session_state.qa_pairs, 1):
                        fb = feedback_list[i - 1] if i <= len(feedback_list) else {}
                        qf = fb.get("question_feedback", {})
                        af = fb.get("answer_feedback", {})

                        report += f"### Question {i}\n"
                        report += f"**Q:** {pair['question']}\n\n"
                        report += f"**A:** {pair['answer']}\n\n"

                        report += f"**Question Rating:** {qf.get('rating', 'N/A')}\n"
                        qf_dict = qf.get("feedback", {})
                        if isinstance(qf_dict, dict):
                            for k, v in qf_dict.items():
                                report += f"- **{k.title()}**: {v}\n"
                        else:
                            report += f"{qf_dict}\n"
                        report += "\n"

                        report += f"**Your Answer Rating:** {af.get('rating', 'N/A')}\n"
                        report += f"{af.get('feedback', 'No feedback')}\n\n"
                        report += "---\n\n"

                    report += "## Final Evaluation\n\n"
                    report += f"**Overall Quality:** {final.get('overall_quality', 'N/A')}\n\n"
                    report += f"**Strengths:**\n"
                    for s in final.get("strengths", []):
                        report += f"- {s}\n"
                    report += "\n"
                    report += f"**Areas for Improvement:**\n"
                    for a in final.get("areas_for_improvement", []):
                        report += f"- {a}\n"
                    report += "\n"
                    report += (
                        f"**Recommendation:** {final.get('recommendation', 'N/A')}\n\n"
                    )
                    report += (
                        f"**Final Feedback:**\n\n{final.get('final_feedback', 'N/A')}\n"
                    )

                    st.session_state.final_report = report
                    st.session_state.show_report = True
                    reply += "\n**Interview complete. View your full report.**"

                st.session_state.messages.pop()
                st.session_state.messages.append(
                    {"role": "assistant", "content": reply}
                )
            else:
                st.session_state.messages.pop()
                st.session_state.messages.append(
                    {"role": "assistant", "content": "Backend error"}
                )
        except Exception as e:
            st.session_state.messages.pop()
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Error: {e}"}
            )
        st.rerun()
