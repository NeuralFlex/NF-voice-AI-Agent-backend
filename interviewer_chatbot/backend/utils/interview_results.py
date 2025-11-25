import requests
import json
import os
from utils.logger import setup_logger

logger = setup_logger(__name__)


def render_interview_results(state: dict, destination: str = "both"):
    """
    Render interview results to console, Slack, or both.

    Args:
        state (dict): Interview state containing questions, answers, feedback, final_evaluation.
        destination (str): "console", "slack", or "both"
    """
    user_id = state.get("user_id", "unknown_user")
    topic = state.get("topic", "unknown_topic")

    questions = state.get("questions", [])
    answers = state.get("answers", [])
    feedback_list = state.get("feedback", [])
    final_eval = state.get("final_evaluation", {})

    qna_section = ""
    for i, (q, a) in enumerate(zip(questions, answers)):
        fb = feedback_list[i] if i < len(feedback_list) else {}
        q_fb = fb.get("question_feedback", {}).get("feedback", "No feedback")
        a_fb = fb.get("answer_feedback", {}).get("feedback", "No feedback")
        qna_section += f"Q{i+1}: {q}\nA{i+1}: {a}\nQuestion Feedback: {q_fb}\nAnswer Feedback: {a_fb}\n\n"

    final_section = ""
    if final_eval:
        final_section += (
            f"Overall Quality: {final_eval.get('overall_quality', 'N/A')}\n"
        )
        final_section += f"Recommendation: {final_eval.get('recommendation', 'N/A')}\n"
        final_section += (
            "Strengths:\n"
            + "\n".join([f" - {s}" for s in final_eval.get("strengths", [])])
            + "\n"
        )
        final_section += (
            "Areas for Improvement:\n"
            + "\n".join(
                [f" - {a}" for a in final_eval.get("areas_for_improvement", [])]
            )
            + "\n"
        )
        final_section += f"Final Feedback: {final_eval.get('final_feedback', '')}\n"

    if destination in ("console", "both"):
        print(f"\n📋 Interview Results for {user_id} on {topic}:\n")
        print("-" * 70)
        print(qna_section)
        print(final_section)
        print("=" * 70)

    if destination in ("slack", "both"):
        webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        if not webhook_url:
            logger.error("SLACK_WEBHOOK_URL not set. Skipping Slack send.")
        else:
            text = f"*📋 Interview Results for {user_id} on {topic}:*\n\n{qna_section}\n{final_section}"
            payload = {"text": text}
            try:
                response = requests.post(
                    webhook_url,
                    data=json.dumps(payload),
                    headers={"Content-Type": "application/json"},
                )
                if response.status_code == 200:
                    logger.info("✅ Slack message sent successfully.")
                else:
                    logger.error(
                        f"Slack message failed: {response.status_code} - {response.text}"
                    )
            except Exception as e:
                logger.error(f"Slack send failed: {e}")
